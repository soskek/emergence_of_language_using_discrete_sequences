#!/usr/bin/env python
from __future__ import print_function
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers as S
from chainer import training
from chainer.training import extensions
from chainer.dataset.convert import concat_examples as convert

# from links import communicator
from links import world

import numpy as np

try:
    from tqdm import tqdm
except:
    tqdm = list

import json

from PIL import Image


def save_images(x, filename, cifar=False):
    x = np.array(x.tolist(), np.float32)
    width = x.shape[0]
    fig, ax = plt.subplots(1, width, figsize=(width, 1))
    for ai, xi in zip(ax.ravel(), x):
        ai.set_axis_off()
        if cifar:
            # (3, 32, 32) -> (32, 32, 3)
            ai.imshow(np.moveaxis(xi, 0, -1))
        else:
            ai.imshow(xi.reshape(28, 28), cmap='Greys_r')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.1, hspace=0.)
    fig.savefig(filename, bbox_inches='tight', pad=0.)
    plt.clf()
    plt.close('all')


def generate(model, image_data, epoch=0, out='./',
             filename='image', printer=False, cifar=False):
    prev_train = model.train
    model.train = False
    sentence_history, log_prob_history, canvas_history = model(
        image_data, generate=True)
    model.train = prev_train

    save_images(image_data.data, out + filename + '_TRUE.png')
    for i in range(model.n_turn):
        save_images(
            canvas_history[i], out + filename + '_{}e_{}.png'.format(epoch, i))
    sentence_log = [[[[int(word_batch[i]) for word_batch in word_batch_list],
                      float(log_prob_batch[i])]
                     for log_prob_batch, word_batch_list
                     in zip(log_prob_history, sentence_history)]
                    for i in range(image_data.shape[0])]
    json_f = open(
        out + 'message.' + filename + '{}e.json'.format(epoch), 'w')
    json.dump(sentence_log, json_f)


def generate_by_message(model, sentence=None, epoch=0, out='./',
                        filename='mimage', printer=False, cifar=False):
    prev_train = model.train
    model.train = False
    if sentence is None:
        sentence = np.fliplr(1 - np.tri(model.n_word, model.n_word + 1).T)
        sentence = np.concatenate(
            [sentence, np.fliplr(1 - np.tri(model.n_word, model.n_word))], axis=0)
        sentence = [model.xp.array(word, np.int32)
                    for word in sentence.T.tolist()]
    shape = (len(sentence[0]), 32 * 32 * 2 if cifar else 784)
    canvas_history = model.generate(sentence, shape)
    model.train = prev_train

    for i in range(model.n_turn):
        save_images(
            canvas_history[i], out + filename + '_{}e_{}.png'.format(epoch, i))


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=15,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='./result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')

    parser.add_argument('--cifar', type=int, default=0,
                        help='Cifar/MNIST')

    parser.add_argument('--unit', '-u', type=int, default=64,
                        help='Number of units')
    parser.add_argument('--image-unit', '-i', type=int, default=128,
                        help='Number of middel units for image expression')

    parser.add_argument('--co-importance', '-importance', '-imp',
                        type=float, default=0.,
                        help='Coef. of importance loss')
    parser.add_argument('--co-orthogonal', '-orthogonal', '-ort',
                        type=float, default=0.,
                        help='Coef. of orthogonal loss')

    parser.add_argument('--word', '-w', type=int, default=3,
                        help='Number of words in a message')
    parser.add_argument('--turn', '-t', type=int, default=3,
                        help='Number of turns')
    parser.add_argument('--vocab', '-v', type=int, default=32,
                        help='Number of words in vocab')

    parser.add_argument('--drop-ratio', '--dropout', type=float, default=0.1,
                        help='dropout ratio')

    args = parser.parse_args()

    args.out = args.out.rstrip('/') + '/'

    import json
    print(json.dumps(args.__dict__, indent=2))

    print('')

    if args.cifar:
        print('CIFAR')
        model = world.World(
            32 * 32 * 3, args.image_unit, args.unit,
            n_vocab=args.vocab, n_word=args.word, n_turn=args.turn,
            drop_ratio=args.drop_ratio, co_importance=args.co_importance,
            co_orthogonal=args.co_orthogonal, cifar=args.cifar)
        # Load the MNIST dataset
        #train, test = chainer.datasets.get_cifar10(withlabel=False)
        alldata = np.load('svhn_test') / 255
        train, test = alldata[:25000], alldata[25000:]
        print('# of train data:', len(train))
        print('# of test data:', len(test))
    else:
        print('MNIST')
        model = world.World(
            28 * 28, args.image_unit, args.unit,
            n_vocab=args.vocab, n_word=args.word, n_turn=args.turn,
            drop_ratio=args.drop_ratio, co_importance=args.co_importance,
            co_orthogonal=args.co_orthogonal, cifar=args.cifar)
        # Load the MNIST dataset
        train, test = chainer.datasets.get_mnist(withlabel=False)
        print('# of train data:', len(train))
        print('# of test data:', len(test))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(1.))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    import os
    if not os.path.isdir(args.out):
        if os.path.exists(args.out):
            print(args.out, 'exists as a file')
            exit()
        else:
            os.mkdir(args.out)

    batchsize = args.batchsize

    log_report = extensions.LogReport()
    best_valid = 10000000.
    best_keep = 0
    print('start')
    for i_epoch in range(1, args.epoch + 1):
        n_iters = len(train) // batchsize
        permutation = np.random.permutation(len(train))
        accum_loss_data = 0.
        for i_iter in range(n_iters):
            ids = permutation[i_iter * batchsize:
                              (i_iter + 1) * batchsize]
            batch = [train[idx] for idx in ids]
            batch = chainer.Variable(
                convert(batch, device=args.gpu), volatile='auto')
            loss = model(batch)
            model.zerograds()
            loss.backward()
            optimizer.update()
            accum_loss_data += loss.data - model.sub_accum_loss
            del loss

        train_loss = '{:.6f}'.format(float(accum_loss_data / n_iters))
        mean_valid_loss_data = evaluate(model, test, batchsize, args.gpu)

        if mean_valid_loss_data < best_valid:
            best_valid = mean_valid_loss_data
            best_keep = 0
            S.save_npz(args.out + 'saved_model.model', model)
            valid_loss = '\t\t{:.6f} *'.format(float(mean_valid_loss_data))

            batch = chainer.Variable(
                convert(train[:50], device=args.gpu), volatile='auto')
            generate(model, batch, epoch=i_epoch, out=args.out, filename='train',
                     printer=(i_epoch == args.epoch - 1), cifar=args.cifar)
            batch = chainer.Variable(
                convert(test[:50], device=args.gpu), volatile='auto')
            generate(model, batch, epoch=i_epoch, out=args.out, filename='test',
                     printer=(i_epoch == args.epoch - 1), cifar=args.cifar)

            generate_by_message(
                model, epoch=i_epoch, out=args.out, filename='checkinc',
                printer=(i_epoch == args.epoch - 1), cifar=args.cifar)
        else:
            best_keep += 1
            valid_loss = '{:.6f}'.format(float(mean_valid_loss_data))

        print('Epoch', i_epoch,
              '\ttrain: {}\tvalid: {}'.format(train_loss, valid_loss))

        if best_keep >= 20:
            break

    #S.save_npz(args.out + 'saved_model.model', model)
    print('Finish at {}/{} epoch'.format(i_epoch, args.epoch))


def evaluate(model, test, batchsize, gpu):

    model.train = False
    accum_valid_loss_data = 0.

    for i_iter in range(len(test) // batchsize + 1):
        ids = [i_iter * batchsize + idx for idx in range(batchsize)]
        ids = [idx for idx in ids if idx < len(test)]
        batch = [test[idx]
                 for idx in ids if idx < len(test)]
        if not batch:
            continue
        batch = chainer.Variable(
            convert(batch, device=gpu), volatile='auto')
        valid_loss_data = model(batch).data
        valid_loss_data -= model.sub_accum_loss
        accum_valid_loss_data += valid_loss_data * len(ids)

    model.train = True

    return accum_valid_loss_data / len(test)


if __name__ == '__main__':
    main()
