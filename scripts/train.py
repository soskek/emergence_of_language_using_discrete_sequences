#!/usr/bin/env python
from __future__ import print_function
import argparse
import copy
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
import chainer.serializers as S
from chainer.dataset.convert import concat_examples as convert

# from links import communicator
from links import coms

import numpy as np
import json


def save_images(x, filename):
    x = np.array(x.tolist(), np.float32)
    width = x.shape[0]
    fig, ax = plt.subplots(1, width, figsize=(width, 1))
    for ai, xi in zip(ax.ravel(), x):
        ai.set_axis_off()
        ai.imshow(xi.reshape(28, 28), cmap='Greys_r')
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.)
    fig.savefig(filename, bbox_inches='tight', pad=0.)
    plt.clf()
    plt.close('all')


def generate(model, image_data, epoch=0, out='./', filename='image'):
    sentence, log_prob, canvas = model(image_data, generate=True, train=False)

    save_images(image_data.data, out + filename + '_TRUE.png')
    save_images(canvas, out + filename + '_{}e.png'.format(epoch))


def make_template_sentence(model):
    sentence = np.fliplr(1 - np.tri(model.n_word, model.n_word + 1).T)
    sentence = np.concatenate(
        [sentence, np.fliplr(1 - np.tri(model.n_word, model.n_word))], axis=0)
    sentence = [model.xp.array(word, np.int32)
                for word in sentence.T.tolist()]
    return sentence


def generate_by_template(model, epoch=0, out='./',
                         filename='mimage', agent_type='sender'):

    sentence = make_template_sentence(model)

    if agent_type == 'sender':
        canvas = model.generate_from_sentence(sentence)
        save_images(canvas, out + filename + '_{}e.png'.format(epoch))
    else:
        canvas = model.infer_from_sentence(sentence)
        save_images(canvas, out + filename + '_{}e.png'.format(epoch))


def evaluate(model, dataset, batchsize, gpu):
    accum_valid_loss_data = 0.
    dataset_iter = chainer.iterators.SerialIterator(
        dataset, batchsize, repeat=False, shuffle=False)
    dataset_iter.is_new_epoch = False
    while not dataset_iter.is_new_epoch:
        batch = chainer.Variable(
            convert(dataset_iter.next(), device=gpu), volatile='auto')
        valid_loss, valid_rein_loss = model(batch, train=False)
        accum_valid_loss_data += valid_loss.data * batch.shape[0]
    return accum_valid_loss_data / len(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=512)
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--unit', '-u', type=int, default=32)
    parser.add_argument('--image-unit', '-i', type=int, default=256)
    parser.add_argument('--vocab', '-v', type=int, default=2)
    parser.add_argument('--word', '-w', type=int, default=8)

    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='./result',
                        help='Directory to output the result')

    args = parser.parse_args()
    args.out = args.out.rstrip('/') + '/'
    print(json.dumps(args.__dict__, indent=2))
    if not os.path.isdir(args.out):
        os.mkdir(args.out)

    model = coms.World(
        28 * 28, args.image_unit, args.unit,
        n_vocab=args.vocab, n_word=args.word)
    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(withlabel=False)
    train, valid = chainer.datasets.split_dataset_random(
        train, 50000, seed=777)
    train_iter = chainer.iterators.SerialIterator(
        train, args.batchsize)
    print('# of train data:', len(train))
    print('# of valid data:', len(valid))
    print('# of test data:', len(test))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(1.))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))
    model.learn_constraint(train)

    n_iters = len(train) // args.batchsize
    best_valid = 10000000.
    n_est_keep = 0
    print('start')
    for i_epoch in range(1, args.epoch + 1):
        accum_loss_data = 0.
        accum_rein_loss_data = 0.
        train_iter.is_new_epoch = False
        while not train_iter.is_new_epoch:
            batch = chainer.Variable(
                convert(train_iter.next(), device=args.gpu), volatile='auto')
            loss, rein_loss = model(batch)
            model.zerograds()
            (loss + rein_loss).backward()
            optimizer.update()
            accum_loss_data += loss.data
            accum_rein_loss_data += rein_loss.data

        train_loss = '{:.6f}'.format(float(accum_loss_data / n_iters))
        mean_valid_loss_data = evaluate(
            model, valid, args.batchsize, args.gpu)

        if mean_valid_loss_data < best_valid:
            valid_loss = '\t\t{:.6f} *'.format(float(mean_valid_loss_data))
            S.save_npz(args.out + 'saved_model.model', model)

            best_valid = mean_valid_loss_data
            best_model = copy.deepcopy(model)
            best_epoch = i_epoch
            n_wins = 0

            batch = chainer.Variable(
                convert(train[:50], device=args.gpu), volatile='auto')
            generate(model, batch, epoch=i_epoch,
                     out=args.out, filename='train')
            batch = chainer.Variable(
                convert(test[:50], device=args.gpu), volatile='auto')
            generate(model, batch, epoch=i_epoch,
                     out=args.out, filename='test')
        else:
            n_wins += 1
            valid_loss = '{:.6f}'.format(float(mean_valid_loss_data))

        generate_by_template(
            model, epoch=i_epoch, out=args.out,
            filename='rec', agent_type='receiver')
        generate_by_template(
            model, epoch=i_epoch, out=args.out,
            filename='sen', agent_type='sender')

        print('Epoch', i_epoch,
              '\ttrain: {}\tvalid: {}'.format(train_loss, valid_loss))

        if n_wins >= 20:
            break

    mean_test_loss_data = evaluate(
        best_model, test, args.batchsize, args.gpu)
    print('TEST by model at', best_epoch, 'epoch.'
          '\ttest: {:.6f}'.format(float(mean_test_loss_data)))
    print('Finish at {}/{} epoch'.format(best_epoch, args.epoch))


if __name__ == '__main__':
    main()
