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

# from links import communicator
from links import world

import numpy as np

try:
    from tqdm import tqdm
except:
    tqdm = list

import json


def generate(model, data, epoch=0, out='./', train=False, printer=False):
    prev_train = model.train
    model.train = False
    sentence_history, log_prob_history, canvas_history = model(
        data, generate=True)
    canvas_history = [c * 255 for c in canvas_history]
    true_image = data * 255
    model.train = prev_train

    def save_images(x, filename):
        x = np.array(x.tolist(), np.float32)
        width = x.shape[0]
        fig, ax = plt.subplots(1, width, figsize=(1 * width, 1))  # , dpi=20)
        for a in ax:
            a.set_xticklabels([])
            a.set_yticklabels([])

        for ai, xi in zip(ax.ravel(), x):
            ai.imshow(xi.reshape(28, 28), cmap='Greys_r')
        fig.savefig(filename)
        plt.clf()
        plt.close('all')

    save_images(true_image.data, out + str(train) + '_.png')
    if printer:
        print('save _.png')
    for i in range(model.n_turn):
        save_images(
            canvas_history[i], out + str(train) + '{}e_{}.png'.format(epoch, i))
        if printer:
            print('save {}.png'.format(i))
    for i in range(data.shape[0]):
        for log_prob_batch, word_batch_list in zip(
                log_prob_history, sentence_history):
            if printer:
                print(str(i) + ",\t",
                      [int(word_batch[i]) for word_batch in word_batch_list],
                      log_prob_batch[i])
    sentence_log = [[[[int(word_batch[i]) for word_batch in word_batch_list],
                      float(log_prob_batch[i])]
                     for log_prob_batch, word_batch_list
                     in zip(log_prob_history, sentence_history)]
                    for i in range(data.shape[0])]
    json_f = open(
        out + 'message.' + str(train) + '{}e.json'.format(epoch), 'w')
    json.dump(sentence_log, json_f)


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

    model = world.World(
        28 * 28, args.image_unit, args.unit,
        n_vocab=args.vocab, n_word=args.word, n_turn=args.turn,
        drop_ratio=args.drop_ratio, co_importance=args.co_importance,
        co_orthogonal=args.co_orthogonal)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(1.))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(withlabel=False)
    print('# of train data:', len(train))
    print('# of test data:', len(test))

    import os
    if not os.path.isdir(args.out):
        if os.path.exists(args.out):
            print(args.out, 'exists as a file')
            exit()
        else:
            os.mkdir(args.out)

    batchsize = args.batchsize
    convert = chainer.dataset.convert.concat_examples
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
                model.xp.array(convert(batch).tolist(), np.float32),
                volatile='auto')
            model.zerograds()
            loss = model(batch)
            loss.backward()
            optimizer.update()
            accum_loss_data += loss.data - model.sub_accum_loss
            del loss

        convert = chainer.dataset.convert.concat_examples
        d = convert(train[:50])
        d = chainer.Variable(
            model.xp.array(d.tolist(), np.float32), volatile='auto')
        generate(model, d, epoch=i_epoch, out=args.out, train=True,
                 printer=(i_epoch == args.epoch - 1))

        d = convert(test[:50])
        d = chainer.Variable(
            model.xp.array(d.tolist(), np.float32), volatile='auto')
        generate(model, d, epoch=i_epoch, out=args.out, train=False,
                 printer=(i_epoch == args.epoch - 1))
        del d

        print(i_epoch, 'loss :', accum_loss_data / n_iters)
        mean_valid_loss_data = evaluate(model, test, batchsize, convert)

        if mean_valid_loss_data < best_valid:
            best_valid = mean_valid_loss_data
            best_keep = 0
            S.save_npz(args.out + 'saved_model.model', model)
            print(i_epoch, 'valid:', mean_valid_loss_data, '*')
        else:
            best_keep += 1
            print(i_epoch, 'valid:', mean_valid_loss_data)
        if best_keep >= 10:
            break

    #S.save_npz(args.out + 'saved_model.model', model)
    print('Finish at {}/{} epoch'.format(i_epoch, args.epoch))


def evaluate(model, test, batchsize, convert):

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
            model.xp.array(convert(batch).tolist(), np.float32),
            volatile='auto')
        valid_loss_data = model(batch).data
        valid_loss_data -= model.sub_accum_loss
        accum_valid_loss_data += valid_loss_data * len(ids)

    model.train = True

    return accum_valid_loss_data / len(test)


if __name__ == '__main__':
    main()
