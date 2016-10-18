#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from links import communicator

import numpy as np

try:
    from tqdm import tqdm
except:
    tqdm = list

import json


def generate(model, data, train=False, printer=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prev_train = model.train
    model.train = train
    sentence_history, log_prob_history, canvas_history = model(data, generate=True)
    canvas_history = [c * 255 for c in canvas_history]
    true_image = data * 255
    model.train = prev_train

    def save_images(x, filename):
        x = np.array(x.tolist(), np.float32)
        width = x.shape[0]
        fig, ax = plt.subplots(1, width, figsize=(1*width, 1))#, dpi=20)
        for ai, xi in zip(ax.ravel(), x):
            ai.imshow(xi.reshape(28, 28), cmap='Greys_r')
        fig.savefig(filename)
        
    save_images(true_image.data, str(train)+'_.png')
    np.save(str(train)+'_.npz', np.array(true_image.data.tolist()))
    if printer: print('save _.png')
    for i in range(model.n_turn):
        save_images(canvas_history[i], str(train)+'{}.png'.format(i))
        np.save(str(train)+'{}.npz'.format(i), np.array(canvas_history[i].tolist()))
        if printer: print('save {}.png'.format(i))
    for i in range(data.shape[0]):
        for log_prob_batch, word_batch_list in zip(log_prob_history, sentence_history):
            if printer: print(str(i)+",\t", [int(word_batch[i]) for word_batch in word_batch_list], log_prob_batch[i])
        save_target = [
            [
                [[int(word[i]) for word in sent] for sent in sentence_history],
                [float(lpb[i]) for lpb in log_prob_history]
            ]
            for i in range(data.shape[0])]
    json.dump(save_target, open(str(train)+'seq.json', 'w'))


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')


    #model = World(n_in, n_middle, n_units, n_vocab)
    #model = communicator.World(28 * 28, 128, 64, 32)
    model = communicator.World(28 * 28, 256, 128, n_vocab=32, n_word=2, n_turn=3)
    #model = communicator.World(28 * 28, 512, 128, n_vocab=32, n_word=2, n_turn=3)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(10.))
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(withlabel=False)
    print('# of train data:', len(train))
    print('# of test data:', len(test))

    convert = chainer.dataset.convert.concat_examples
    d = convert(train[:50])
    d = chainer.Variable(model.xp.array(d.tolist(), np.float32), volatile='auto')
    generate(model, d, train=True)

    batchsize = args.batchsize
    convert = chainer.dataset.convert.concat_examples
    log_report = extensions.LogReport()
    print('start')
    for i_epoch in range(args.epoch):
        n_iters = len(train) // batchsize
        permutation = np.random.permutation(len(train))
        accum_loss_data = 0.
        for i_iter in tqdm(range(n_iters)):
            ids = permutation[i_iter*batchsize:
                              (i_iter+1)*batchsize]
            batch = [train[idx] for idx in ids]
            batch = chainer.Variable(model.xp.array(convert(batch).tolist(), np.float32), volatile='auto')
            loss = model(batch)
            model.zerograds()
            loss.backward()
            optimizer.update()
            accum_loss_data += loss.data

        convert = chainer.dataset.convert.concat_examples
        d = convert(train[:50])
        d = chainer.Variable(model.xp.array(d.tolist(), np.float32), volatile='auto')
        generate(model, d, train=True, printer=(i_epoch == args.epoch - 1))
        
        convert = chainer.dataset.convert.concat_examples
        d = convert(test[:50])
        d = chainer.Variable(model.xp.array(d.tolist(), np.float32), volatile='auto')
        generate(model, d, train=False, printer=(i_epoch == args.epoch - 1))


        print(i_epoch, 'loss :', accum_loss_data)

        model.train = False
        batch = chainer.Variable(model.xp.array(convert(test).tolist(), np.float32), volatile='auto')
        valid_loss_data = model(batch).data
        print(i_epoch, 'valid:', valid_loss_data)
        model.train = True

    import chainer.serializers as S
    S.save_npz('saved_model.model', model)


if __name__ == '__main__':
    main()
