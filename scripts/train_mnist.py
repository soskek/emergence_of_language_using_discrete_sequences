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


def generate(model, data, train=False):
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
        fig, ax = plt.subplots(1, width, figsize=(4*width, 10), dpi=5)
        for ai, xi in zip(ax.ravel(), x):
            ai.imshow(xi.reshape(28, 28), cmap='Greys_r')
        fig.savefig(filename)
        
    save_images(true_image.data, str(train)+'_.png')
    print('save _.png')
    for i in range(model.n_turn):
        save_images(canvas_history[i], str(train)+'{}.png'.format(i))
        print('save {}.png'.format(i))
    for i in range(data.shape[0]):
        for log_prob_batch, word_batch_list in zip(log_prob_history, sentence_history):
            print(str(i)+",\t", [int(word_batch[i]) for word_batch in word_batch_list], log_prob_batch[i])


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
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(10.))
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(withlabel=False)
    print('# of train data:', len(train))
    print('# of test data:', len(test))

    convert = chainer.dataset.convert.concat_examples
    d = convert(test[:50])
    d = chainer.Variable(model.xp.array(d.tolist(), np.float32), volatile='auto')
    generate(model, d)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    eval_model = model.copy()  # Model with shared params and distinct states
    eval_model.train = False
    evaluator = extensions.Evaluator(test_iter, eval_model, device=args.gpu)
    evaluator.default_name = 'val'
    trainer.extend(evaluator)

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    ##trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    ## TODO: image-generate function as Trigger?

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    N = model.n_turn
    trainer.extend(extensions.PrintReport(
        ['epoch'] \
        + ['main/total'] \
        + ['main/reconst'] \
        + ['main/ortho'] \
        + ['main/l{}'.format(j) for j in range(N)] \
        + ['main/nr{}'.format(j) for j in range(N)] \
        + ['main/p{}'.format(j) for j in range(N)] \
        + ['val/main/total'] \
        + ['val/reconst'] \
        + ['val/ortho'] \
        + ['val/main/l{}'.format(j) for j in range(N)] \
        + ['val/main/nr{}'.format(j) for j in range(N)] \
        + ['val/main/p{}'.format(j) for j in range(N)]
    ))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    convert = chainer.dataset.convert.concat_examples
    d = convert(train[:50])
    d = chainer.Variable(model.xp.array(d.tolist(), np.float32), volatile='auto')
    generate(model, d, train=True)

    convert = chainer.dataset.convert.concat_examples
    d = convert(test[:50])
    d = chainer.Variable(model.xp.array(d.tolist(), np.float32), volatile='auto')
    generate(model, d, train=False)

if __name__ == '__main__':
    main()
