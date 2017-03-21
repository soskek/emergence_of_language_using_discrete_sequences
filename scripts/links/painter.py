#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import reporter

import numpy as np


class NaiveFCPainter(chainer.Chain):

    def __init__(self, n_in, n_middle, n_turn):
        super(NaiveFCPainter, self).__init__(
            l1=L.Linear(n_middle, n_middle),
            l2=L.Linear(n_middle, n_middle),
            l3=L.Linear(n_middle, n_in),
            l3_gate=L.Linear(n_middle, n_in),

            act1=L.PReLU(n_middle),
            act2=L.PReLU(n_middle),

            bn_list2=chainer.ChainList(*[
                L.BatchNormalization(n_middle, use_cudnn=False)
                for i in range(n_turn)
            ])
        )

    def __call__(self, concept, turn, train=True):
        h1 = self.act1(self.l1(concept))
        h2 = self.l2(h1)
        h2 = self.act2(self.bn_list2[turn](h2, test=not train))
        plus_draw = F.tanh(self.l3(h2))
        gate_draw = F.sigmoid(self.l3_gate(h2))
        return plus_draw * gate_draw


class DeconvPainter(chainer.Chain):
    """
    https://github.com/pfnet/chainer/tree/master/examples/dcgan

    Now, turn is ignored.
    """

    def __init__(self, n_hidden, bottom_width=4, ch=512, wscale=0.02):
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        w = chainer.initializers.Normal(wscale)
        super(Generator, self).__init__(
            l0=L.Linear(self.n_hidden, bottom_width *
                        bottom_width * ch, initialW=w),
            dc1=L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w),
            dc2=L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w),
            dc3=L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w),
            dc4=L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w),
            bn0=L.BatchNormalization(bottom_width * bottom_width * ch),
            bn1=L.BatchNormalization(ch // 2),
            bn2=L.BatchNormalization(ch // 4),
            bn3=L.BatchNormalization(ch // 8),
        )

    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(numpy.float32)

    def __call__(self, concept, turn, train=True):
        test = not train
        new_shape = (concept.data.shape[0], self.ch,
                     self.bottom_width, self.bottom_width)
        h = F.reshape(
            F.relu(self.bn0(self.l0(concept), test=test)),
            new_shape)
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = F.sigmoid(self.dc4(h))
        return x
