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


class NaiveCNNSensor(chainer.Chain):
    # like
    # https://github.com/mitmul/chainer-cifar10/blob/master/models/Cifar10.py

    def __init__(self, n_in, middle_units, n_turn):
        super(NaiveCNNSensor, self).__init__(
            conv1=L.Convolution2D(3, 32, 5, stride=1, pad=2),
            conv2=L.Convolution2D(32, 32, 5, stride=1, pad=2),
            conv3=L.Convolution2D(32, 64, 5, stride=1, pad=2),
            fc4=F.Linear(None, middle_units),
            fc5=F.Linear(middle_units, middle_units),
            bn_list=chainer.ChainList(*[
                L.BatchNormalization(middle_units, use_cudnn=False)
                for i in range(n_turn)
            ])
        )
        self.act = F.relu

    def __call__(self, x, turn, train=True):
        h = F.max_pooling_2d(self.act(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(self.act(self.conv2(h)), 3, stride=2)
        h = self.act(self.conv3(h))
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = self.act(self.fc4(h))
        h = self.fc5(h)
        h = self.bn_list[turn](h, test=not train)
        return h


class NaiveFCSensor(chainer.Chain):

    def __init__(self, n_in, middle_units, n_turn):
        super(NaiveFCSensor, self).__init__(
            l1=L.Linear(n_in, middle_units),
            l2=L.Linear(middle_units, middle_units),
            l3=L.Linear(middle_units, middle_units),
            bn_list=chainer.ChainList(*[
                L.BatchNormalization(middle_units, use_cudnn=False)
                for i in range(n_turn)
            ])
        )
        self.act = F.relu

    def __call__(self, x, turn, train=True):
        h1 = self.act(self.l1(x))
        h2 = self.act(self.l2(h1))
        h3 = self.act(self.l3(h2))
        h3 = self.bn_list[turn](h3, test=not train)
        return h3
