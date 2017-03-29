#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
from chainer import cuda
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

    def __init__(self, n_in, middle_units, n_turn, drop_ratio=0.):
        super(NaiveFCSensor, self).__init__(
            l1=L.Linear(n_in, middle_units),
            l2=L.Linear(middle_units, middle_units),
            l3=L.Linear(middle_units, middle_units),
            act1=L.PReLU(middle_units),
            act2=L.PReLU(middle_units),
            act3=L.PReLU(middle_units),

            bn_list1=chainer.ChainList(*[
                L.BatchNormalization(middle_units, use_cudnn=False)
                for i in range(n_turn)
            ]),
            bn_list2=chainer.ChainList(*[
                L.BatchNormalization(middle_units, use_cudnn=False)
                for i in range(n_turn)
            ]),
            bn_list3=chainer.ChainList(*[
                L.BatchNormalization(middle_units, use_cudnn=False)
                for i in range(n_turn)
            ]),
        )
        self.drop_ratio = drop_ratio

    def __call__(self, x, turn, train=True):
        h1 = self.act1(self.l1(x))
        h1 = F.dropout(h1, ratio=self.drop_ratio, train=train)
        h2 = self.l2(h1)
        h2 = self.act2(self.bn_list2[turn](h2, test=not train))
        h2 = F.dropout(h2, ratio=self.drop_ratio, train=train)
        h3 = self.l3(h2)
        h3 = self.act3(self.bn_list3[turn](h3, test=not train))
        return h3


class EricFCSensor(chainer.Chain):

    def __init__(self, n_in, middle_units, n_units, n_turn, drop_ratio=0.):
        super(EricFCSensor, self).__init__(
            l1=L.Linear(n_in, middle_units),
            #l2=L.Linear(middle_units, n_units),
            l2=L.Linear(middle_units, middle_units),
            #l3=L.Linear(middle_units, n_units),
            l3=L.Linear(middle_units, middle_units),
            # bn=L.BatchNormalization(n_units),
            bn1=L.BatchNormalization(middle_units, use_cudnn=False),
            bn2=L.BatchNormalization(middle_units, use_cudnn=False),
            bn3=L.BatchNormalization(n_units, use_cudnn=False),

            act1=L.PReLU(middle_units),
            act2=L.PReLU(middle_units),
        )
        self.drop_ratio = drop_ratio

    def __call__(self, x, turn, train=True):
        h = self.l1(x)
        #h = self.bn1(h, test=not train)
        #h = self.act1(h)
        h = F.tanh(h)
        #h = F.dropout(h, ratio=self.drop_ratio, train=train)
        #h = self.l2(h)
        #h = self.bn2(h, test=not train)
        #h = self.act2(h)
        #h = F.dropout(h, ratio=self.drop_ratio, train=train)
        #h = self.l3(h)

        #h = self.bn3(h, test=not train)
        #h = F.leaky_relu(h)
        #h = F.dropout(h, ratio=self.drop_ratio, train=train)
        return h


def add_noise(h, test, sigma=0.1):
    xp = cuda.get_array_module(h.data)
    return h
    if test:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)


class ConvSensor(chainer.Chain):
    """
    https://github.com/pfnet/chainer/tree/master/examples/dcgan

    Now, turn is ignored.
    """

    def __init__(self, n_units, bottom_width=4, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(ConvSensor, self).__init__(
            c0_0=L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w),
            c0_1=L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w),
            c1_0=L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w),
            c1_1=L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w),
            c2_0=L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w),
            c2_1=L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w),
            c3_0=L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w),
            l4=L.Linear(bottom_width * bottom_width * ch, n_units, initialW=w),
            bn0_1=L.BatchNormalization(ch // 4, use_gamma=False),
            bn1_0=L.BatchNormalization(ch // 4, use_gamma=False),
            bn1_1=L.BatchNormalization(ch // 2, use_gamma=False),
            bn2_0=L.BatchNormalization(ch // 2, use_gamma=False),
            bn2_1=L.BatchNormalization(ch // 1, use_gamma=False),
            bn3_0=L.BatchNormalization(ch // 1, use_gamma=False),
        )

    def __call__(self, x, turn, train=True):
        test = not train
        h = add_noise(x, test=test)
        h = F.leaky_relu(add_noise(self.c0_0(h), test=test))
        h = F.leaky_relu(add_noise(self.bn0_1(
            self.c0_1(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn1_0(
            self.c1_0(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn1_1(
            self.c1_1(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn2_0(
            self.c2_0(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn2_1(
            self.c2_1(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn3_0(
            self.c3_0(h), test=test), test=test))
        return self.l4(h)
