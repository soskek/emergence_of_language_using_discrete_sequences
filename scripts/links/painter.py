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


class EricFCPainter(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units, n_turn, drop_ratio=0.0):
        super(EricFCPainter, self).__init__(
            #l1=L.Linear(n_units, n_middle),
            l1=L.Linear(n_middle, n_middle),
            l2=L.Linear(n_middle, n_middle),
            l3=L.Linear(n_middle, n_in),

            bn1=L.BatchNormalization(n_middle, use_cudnn=False),
            bn2=L.BatchNormalization(n_middle, use_cudnn=False),
            bn3=L.BatchNormalization(n_in, use_cudnn=False),

            act1=L.PReLU(n_middle),
            act2=L.PReLU(n_middle),
        )
        self.drop_ratio = drop_ratio
        if n_turn >= 2:
            self.add_link('lo_tanh', L.Linear(n_middle, n_in))
            self.add_link('bn_lo', L.BatchNormalization(n_in, use_cudnn=False))

    def __call__(self, x, turn, train=True):
        h = x
        #h = self.l1(x)
        #h = self.bn1(h, test=not train)
        #h = self.act1(h)
        #h = self.l2(h)
        #h = self.bn2(h, test=not train)
        #h = self.act2(h)
        h_out = self.l3(h)
        #h_out = self.bn3(h_out, test=not train)
        h_out = F.sigmoid(h_out)
        if hasattr(self, 'lo_tanh'):
            h_out = h_out * F.tanh(self.bn_lo(self.lo_tanh(h), test=not train))
        return h_out


class NaiveFCColorPainter(chainer.Chain):

    def __init__(self, n_in, n_middle, n_turn):
        super(NaiveFCColorPainter, self).__init__(
            l1=L.Linear(n_middle, n_middle),
            l2=L.Linear(n_middle, n_middle),
            l3=L.Linear(n_middle, n_middle),
            l4=L.Linear(n_middle, n_in),

            act1=L.PReLU(n_middle),
            act2=L.PReLU(n_middle),
            act3=L.PReLU(n_middle),

            bn_list2=chainer.ChainList(*[
                L.BatchNormalization(n_middle, use_cudnn=False)
                for i in range(n_turn)
            ]),
            bn_list3=chainer.ChainList(*[
                L.BatchNormalization(n_middle, use_cudnn=False)
                for i in range(n_turn)
            ]),

            l1_attention=L.Linear(n_middle, n_middle),
            act1_attention=L.PReLU(n_middle),
            l2_attention=L.Linear(n_middle, n_in),
        )
        field = n_in // 3
        rang = int(field ** 0.5)
        self.image_shape = (3, rang, rang)
        self.image_size = n_in

    def __call__(self, concept, turn, train=True):
        h1 = self.act1(self.l1(concept))
        h2 = self.l2(h1)
        h2 = self.act2(self.bn_list2[turn](h2, test=not train))
        h3 = self.l3(h2)
        h3 = self.act3(self.bn_list3[turn](h3, test=not train))
        draw = F.sigmoid(self.l4(h3))
        draw = F.reshape(draw, (draw.shape[0], ) + self.image_shape)
        return draw

    def attention(self, concept, train=True):
        h1 = self.act1_attention(self.l1_attention(concept))
        h2 = self.l2_attention(h1)
        attention = F.softmax(h2)
        batch = attention.shape[0]
        attention = F.reshape(attention,
                              (batch, ) + self.image_shape)
        return attention * self.image_size


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
        super(DeconvPainter, self).__init__(
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
        h = F.leaky_relu(self.bn0(self.l0(concept), test=test))
        new_shape = (concept.data.shape[0], self.ch,
                     self.bottom_width, self.bottom_width)
        h = F.reshape(
            h, new_shape)
        h = F.leaky_relu(self.bn1(self.dc1(h), test=test))
        h = F.leaky_relu(self.bn2(self.dc2(h), test=test))
        h = F.leaky_relu(self.bn3(self.dc3(h), test=test))
        x = F.sigmoid(self.dc4(h))
        return x
