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
