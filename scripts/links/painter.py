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

    def __init__(self, n_in, n_middle):
        super(NaiveFCPainter, self).__init__(
            l1=L.Linear(n_middle, n_middle),
            l2=L.Linear(n_middle, n_in),
        )
        self.act = F.relu

    def __call__(self, concept, train=True):
        h1 = self.act(self.l1(concept))
        plus_draw = F.tanh(self.l2(h1))
        return plus_draw ** 3
