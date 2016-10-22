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


class NaiveFCReconstructor(chainer.Chain):

    def __init__(self, n_in, n_middle):
        super(NaiveFCReconstructor, self).__init__(
            l1=L.Linear(n_middle, n_middle),
            l2=L.Linear(n_middle, n_middle),
            l3=L.Linear(n_middle, n_in),
        )
        self.act = F.relu

    def __call__(self, true_image, hidden_image, train=True):
        h1 = self.act(self.l1(hidden_image))
        h2 = self.act(self.l2(h1))
        recon_image = F.sigmoid(self.l3(h2))
        return F.mean_squared_error(true_image, recon_image)
