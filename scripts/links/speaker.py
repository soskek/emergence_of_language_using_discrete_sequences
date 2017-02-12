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


class NaiveSpeaker(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units, n_turn,
                 sensor, language, reconstructor=None):
        super(NaiveSpeaker, self).__init__(
            sensor=sensor,
            language=language,
            l1_image=L.Linear(n_middle, n_middle),
            l1_canvas=L.Linear(n_middle, n_middle),
            l2=L.Linear(n_middle, n_middle),
            l3=L.Linear(n_middle, n_middle),
            l4=L.Linear(n_middle, n_units),
            bn_list=chainer.ChainList(*[
                #L.BatchNormalization(n_units, use_cudnn=False)
                L.BatchNormalization(n_middle, use_cudnn=False)
                for i in range(n_turn)
            ])
        )
        if reconstructor:
            self.add_link('reconstructor', reconstructor)
        else:
            self.reconstructor = None

        self.act = F.relu

    def perceive(self, image, turn, train=True):
        hidden_image = self.sensor(image, turn, train=train)
        return hidden_image

    def think(self, hidden_image, hidden_canvas, turn, train=True):
        h1_image = self.l1_image(hidden_image)
        h1_canvas = self.l1_canvas(hidden_canvas)

        h1 = self.act(h1_canvas + h1_image)
        h2 = self.l2(h1)
        h2 = self.act(self.bn_list[turn](h2, test=not train))
        h3 = self.act(self.l3(h2))
        thought = F.tanh(self.l4(h3))
        return thought

    def __call__(self, hidden_image, hidden_canvas, turn, train=True):
        return self.think(hidden_image, hidden_canvas, turn, train=train)

    def speak(self, thought, n_word=3, train=True):
        sampled_word_idx_seq, total_log_probability = \
            self.language.decode_thought(thought, n_word, train=train)
        return sampled_word_idx_seq, total_log_probability
