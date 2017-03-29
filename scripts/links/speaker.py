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
            bn_list2=chainer.ChainList(*[
                L.BatchNormalization(n_middle, use_cudnn=False)
                for i in range(n_turn)
            ]),
            bn_list3=chainer.ChainList(*[
                L.BatchNormalization(n_middle, use_cudnn=False)
                for i in range(n_turn)
            ]),
            bn_list4=chainer.ChainList(*[
                L.BatchNormalization(n_units, use_cudnn=False)
                for i in range(n_turn)
            ]),
            act1=L.PReLU(n_middle),
            act2=L.PReLU(n_middle),
            act3=L.PReLU(n_middle),
        )
        if reconstructor:
            self.add_link('reconstructor', reconstructor)
        else:
            self.reconstructor = None

    def perceive(self, image, turn, train=True):
        hidden_image = self.sensor(image, turn, train=train)
        return hidden_image

    def think(self, hidden_image, hidden_canvas, turn, train=True):
        h1_image = self.l1_image(hidden_image)
        h1_canvas = self.l1_canvas(hidden_canvas)

        h1 = self.act1(h1_canvas + h1_image)
        h2 = self.l2(h1)
        h2 = self.act2(self.bn_list2[turn](h2, test=not train))
        h3 = self.l3(h2)
        h3 = self.act3(self.bn_list3[turn](h3, test=not train))
        h4 = self.l4(h3)
        thought = F.tanh(self.bn_list4[turn](h4, test=not train))
        return thought

    def __call__(self, hidden_image, hidden_canvas, turn, train=True):
        return self.think(hidden_image, hidden_canvas, turn, train=train)

    def speak(self, thought, n_word=3, train=True):
        sampled_word_idx_seq, total_log_probability, p_dists = \
            self.language.decode_thought(thought, n_word, train=train)
        return sampled_word_idx_seq, total_log_probability, p_dists

    def speak_loss(self, thought, word_idx_seq, n_word=3, train=True):
        loss = self.language.decode_thought_loss(
            thought, word_idx_seq, n_word, train=train)
        return loss


class EricSpeaker(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units, n_turn,
                 sensor, language, reconstructor=None):
        super(EricSpeaker, self).__init__(
            sensor=sensor,
            language=language,
            l1=L.Linear(n_middle, n_units),
            bn1=L.BatchNormalization(n_units),
        )
        if reconstructor:
            self.add_link('reconstructor', reconstructor)
        else:
            self.reconstructor = None
        self.n_turn = n_turn

    def perceive(self, image, turn, train=True):
        hidden_image = self.sensor(image, turn, train=train)
        return hidden_image

    def think(self, hidden_image, hidden_canvas, turn, train=True):
        thought = self.l1(hidden_image)
        thought = self.bn1(thought, test=not train)
        thought = F.tanh(thought)
        return thought

    def __call__(self, hidden_image, hidden_canvas, turn, train=True):
        return self.think(hidden_image, hidden_canvas, turn, train=train)

    def speak(self, thought, n_word=3, train=True):
        sampled_word_idx_seq, total_log_probability, p_dists = \
            self.language.decode_thought(thought, n_word, train=train)
        return sampled_word_idx_seq, total_log_probability, p_dists

    def speak_loss(self, thought, word_idx_seq, n_word=3, train=True):
        loss = self.language.decode_thought_loss(
            thought, word_idx_seq, n_word, train=train)
        return loss
