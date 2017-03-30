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


class NaiveListener(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units, n_turn,
                 sensor, language, painter, reconstructor=None):
        super(NaiveListener, self).__init__(
            sensor=sensor,
            painter=painter,
            language=language,
            l1_language=L.Linear(n_units, n_middle),
            l1_canvas=L.Linear(n_middle, n_middle),
            l2=L.Linear(n_middle, n_middle),
            l3=L.Linear(n_middle, n_middle),
            bn_list2=chainer.ChainList(*[
                L.BatchNormalization(n_middle, use_cudnn=False)
                for i in range(n_turn)
            ]),
            bn_list3=chainer.ChainList(*[
                L.BatchNormalization(n_middle, use_cudnn=False)
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

        self.act = F.relu

    def __call__(self, hidden_canvas, message_sentence, turn, train=True):
        return self.think(hidden_canvas, message_sentence,
                          turn=turn, train=train)

    def paint(self, concept, turn, train=True):
        return self.painter(concept, turn, train=train)

    def think(self, hidden_canvas, message_meaning, turn, train=True):

        h1_lang = self.l1_language(message_meaning)
        h1_canv = self.l1_canvas(hidden_canvas)

        h1 = self.act1(h1_lang + h1_canv)
        h2 = self.l2(h1)
        h2 = self.act2(self.bn_list2[turn](h2, test=not train))
        h3 = self.l3(h2)
        h3 = self.act3(self.bn_list3[turn](h3, test=not train))
        concept = h3
        return concept

    def listen(self, message_sentence, turn, train=True):
        message_meaning = self.language.interpret_sentence(
            message_sentence, train=train)
        return message_meaning

    def perceive(self, image, turn, train=True):
        hidden_image = self.sensor(image, turn, train=train)
        return hidden_image


class EricListener(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units, n_turn,
                 sensor, language, painter, reconstructor=None):
        super(EricListener, self).__init__(
            sensor=sensor,
            painter=painter,
            language=language,
            l1=L.Linear(n_units, n_middle),
            act1=L.PReLU(n_middle),
            l2=L.Linear(n_middle, n_middle),
            act2=L.PReLU(n_middle),
            l3=L.Linear(n_middle, n_middle),
            act3=L.PReLU(n_middle),
            bn1=L.BatchNormalization(n_middle, use_cudnn=False),
        )

        if reconstructor:
            self.add_link('reconstructor', reconstructor)
        else:
            self.reconstructor = None

    def __call__(self, hidden_canvas, message_sentence, turn, train=True):
        return self.think(hidden_canvas, message_sentence,
                          turn=turn, train=train)

    def paint(self, concept, turn, train=True):
        return self.painter(concept, turn, train=train)

    def think(self, hidden_canvas, message_meaning, turn, train=True):
        concept = self.l1(message_meaning)
        #concept = self.bn1(concept, test=not train)
        #concept = self.act1(concept)
        concept = F.tanh(concept)
        """
        concept = self.l2(concept)
        concept = self.act2(concept)
        concept = self.l3(concept)
        concept = self.act3(concept)
        """
        return concept

    def listen(self, message_sentence, turn, train=True):
        #message_sentence = message_sentence[::-1]
        message_meaning = self.language.interpret_sentence(
            message_sentence, train=train)
        return message_meaning

    def perceive(self, image, turn, train=True):
        hidden_image = self.sensor(image, turn, train=train)
        return hidden_image
