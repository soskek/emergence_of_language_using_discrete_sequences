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
            bn_concept_list=chainer.ChainList(*[
                L.BatchNormalization(n_middle, use_cudnn=False)
                for i in range(n_turn)
            ]),
            bn_message_list=chainer.ChainList(*[
                L.BatchNormalization(n_units, use_cudnn=False)
                for i in range(n_turn)
            ]),
        )

        if reconstructor:
            self.add_link('reconstructor', reconstructor)
        else:
            self.reconstructor = None

        self.act = F.relu

    def __call__(self, hidden_canvas, message_sentence, turn, train=True):
        return self.think(hidden_canvas, message_sentence,
                          turn=turn, train=train)

    def paint(self, concept, train=True):
        return self.painter(concept, train=train)

    def think(self, hidden_canvas, message_meaning, turn, train=True):

        h1_lang = self.l1_language(message_meaning)
        h1_canv = self.l1_canvas(hidden_canvas)

        h1 = self.act(h1_lang + h1_canv)
        concept = self.act(self.l2(h1))
        concept = self.bn_concept_list[turn](
            concept, test=not train)
        return concept

    def listen(self, message_sentence, turn, train=True):
        message_meaning = self.language.interpret_sentence(
            message_sentence, train=train)
        message_meaning = self.bn_message_list[turn](
            message_meaning, test=not train)
        return message_meaning

    def perceive(self, image, turn, train=True):
        hidden_image = self.sensor(image, turn, train=train)
        return hidden_image
