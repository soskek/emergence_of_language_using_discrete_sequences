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

    def __init__(self, n_in, n_middle, n_units,
                 sensor, language, painter):
        super(NaiveListener, self).__init__(
            sensor=sensor,
            painter=painter,
            language=language,
            l1_language=L.Linear(n_units, n_middle),
            l1_canvas=L.Linear(n_middle, n_middle),
            l2=L.Linear(n_middle, n_middle),
            l3=L.Linear(n_middle, n_middle),
        )
        self.act = F.relu

    def __call__(self, hidden_canvas, message_sentence, train=True):
        return self.think(hidden_canvas, message_sentence, train=train)

    def paint(self, concept, train=True):
        return self.painter(concept, train=train)

    def think(self, hidden_canvas, message_meaning, train=True):

        h1_lang = self.l1_language(message_meaning)
        h1_canv = self.l1_canvas(hidden_canvas)

        h1 = self.act(h1_lang + h1_canv)
        h2 = self.act(self.l2(h1))
        concept = self.act(self.l3(h2))
        return concept

    def listen(self, message_sentence, train=True):
        message_meaning = self.language.interpret_sentence(
            message_sentence, train=train)
        return message_meaning

    def perceive(self, image, train=True):
        hidden_image = self.sensor(image, train=train)
        return hidden_image
