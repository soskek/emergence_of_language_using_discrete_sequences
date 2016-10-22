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

import language as Language
import listener as Listener
import painter as Painter
import sensor as Sensor
import speaker as Speaker
import reconstructor as Recon


class World(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units, n_vocab, n_word, n_turn):
        sensor_for_listener = Sensor.NaiveFCSensor(n_in, n_middle)
        sensor_for_speaker = Sensor.NaiveFCSensor(n_in, n_middle)

        reconstructor_for_speaker = Recon.NaiveFCReconstructor(n_in, n_middle)

        painter = Painter.NaiveFCPainter(n_in, n_middle)

        language_for_listener = Language.NaiveLanguage(n_units, n_vocab,
                                                       listener=True)
        language_for_speaker = Language.NaiveLanguage(n_units, n_vocab,
                                                      speaker=True)

        listener = Listener.NaiveListener(n_in, n_middle, n_units,
                                          sensor_for_listener,
                                          language_for_listener,
                                          painter)

        speaker = Speaker.NaiveSpeaker(n_in, n_middle, n_units,
                                       sensor_for_speaker,
                                       language_for_speaker,
                                       reconstructor_for_speaker)

        super(World, self).__init__(
            listener=listener,
            speaker=speaker,
        )

        self.n_turn = n_turn
        self.n_word = n_word

        self.train = True

        self.calc_reconstruction = True
        self.calc_full_turn = True
        self.calc_modification = True
        self.calc_orthogonal_loss = False
        self.calc_word_l2_loss = False

    def __call__(self, image, generate=False):
        n_turn, n_word = self.n_turn, self.n_word
        train = self.train

        accum_loss = 0.
        sub_accum_loss = 0.

        accum_loss_reconstruction = 0.

        batchsize = image.data.shape[0]
        sentence_history = []
        log_prob_history = []
        canvas_history = []

        # Initialize canvas of Listener
        canvas = chainer.Variable(
            self.xp.ones(image.data.shape, np.float32), volatile='auto')

        loss_list = []
        raw_loss_list = []

        for i in range(n_turn):
            # [Speaker]
            # Express the image x compared to canvas
            # Perceive
            hidden_image = self.speaker.perceive(image)
            hidden_canvas = self.speaker.perceive(canvas)
            # Express
            thought = self.speaker.think(
                hidden_image, hidden_canvas, train=train)
            sampled_word_idx_seq, log_probability = self.speaker.speak(
                thought, n_word=n_word, train=train)

            # Self-reconstruction
            loss_recon = self.speaker.reconstructor(image, hidden_image) + \
                self.speaker.reconstructor(canvas, hidden_canvas)
            accum_loss_reconstruction += loss_recon

            # [Listener]
            # Interpret the expression & Paint it into canvas
            # Perceive (only canvas)
            hidden_canvas = self.listener.perceive(canvas, train=train)

            # Interpret the expression with current situation (canvas)
            message_meaning = self.listener.listen(
                sampled_word_idx_seq, train=train)
            concept = self.listener.think(
                hidden_canvas, message_meaning, train=train)
            plus_draw = self.listener.painter(concept, train=train)

            # Paint
            canvas = canvas + plus_draw
            # Physical limitations of canvas (leaky to make gradient active)
            canvas = F.clip(canvas, 0., 1.) * 0.9 + canvas * 0.1

            # Save
            canvas_history.append(canvas)
            sentence_history.append(sampled_word_idx_seq)
            log_prob_history.append(log_probability)

            # Calculate communication loss
            raw_loss = F.sum((canvas - image) ** 2, axis=1)
            raw_loss_list.append(raw_loss)

            loss = F.sum(raw_loss) / image.data.size
            loss_list.append(loss)

            reporter.report({'l{}'.format(i): loss}, self)
            reporter.report({'p{}'.format(i): self.xp.exp(
                log_probability.data.mean())}, self)

        # Add the last loss
        accum_loss += loss_list[n_turn - 1]
        reporter.report({'loss': accum_loss}, self)

        # Add (minus) reinforce
        reward = (1. - raw_loss_list[-1]).data
        baseline = self.xp.mean(reward)
        reinforce = F.sum(sum(log_prob_history) *
                          (reward - baseline)) / reward.size
        accum_reinforce = reinforce
        reporter.report({'reward': accum_reinforce}, self)
        sub_accum_loss -= accum_reinforce * 0.001

        # Add loss of self-reconstruction
        if self.calc_reconstruction:
            sub_accum_loss += accum_loss_reconstruction * 0.01
            reporter.report({'recon': accum_loss_reconstruction}, self)

        # Add loss at full turn
        if self.calc_full_turn:
            decay = 0.5
            accum_loss_full_turn = sum(
                loss_list[j] * decay ** (n_turn - j - 1)
                for j in range(n_turn - 1))
            sub_accum_loss += accum_loss_full_turn
            reporter.report({'full': accum_loss_full_turn}, self)

        # Add loss of modification
        if self.calc_modification:
            margin = 0.1
            accum_loss_modification = sum(
                F.relu(margin + loss_list[i] - loss_list[i - 1].data)
                for i in range(1, n_turn))
            sub_accum_loss += accum_loss_modification
            reporter.report({'mod': accum_loss_modification}, self)

        # Add loss to orthogonal matrix
        if self.calc_orthogonal_loss:
            def orthogonal_regularizer(M):
                MM = F.matmul(M, F.transpose(M))
                iden = self.xp.identity(MM.shape[0])
                norm_loss = F.sum((iden - MM * iden) ** 2)
                return F.sum((MM - MM * iden) ** 2) + norm_loss

            orthogonal_loss = orthogonal_regularizer(
                self.speaker.language.expression.W) + \
                orthogonal_regularizer(
                self.listener.language.definition.W)
            sub_accum_loss += orthogonal_loss * 0.001
            reporter.report({'ortho': orthogonal_loss}, self)

        # Add l2 norm of language by usage freq
        if self.calc_word_l2_loss:
            def word_l2(idx):
                definition_l2 = self.listener.language.definition(idx) ** 2
                expression_l2 = F.embed_id(
                    idx, self.speaker.language.expression.W) ** 2
                return definition_l2 + expression_l2
                word_l2_loss = F.sum(
                    sum(sum(word_l2(i) for i in sent)
                        for sent in sentence_history)) / batchsize
            sub_accum_loss += word_l2_loss * 0.0001
            reporter.report({'word_l2': word_l2_loss}, self)

        reporter.report({'total': accum_loss}, self)

        # Merge main and sub loss
        accum_loss += sub_accum_loss
        self.sub_accum_loss = sub_accum_loss.data

        if generate:
            return [[i.data for i in s] for s in sentence_history], \
                [lp.data for lp in log_prob_history], \
                [F.clip(cv, 0., 1.).data for cv in canvas_history]
        else:
            return accum_loss
