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

    def __init__(self, n_in, n_middle, n_units, n_vocab, n_word, n_turn,
                 drop_ratio=0., co_importance=0.):
        sensor_for_listener = Sensor.NaiveFCSensor(
            n_in, n_middle, n_turn, drop_ratio)
        sensor_for_speaker = Sensor.NaiveFCSensor(
            n_in, n_middle, n_turn + 1, drop_ratio)

        reconstructor_for_listener = Recon.NaiveFCReconstructor(n_in, n_middle)
        reconstructor_for_speaker = Recon.NaiveFCReconstructor(n_in, n_middle)

        painter = Painter.NaiveFCPainter(n_in, n_middle, n_turn)

        language_for_listener = Language.NaiveLanguage(n_units, n_vocab, n_turn,
                                                       listener=True)
        language_for_speaker = Language.NaiveLanguage(n_units, n_vocab, n_turn,
                                                      speaker=True)

        listener = Listener.NaiveListener(n_in, n_middle, n_units, n_turn,
                                          sensor_for_listener,
                                          language_for_listener,
                                          painter,
                                          reconstructor_for_listener)

        speaker = Speaker.NaiveSpeaker(n_in, n_middle, n_units, n_turn,
                                       sensor_for_speaker,
                                       language_for_speaker,
                                       reconstructor_for_speaker)

        super(World, self).__init__(
            listener=listener,
            speaker=speaker,
        )

        self.n_turn = n_turn
        self.n_word = n_word
        self.drop_ratio = drop_ratio

        self.train = True

        self.calc_reconstruction = False
        self.calc_full_turn = True
        self.calc_modification = False
        # self.calc_orthogonal_loss = False
        self.calc_orthogonal_loss = True
        self.calc_importance_loss = co_importance
        self.calc_word_l2_loss = False

        self.baseline = None

    def __call__(self, image, generate=False):
        n_turn, n_word = self.n_turn, self.n_word
        train = self.train
        # train = True
        # traing no atai izon de okasiku naru...

        accum_loss = 0.
        sub_accum_loss = 0.

        accum_loss_reconstruction = 0.

        batchsize = image.data.shape[0]
        sentence_history = []
        log_prob_history = []
        canvas_history = []
        p_dists_history = []

        # Initialize canvas of Listener
        canvas = chainer.Variable(
            self.xp.ones(image.data.shape, np.float32), volatile='auto')

        loss_list = []
        raw_loss_list = []

        # [Speaker]
        # Percieve
        hidden_image = self.speaker.perceive(image, n_turn, train=train)
        # bn_list[n_turn] is used for real image

        # Self-reconstruction
        if self.calc_reconstruction:
            loss_recon = self.speaker.reconstructor(image, hidden_image)
            accum_loss_reconstruction += loss_recon

        for turn in range(n_turn):
            # [Speaker]
            # Express the image x compared to canvas
            # Perceive
            hidden_canvas = self.speaker.perceive(canvas, turn, train=train)
            # Self-reconstruction
            if self.calc_reconstruction:
                loss_recon = self.speaker.reconstructor(canvas, hidden_canvas)
                accum_loss_reconstruction += loss_recon

            # Express
            thought = self.speaker.think(
                hidden_image, hidden_canvas, turn, train=train)

            sampled_word_idx_seq, log_probability, p_dists = self.speaker.speak(
                thought, n_word=n_word, train=train)

            # [Listener]
            # Interpret the expression & Paint it into canvas
            # Perceive (only canvas)
            hidden_canvas = self.listener.perceive(canvas, turn, train=train)
            # Self-reconstruction
            if self.calc_reconstruction:
                loss_recon = self.listener.reconstructor(canvas, hidden_canvas)
                accum_loss_reconstruction += loss_recon

            # Interpret the expression with current situation (canvas)
            message_meaning = self.listener.listen(
                sampled_word_idx_seq, turn, train=train)

            # ZURU
            # message_meaning += thought

            concept = self.listener.think(
                hidden_canvas, message_meaning, turn, train=train)

            # Paint
            # canvas = self.listener.painter(
            #    canvas, concept, turn, train=train)
            canvas += self.listener.painter(
                concept, turn, train=train)

            # Physical limitations of canvas (leaky to make gradient active)
            canvas = F.clip(canvas, 0., 1.) * 0.9 + canvas * 0.1

            # Save
            canvas_history.append(canvas)
            sentence_history.append(sampled_word_idx_seq)
            log_prob_history.append(log_probability)
            p_dists_history.append(p_dists)

            # Calculate communication loss
            raw_loss = F.sum((canvas - image) ** 2, axis=1)
            raw_loss_list.append(raw_loss)

            loss = F.sum(raw_loss) / image.data.size
            loss_list.append(loss)

            reporter.report({'l{}'.format(turn): loss}, self)
            reporter.report({'p{}'.format(turn): self.xp.exp(
                log_probability.data.mean())}, self)

        # Add the last loss
        accum_loss += loss_list[-1]
        reporter.report({'loss': accum_loss}, self)

        # Add (minus) reinforce
        reward = (1. - raw_loss_list[-1]).data
        baseline = self.baseline if not self.baseline is None \
            else self.xp.mean(reward)
        reinforce = F.sum(sum(log_prob_history) / n_turn *
                          (reward - baseline)) / reward.size
        self.baseline = self.baseline * 0.95 + self.xp.mean(reward) * 0.05 \
            if not self.baseline is None \
            else self.xp.mean(reward)
        accum_reinforce = reinforce
        reporter.report({'reward': accum_reinforce}, self)
        sub_accum_loss -= accum_reinforce * 0.00001

        # Add loss of self-reconstruction
        if self.calc_reconstruction:
            sub_accum_loss += accum_loss_reconstruction * 0.001
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
                nM = F.normalize(M)
                MM = F.matmul(nM, F.transpose(nM))
                iden = self.xp.identity(MM.shape[0])
                return F.sum((MM - MM * iden) ** 2)

            orthogonal_loss = orthogonal_regularizer(
                self.speaker.language.definition.W) + \
                orthogonal_regularizer(
                    self.listener.language.definition.W)
            sub_accum_loss += orthogonal_loss * 0.01
            reporter.report({'ortho': orthogonal_loss}, self)

        # Add balancing vocabulary
        if self.calc_importance_loss:
            def importance_regularizer(p):
                importance = F.sum(p, axis=0)
                mean_i = F.sum(importance) / importance.size
                mean_i_bc = F.broadcast_to(mean_i[None, ], importance.shape)
                std_i = (F.sum((importance - mean_i_bc) ** 2) /
                         importance.size) ** 0.5
                cv = std_i / mean_i
                return cv ** 2

            importance_loss = importance_regularizer(
                F.concat(sum(p_dists_history, []), axis=0))
            sub_accum_loss += importance_loss * self.calc_importance_loss
            reporter.report({'importance': importance_loss}, self)

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
