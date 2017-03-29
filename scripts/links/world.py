#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import report

import numpy as np

import language as Language
import listener as Listener
import painter as Painter
import sensor as Sensor
import speaker as Speaker


class World(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units, n_vocab, n_word, n_turn,
                 drop_ratio=0., co_importance=0., co_orthogonal=0.,
                 cifar=True):

        if cifar:
            # sensor_for_listener = Sensor.ConvSensor(n_middle)
            # sensor_for_speaker = Sensor.ConvSensor(n_middle)
            sensor_for_listener = Sensor.NaiveFCSensor(
                n_in, n_middle, n_turn, drop_ratio)
            sensor_for_speaker = Sensor.NaiveFCSensor(
                n_in, n_middle, n_turn + 1, drop_ratio)
            # painter = Painter.DeconvPainter(n_middle)
            painter = Painter.NaiveFCColorPainter(n_in, n_middle, n_turn)
        else:
            eric_sensor = True
            if eric_sensor:
                sensor_for_listener = Sensor.EricFCSensor(
                    n_in, n_middle, n_units, n_turn, drop_ratio)
                sensor_for_speaker = Sensor.EricFCSensor(
                    n_in, n_middle, n_units, n_turn, drop_ratio)
            else:
                sensor_for_listener = Sensor.NaiveFCSensor(
                    n_in, n_middle, n_turn, drop_ratio)
                sensor_for_speaker = Sensor.NaiveFCSensor(
                    n_in, n_middle, n_turn + 1, drop_ratio)
            #painter = Painter.NaiveFCPainter(n_in, n_middle, n_turn)
            painter = Painter.EricFCPainter(n_in, n_middle, n_units, n_turn)

        language_for_listener = Language.NaiveLanguage(n_units, n_vocab, n_turn,
                                                       listener=True)
        language_for_speaker = Language.NaiveLanguage(n_units, n_vocab, n_turn,
                                                      speaker=True)

        """
        listener = Listener.NaiveListener(n_in, n_middle, n_units, n_turn,
                                          sensor_for_listener,
                                          language_for_listener,
                                          painter)

        """
        listener = Listener.EricListener(n_in, n_middle, n_units, n_turn,
                                         sensor_for_listener,
                                         language_for_listener,
                                         painter)
        #"""
        """
        speaker = Speaker.NaiveSpeaker(n_in, n_middle, n_units, n_turn,
                                       sensor_for_speaker,
                                       language_for_speaker)
        """
        speaker = Speaker.EricSpeaker(n_in, n_middle, n_units, n_turn,
                                      sensor_for_speaker,
                                      language_for_speaker)

        assert(speaker.language is not listener.language)

        super(World, self).__init__(
            listener=listener,
            speaker=speaker,
        )

        self.n_turn = n_turn
        self.n_word = n_word
        self.drop_ratio = drop_ratio

        self.train = True

        self.calc_full_turn = True
        self.calc_modification = False
        # self.calc_orthogonal_loss = False
        self.calc_orthogonal_loss = co_orthogonal
        self.calc_importance_loss = co_importance

        self.baseline = None

        self.zuru = False

    def __call__(self, image, generate=False):
        n_turn, n_word = self.n_turn, self.n_word
        train = self.train
        # train = True
        # traing no atai izon de okasiku naru...

        accum_loss = 0.
        sub_accum_loss = 0.

        batchsize = image.data.shape[0]
        sentence_history = []
        log_prob_history = []
        canvas_history = []
        p_dists_history = []

        # Initialize canvas of Listener
        canvas = chainer.Variable(
            self.xp.zeros(image.data.shape, np.float32), volatile='auto')

        loss_list = []
        raw_loss_list = []

        # [Speaker]
        # Percieve
        hidden_image = self.speaker.perceive(image, n_turn, train=train)
        # bn_list[n_turn] is used for real image
        for turn in range(n_turn):
            # [Speaker]
            # Express the image x compared to canvas
            # Perceive
            hidden_canvas = self.speaker.perceive(canvas, turn, train=train)

            # Express
            thought = self.speaker.think(
                hidden_image, hidden_canvas, turn, train=train)

            sampled_word_idx_seq, log_probability, p_dists = self.speaker.speak(
                thought, n_word=n_word, train=train)

            # [Listener]
            # Interpret the expression & Paint it into canvas
            # Perceive (only canvas)
            hidden_canvas = self.listener.perceive(canvas, turn, train=train)

            # Interpret the expression with current situation (canvas)
            message_meaning = self.listener.listen(
                sampled_word_idx_seq, turn, train=train)

            concept = self.listener.think(
                hidden_canvas, message_meaning, turn, train=train)

            # ZURU
            if self.zuru:
                # concept = F.dropout(thought, ratio=0.5, train=train)
                concept = thought

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
            raw_loss = (canvas - image) ** 2

            second = reduce(lambda a, b: a * b, raw_loss.shape[1:])
            raw_loss = F.reshape(raw_loss,
                                 (raw_loss.shape[0], second))
            raw_loss = F.sum(raw_loss, axis=1)
            raw_loss_list.append(raw_loss)

            loss = F.sum(raw_loss) / image.data.size
            loss_list.append(loss)

            report({'l{}'.format(turn): loss}, self)
            report({'p{}'.format(turn): self.xp.exp(
                log_probability.data.mean())}, self)

        # Add the last loss
        accum_loss += loss_list[-1]
        report({'loss': accum_loss}, self)

        # Add (minus) reinforce
        #reward = (1. - raw_loss_list[-1]).data
        reward = (- raw_loss_list[-1]).data
        baseline = self.baseline if not self.baseline is None \
            else self.xp.mean(reward)

        reinforce = F.sum(sum(log_prob_history) / n_turn *
                          (reward - baseline)) / reward.size
        self.baseline = self.baseline * 0.99 + self.xp.mean(reward) * 0.01 \
            if not self.baseline is None \
            else self.xp.mean(reward)
        accum_reinforce = reinforce
        report({'reward': accum_reinforce}, self)
        #sub_accum_loss -= accum_reinforce * 0.00001
        #sub_accum_loss -= accum_reinforce * 100.
        #sub_accum_loss -= accum_reinforce * 0.1
        sub_accum_loss -= accum_reinforce * 1.

        # Add loss at full turn
        if self.calc_full_turn:
            decay = 0.5
            accum_loss_full_turn = sum(
                loss_list[j] * decay ** (n_turn - j - 1)
                for j in range(n_turn - 1))
            sub_accum_loss += accum_loss_full_turn
            report({'full': accum_loss_full_turn}, self)

        # Add loss of modification
        if self.calc_modification:
            margin = 0.1
            accum_loss_modification = sum(
                F.relu(margin + loss_list[i] - loss_list[i - 1].data)
                for i in range(1, n_turn))
            sub_accum_loss += accum_loss_modification
            report({'mod': accum_loss_modification}, self)

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
            sub_accum_loss += orthogonal_loss * self.calc_orthogonal_loss
            report({'ortho': orthogonal_loss}, self)

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

            concat_p = F.concat(sum(p_dists_history, []), axis=0)
            importance_loss = importance_regularizer(concat_p)
            sub_accum_loss += importance_loss * self.calc_importance_loss

            p_mean = F.sum(concat_p, axis=0) / concat_p.shape[0]
            report({'p_mean': p_mean}, self)

            report({'importance': importance_loss}, self)

        report({'total': accum_loss}, self)

        # Merge main and sub loss
        accum_loss += sub_accum_loss
        self.sub_accum_loss = sub_accum_loss.data

        if generate:
            return [[i.data for i in s] for s in sentence_history], \
                [lp.data for lp in log_prob_history], \
                [F.clip(cv, 0., 1.).data for cv in canvas_history]
        else:
            return accum_loss

    def generate(self, sampled_word_idx_seq, shape):
        n_turn, n_word = self.n_turn, self.n_word
        train = self.train

        batchsize = shape[0]
        sentence_history = []
        log_prob_history = []
        canvas_history = []
        p_dists_history = []

        # Initialize canvas of Listener
        canvas = chainer.Variable(
            self.xp.zeros(shape, np.float32), volatile='auto')

        for turn in range(n_turn):
            # [Listener]
            # Interpret the expression & Paint it into canvas
            # Perceive (only canvas)
            hidden_canvas = self.listener.perceive(canvas, turn, train=train)

            # Interpret the expression with current situation (canvas)
            message_meaning = self.listener.listen(
                sampled_word_idx_seq, turn, train=train)

            concept = self.listener.think(
                hidden_canvas, message_meaning, turn, train=train)

            # ZURU
            if self.zuru:
                concept = thought

            # Paint
            # canvas = self.listener.painter(
            #    canvas, concept, turn, train=train)
            canvas += self.listener.painter(
                concept, turn, train=train)

            # Physical limitations of canvas (leaky to make gradient active)
            canvas = F.clip(canvas, 0., 1.) * 0.9 + canvas * 0.1

            # Save
            canvas_history.append(canvas)

        return [F.clip(cv, 0., 1.).data for cv in canvas_history]

    def infer(self, sampled_word_idx_seq, shape):
        self.train = False
        train = False
        n_turn, n_word = self.n_turn, self.n_word

        batchsize = shape[0]

        # Initialize dreams of speaker
        dream = chainer.Link(dream=shape)
        if np != self.xp:
            dream.to_gpu()
            sampled_word_idx_seq = [
                self.xp.array(x) for x in sampled_word_idx_seq]

        dream.dream.data[:] = 0.01
        optimizer = chainer.optimizers.Adam(0.1)
        optimizer.setup(dream)

        n_iter = 400
        decay = 0.99
        for i_iter in range(n_iter):
            if i_iter == 200:
                optimizer.alpha /= 4
                decay = 1 - (1 - decay) / 8
                print('change alpha')
            dream.dream.data[:] *= decay

            dream.dream.data[:] = self.xp.clip(dream.dream.data[:], 0., 1.)

            turn = 0
            # Initialize canvas of Listener
            canvas = chainer.Variable(
                self.xp.zeros(shape, np.float32), volatile='off')

            # [Speaker]
            # Express the image x compared to canvas
            # Perceive
            hidden_canvas = self.speaker.perceive(
                canvas, turn, train=train)

            image = dream.dream
            image = F.where(self.xp.random.uniform(size=shape) < 0.1,
                            self.xp.zeros(shape).astype('f'), image)
            # Percieve
            hidden_image = self.speaker.perceive(
                image, n_turn, train=train)

            thought = self.speaker.think(
                hidden_image, hidden_canvas, turn, train=train)

            loss = self.speaker.speak_loss(
                thought, sampled_word_idx_seq, n_word=n_word, train=train)
            print(i_iter, loss.data)
            dream.zerograds()
            loss.backward()
            self.zerograds()
            optimizer.update()

            dream.dream.data[:] = self.xp.clip(dream.dream.data[:], 0., 1.)

        return F.clip(dream.dream.data, 0., 1.).data
