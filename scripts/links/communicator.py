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


class Speaker(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units):
        super(Speaker, self).__init__(
            sensor=Sensor(n_in, n_middle),

            l1_first=L.Linear(n_middle, n_units),
            l1_next=L.Linear(n_middle * 2, n_units),
            l2_first=L.Linear(n_units, n_units),
            l2_next=L.Linear(n_units, n_units),
            bn1_first=L.BatchNormalization(n_units, use_cudnn=False),
            bn1_next=L.BatchNormalization(n_units, use_cudnn=False),
            bn2_first=L.BatchNormalization(n_units, use_cudnn=False),
            bn2_next=L.BatchNormalization(n_units, use_cudnn=False),
        )
        self.act = F.tanh

    def __call__(self, x, z, language, turn, n_word=3, train=True, with_recon=False):
        if with_recon:
            true_image, rec_loss = self.sensor(
                x, true=True, train=train, with_recon=True)
        else:
            true_image = self.sensor(x, true=True, train=train)

        if turn == 0:
            h1 = self.act(self.bn1_first(
                self.l1_first(true_image), test=not train))
            thought = self.act(self.bn2_first(
                self.l2_first(h1), test=not train))
            rec_loss_now, rec_loss = 0., 0.
        else:
            if with_recon:
                now_image, rec_loss_now = self.sensor(
                    z, true=False, train=train, with_recon=True)
            else:
                now_image = self.sensor(z, true=False, train=train)
                rec_loss_now = 0.
            comparison = F.concat([true_image, now_image], axis=1)
            h1 = self.act(self.bn1_next(
                self.l1_next(comparison), test=not train))
            thought = self.act(self.bn2_next(self.l2_next(h1), test=not train))

        sampled_word_idx_seq, total_log_probability = language.decode_thought(
            thought, n_word, turn, train=train)

        if with_recon:
            return sampled_word_idx_seq, total_log_probability, rec_loss + rec_loss_now
        else:
            return sampled_word_idx_seq, total_log_probability


class Sensor(chainer.Chain):

    def __init__(self, n_in, n_units):
        super(Sensor, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_units),

            rec_l1=L.Linear(n_units, n_units),
            rec_l2=L.Linear(n_units, n_units),
            rec_l3=L.Linear(n_units, n_in),
        )
        self.act = F.relu

    def __call__(self, x, true=True, train=True, with_recon=False):
        h1 = self.act(self.l1(x))
        h2 = self.act(self.l2(h1))
        h3 = self.act(self.l3(h2))

        if with_recon:
            return h3, self.reconstruct(x, h3, true=true, train=train)
        else:
            return h3

    def reconstruct(self, t, h, true=True, train=True):
        h1 = self.act(self.rec_l1(h))
        h2 = self.act(self.rec_l2(h1))
        h3 = self.act(self.rec_l3(h2))
        return F.mean_squared_error(t, h3)


class Language(chainer.Chain):

    def __init__(self, n_units, n_vocab):
        super(Language, self).__init__(
            definition=L.EmbedID(n_vocab, n_units),
            expression=L.Linear(n_units, n_vocab, nobias=True),
            interpreter=L.StatefulGRU(n_units, n_units),
            decoder=L.StatefulGRU(n_units, n_units),
            bn_first_interpreter=L.BatchNormalization(
                n_units, use_cudnn=False),
            bn_next_interpreter=L.BatchNormalization(n_units, use_cudnn=False),
            bn_first_expression=L.BatchNormalization(n_vocab, use_cudnn=False),
            bn_next_expression=L.BatchNormalization(n_vocab, use_cudnn=False),
        )
        self.n_vocab = n_vocab
        self.n_units = n_units

        self.add_param('eos', (n_units,), dtype='f')
        self.eos.data[:] = 0
        self.add_param('bos', (n_units,), dtype='f')
        self.bos.data[:] = 0

    def decode_word(self, x, turn, train=True):
        probability = F.softmax(self.expression(x))

        prob_data = probability.data
        if self.xp != np:
            prob_data = self.xp.asnumpy(prob_data)
        batchsize = x.data.shape[0]

        if train:
            sampled_ids = np.zeros((batchsize,), np.int32)
            for i_batch, one_prob_data in enumerate(prob_data):
                sampled_ids[i_batch] = np.random.choice(
                    self.n_vocab, p=one_prob_data)
        else:
            sampled_ids = np.zeros((batchsize,), np.int32)
            for i_batch, one_prob_data in enumerate(prob_data):
                sampled_ids[i_batch] = np.argmax(
                    one_prob_data).astype(np.int32)

        if self.xp != np:
            sampled_ids = self.xp.array(sampled_ids)
        sampled_ids = chainer.Variable(sampled_ids, volatile='auto')
        sampled_probability = F.select_item(probability, sampled_ids)

        return sampled_ids, sampled_probability

    def interpret_word(self, x):
        return self.definition(x)

    def interpret_sentence(self, x_seq, turn, train=True):
        self.interpreter.reset_state()

        for message_word in x_seq:
            message_meaning = self.interpreter(
                self.interpret_word(message_word))

        if turn == 0:
            message_meaning = self.bn_first_interpreter(
                message_meaning, test=not train)
        else:
            message_meaning = self.bn_next_interpreter(
                message_meaning, test=not train)

        self.interpreter.reset_state()
        return message_meaning

    def decode_thought(self, thought, n_word, turn, train=True):
        sampled_word_idx_seq = []
        total_log_probability = 0.
        if n_word == 1:
            sampled_word_idx, probability = self.decode_word(
                thought, turn, train=train)
            sampled_word_idx_seq.append(sampled_word_idx)
            total_log_probability += F.log(probability)
        else:
            self.decoder.reset_state()
            self.decoder.h = thought
            bos = F.broadcast_to(
                self.bos, (thought.data.shape[0], len(self.bos.data)))
            x_input = bos
            for i in range(n_word):
                h = self.decoder(x_input)
                sampled_word_idx, probability = self.decode_word(
                    h, turn, train=train)
                sampled_word_idx_seq.append(sampled_word_idx)
                total_log_probability += F.log(probability)
                x_input = self.interpret_word(sampled_word_idx)
            self.decoder.reset_state()
        return sampled_word_idx_seq, total_log_probability


class Listener(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units):
        super(Listener, self).__init__(
            sensor=Sensor(n_in, n_middle),

            l1_meaning=L.Linear(n_units, n_middle + n_units),
            l1_addnext=L.Linear(n_middle, n_middle + n_units),
            l2=L.Linear(n_middle + n_units, n_middle),
            bn2_first=L.BatchNormalization(n_middle, use_cudnn=False),
            bn2_next=L.BatchNormalization(n_middle, use_cudnn=False),
            l3=L.Linear(n_middle, n_middle),
            l4=L.Linear(n_middle, n_in),
        )
        self.act = F.relu

    def __call__(self, canvas, message_sentence, language, turn, train=True, with_recon=False):
        message_meaning = language.interpret_sentence(
            message_sentence, turn, train=train)
        rec_loss = 0.
        if turn == 0:
            h1 = self.act(self.l1_meaning(message_meaning))
            plus_draw = F.tanh(self.l4(self.act(self.l3(self.act(
                self.bn2_first(self.l2(h1), test=not train))))))
        else:
            h1 = self.l1_meaning(message_meaning)
            if with_recon:
                hidden_canvas, rec_loss = self.sensor(
                    canvas, true=False, train=train, with_recon=True)
            else:
                hidden_canvas = self.sensor(
                    canvas, true=False, train=train, with_recon=False)
            h1 = self.act(h1 + self.l1_addnext(hidden_canvas))
            plus_draw = F.tanh(self.l4(self.act(self.l3(self.act(
                self.bn2_next(self.l2(h1), test=not train))))))
        if with_recon:
            return plus_draw ** 3, rec_loss
        else:
            return plus_draw ** 3


class World(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units, n_vocab, n_word, n_turn):
        super(World, self).__init__(
            language=Language(n_units, n_vocab),
            listener=Listener(n_in, n_middle, n_units),
            speaker=Speaker(n_in, n_middle, n_units),
        )
        self.n_turn = n_turn
        self.n_word = n_word

        self.baseline_reward = [None for _ in range(10)]
        self.train = True

    def __call__(self, image, generate=False):
        n_turn, n_word = self.n_turn, self.n_word
        accum_loss = 0.

        sub_accum_loss = 0.
        accum_reward_obj = 0.
        accum_rec_loss = 0.

        batchsize = image.data.shape[0]
        sentence_history = []
        log_prob_history = []
        canvas_history = []

        # Initialize canvas of Listener
        #canvas = chainer.Variable(self.xp.zeros(image.data.shape, np.float32), volatile='auto')
        canvas = chainer.Variable(self.xp.ones(
            image.data.shape, np.float32), volatile='auto')

        loss_list = []
        raw_loss_list = []

        for i in range(n_turn):

            # Express the image x compared to canvas by Speaker
            sampled_word_idx_seq, log_probability, rec_loss = self.speaker(
                image, canvas, self.language, turn=i, n_word=n_word, with_recon=True, train=self.train)
            accum_rec_loss += rec_loss

            # Interpret the expression
            # Paint it into canvas
            plus_draw, rec_loss = self.listener(
                canvas, sampled_word_idx_seq, self.language, turn=i, with_recon=True, train=self.train)
            accum_rec_loss += rec_loss

            canvas = canvas + plus_draw
            canvas = F.clip(canvas, 0., 1.) * 0.9 + canvas * 0.1

            if generate:
                canvas_history.append(F.clip(canvas, 0., 1.).data)

            sentence_history.append(sampled_word_idx_seq)
            log_prob_history.append(log_probability)

            # Calculate comunication loss
            raw_loss = F.sum((canvas - image) ** 2, axis=1)
            raw_loss_list.append(raw_loss)
            loss = F.sum(raw_loss) / image.data.size
            reporter.report({'l{}'.format(i): loss}, self)
            loss_list.append(loss)
            reporter.report({'p{}'.format(i): self.xp.exp(
                log_probability.data.mean())}, self)

        #"""
        decay = 0.5
        accum_loss_pre_step = sum(
            loss_list[j] * decay ** (n_turn - j - 1) for j in range(n_turn - 1))
        sub_accum_loss += accum_loss_pre_step
        #"""

        accum_loss += loss_list[n_turn - 1]

        """
        # modification loss
        margin = 0.1
        sub_accum_loss += sum(F.relu(margin + loss_list[i] - loss_list[i-1].data) for i in range(1, n_turn))
        """

        reward = (1. - raw_loss_list[-1]).data
        #reward = (1.-raw_loss_list[-1])

        i = 0
        if self.baseline_reward[i] is None:
            obj = F.sum(sum(log_prob_history) / n_word *
                        (reward - 0.)) / reward.size
        else:
            obj = F.sum(sum(log_prob_history) / n_word *
                        (reward - self.baseline_reward[i])) / reward.size
        #reward = reward.data

        accum_reward_obj += obj
        reporter.report({'nr{}'.format(i): obj}, self)

        sub_accum_loss -= accum_reward_obj * 0.00001

        if self.train:
            if self.baseline_reward[0] is None:
                self.baseline_reward[0] = (self.xp.sum(reward) / reward.size)
            else:
                self.baseline_reward[0] = 0.95 * self.baseline_reward[0] \
                    + 0.05 * (self.xp.sum(reward) / reward.size)

        reporter.report({'loss': accum_loss}, self)
        reporter.report({'reward': accum_reward_obj}, self)
        reporter.report({'reconst': accum_rec_loss}, self)
        sub_accum_loss += accum_rec_loss * 0.01

        def orthogonal_regularizer(M):
            MM = F.matmul(M, F.transpose(M))
            iden = self.xp.identity(MM.shape[0])
            norm_loss = F.sum((iden - MM * iden) ** 2)
            return F.sum((MM - MM * iden) ** 2) + norm_loss

        """
        orthogonal_loss = orthogonal_regularizer(self.language.expression.W) + \
        #                  orthogonal_regularizer(self.language.definition.W)
        reporter.report({'ortho': orthogonal_loss}, self)
        sub_accum_loss += 0.001 * orthogonal_loss
        """

        """
        def word_l2(idx):
            definition_l2 = self.language.definition(idx) ** 2
            expression_l2 = F.embed_id(idx, self.language.expression.W) ** 2
            return definition_l2 + expression_l2
        L2norm_used_embed = F.sum(sum([sum([word_l2(i) for i in s])
                                       for s in sentence_history]))
        sub_accum_loss += 0.0001 * L2norm_used_embed / batchsize
        """

        reporter.report({'total': accum_loss}, self)

        accum_loss += sub_accum_loss
        self.sub_accum_loss = sub_accum_loss.data

        if generate:
            return [[i.data for i in s] for s in sentence_history], [lp.data for lp in log_prob_history], canvas_history
        else:
            return accum_loss
