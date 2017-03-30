#!/usr/bin/env python
from __future__ import print_function

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


class ImageEncoder(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units):
        super(ImageEncoder, self).__init__(
            l1=L.Linear(n_in, n_middle),
            l2=L.Linear(n_middle, n_units),
            bn=L.BatchNormalization(n_units, use_cudnn=False))

    def __call__(self, x, train=True):
        h = self.l2(F.tanh(self.l1(x)))
        h = F.tanh(self.bn(h, test=not train))
        return h


class ImageDecoder(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units):
        super(ImageDecoder, self).__init__(
            l1=L.Linear(n_units, n_middle),
            l2=L.Linear(n_middle, n_in))

    def __call__(self, x, train=True):
        h = F.sigmoid(self.l2(F.tanh(self.l1(x))))
        return h


def sampling(score):
    xp = cuda.get_array_module(score.data)
    noise = xp.random.gumbel(size=score.shape)
    sampled_idx = F.argmax(score + noise, axis=1)
    return sampled_idx


class Language(chainer.Chain):

    def __init__(self, n_units, n_vocab, agent_type='sender'):
        super(Language, self).__init__(
            meanings=L.EmbedID(n_vocab, n_units))

        if agent_type == 'sender':
            self.add_link('decoder', L.StatefulGRU(n_units, n_units))
            self.add_param('bos', (n_units,), dtype='f')
            self.bos.data[:] = 0
        else:
            self.add_link('encoder', L.StatefulGRU(n_units, n_units))

        self.n_vocab = n_vocab
        self.n_units = n_units

    def interpret_sentence(self, sentence, train=True):
        self.encoder.reset_state()
        for word in sentence:
            sentence_meaning = self.encoder(self.meanings(word))
        self.encoder.reset_state()
        return sentence_meaning

    def decode_word(self, x, train=True):
        score = F.linear(x, self.meanings.W)
        if train:
            word = sampling(score)
        else:
            word = F.argmax(score, axis=1)
        probability_dist = F.softmax(score)
        probability = F.select_item(probability_dist, word)
        return word, probability, probability_dist

    def decode_sentence(self, thought, n_word,
                        teaching_sentence=None, train=True):
        sentence = []
        log_probability = 0.
        p_dists = []
        loss = 0.

        self.decoder.reset_state()
        self.decoder.h = thought
        x_input = F.broadcast_to(
            self.bos, (thought.shape[0], self.bos.shape[0]))

        for i in range(n_word):
            if n_word == 1:
                h = x_input
            else:
                h = self.decoder(x_input)
            word, probability, p_dist = self.decode_word(
                h, train=train)

            if teaching_sentence is None:
                sentence.append(word)
                log_probability += F.log(probability)
                p_dists.append(p_dist)
                x_input = self.meanings(word)
            else:
                t = teaching_sentence[i]
                log_probability = F.log(F.select_item(p_dist, t))
                loss += F.sum(- log_p) / log_p.shape[0]
                x_input = self.meanings(teaching_sentence[i])
        self.decoder.reset_state()

        if teaching_sentence is None:
            return sentence, log_probability, p_dists
        else:
            return loss


class Sender(chainer.Chain):

    def __init__(self, image_encoder, language):
        super(Sender, self).__init__(
            image_encoder=image_encoder,
            language=language)

    def perceive(self, image, train=True):
        return self.image_encoder(image, train=train)

    def speak(self, thought, n_word=3, train=True):
        return self.language.decode_sentence(thought, n_word, train=train)

    def speak_loss(self, thought, sentence, n_word=3, train=True):
        return self.language.decode_sentence(
            thought, n_word, teaching_sentence=sentence, train=train)


class Receiver(chainer.Chain):

    def __init__(self, image_decoder, language):
        super(Receiver, self).__init__(
            image_decoder=image_decoder,
            language=language)

    def paint(self, concept, train=True):
        return self.image_decoder(concept, train=train)

    def listen(self, sentence, train=True):
        return self.language.interpret_sentence(sentence, train=train)


class World(chainer.Chain):

    def __init__(self, n_in, n_middle, n_units, n_vocab, n_word):
        image_encoder = ImageEncoder(n_in, n_middle, n_units)
        image_decoder = ImageDecoder(n_in, n_middle, n_units)

        language_of_sender = Language(n_units, n_vocab, 'sender')
        language_of_receiver = Language(n_units, n_vocab, 'receiver')

        sender = Sender(image_encoder, language_of_sender)
        receiver = Receiver(image_decoder, language_of_receiver)
        assert(sender.language is not receiver.language)

        super(World, self).__init__(
            receiver=receiver,
            sender=sender,
        )

        self.n_word = n_word
        self.n_vocab = n_vocab
        self.train = True
        self.baseline = None

    def __call__(self, image, generate=False, train=True):
        batchsize = image.shape[0]

        # Send and paint
        hidden_image = self.sender.perceive(image, train=train)
        sentence, log_probability, p_dists = self.sender.speak(
            hidden_image, n_word=self.n_word, train=train)
        sentence_meaning = self.receiver.listen(sentence, train=train)
        canvas = self.receiver.paint(sentence_meaning, train=train)

        # Calculate reconstruction error
        raw_loss = F.sum((canvas - image) ** 2, axis=1)
        loss = F.sum(raw_loss) / image.data.size

        # Add (minus) reinforce
        reward = - raw_loss.data
        if self.baseline is None:
            self.baseline = self.xp.mean(reward)
        reinforce = F.sum(
            log_probability * (reward - self.baseline)) / batchsize
        reinforce_loss = - reinforce * 1.

        # Update baseline
        self.baseline = self.baseline * 0.99 + self.xp.mean(reward) * 0.01

        if generate:
            return [i.data for i in sentence], \
                log_probability.data, \
                F.clip(canvas, 0., 1.).data
        else:
            return loss, reinforce_loss

    def generate_from_sentence(self, sentence, shape):
        sentence_meaning = self.receiver.listen(sentence, train=False)
        canvas = self.receiver.paint(sentence_meaning, train=False)
        return F.clip(canvas, 0., 1.).data

    def infer(self, sentence, shape):
        batchsize = shape[0]

        dream = chainer.Link(dream=shape)
        if self._device_id is not None:
            dream.to_gpu(self._device_id)
            sentence = [self.xp.array(x) for x in sentence]

        dream.dream.data[:] = 0.01
        optimizer = chainer.optimizers.Adam(0.1)
        optimizer.setup(dream)
        optimizer.add_hook(optimizer.WeightDecay(0.01))

        n_iter = 200
        for i_iter in range(n_iter):
            dream.dream.data[:] = self.xp.clip(dream.dream.data, 0., 1.)

            image = dream.dream * \
                (self.xp.random.rand(*shape) > 0.01)  # add drop

            hidden_image = self.sender.perceive(image, train=False)
            loss = self.sender.speak_loss(
                hidden_image, sentence, n_word=self.n_word, train=False)

            print(i_iter, loss.data)
            dream.zerograds()
            loss.backward()
            self.zerograds()
            optimizer.update()

        return self.xp.clip(dream.dream.data, 0., 1.)
