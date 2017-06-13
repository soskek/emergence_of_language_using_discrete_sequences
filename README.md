# Communication Model using Discrete Sequences

This is a model for observing emergence of symbol-based messaging protocol.
Autoencoder can be seen regarded as a communication model,
which consists of a speaker and a listener.
Additionally,
if communication messages are discretized (sequentially) in a vocabulary set of a limited size,
the messaging protocol is partly similar to a language.

I experimented the emergence process in an image dataset, MNIST without labels.
An emergence of a language can be observed by tracking
(1) each image which Listener decodes from each sentence and
(2) each image which Speaker decodes each sentence with quite high probability.
Images of (1) can be built easily.
It is difficult to get images of (2).
So, I build it by backpropagation-based image generation, like DeepDream.

An emergence of a language of vocabulary size 2 (binary) and fixed sequence length 8 is as below:

![An emergence of a language on MNIST world.](https://github.com/soskek/interactive_ae/blob/master/emergence_process.gif).

This shows listener's image (1) in grey and speaker's image (2) in red.
Even if the listener's language (inducing grey images) is changed,
the speaker's language interpretation (inducing red images) obeys it, and vice versa.
Starting from left to right (and from top to bottom),
messages are sorted incrementally `00000000`, `00000001`, `00000010`, `00000011`, ..., `11111110`, `11111111`.
You can also see interpolation and adjacent messages often have similar images.

# Model Architecture
- Speaker-Encoder: image *x* -> feature *v*
- Speaker-Decoder: feature *v* -> symbol sequence *z* (sentence)
- Listener-Encoder: symbol sequence *z* -> feature *u*
- Listener-Decoder: feature *u* -> image *y*


# How to learn/emerge
- Speaker encodes an image and decodes a sentence by sampling.
- Listener encodes the sentence and decode an image.
- Listener is trained by minimizing reconstruction error.
- Speaker is trained based on REINFORCE for probability of the sampled sentence.

Each agent never does see any parameters of each other.
This model is trained through discrete sampling process
but not completely decentralized.
After a messaging trial, the listener can see the speaker's image and calculate error,
and the speaker can see the error value itself for REINFORCE.
