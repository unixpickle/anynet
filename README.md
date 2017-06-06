# anynet [![GoDoc](https://godoc.org/github.com/unixpickle/anynet?status.svg)](https://godoc.org/github.com/unixpickle/anynet)

**anynet** is a [neural network](https://en.wikipedia.org/wiki/Artificial_neural_network) framework based on [anydiff](https://github.com/unixpickle/anydiff) and [anyvec](https://github.com/unixpickle/anyvec).

# Supported features

*anynet* ships with a ton of built-in features:

 * Feed-forward neural networks
   * Fully-connected layers
   * Convolution
   * Dropout
   * Max/Mean pooling
   * Batch normalization
   * Residual connections
   * Image scaling
   * Image padding
 * Recurrent neural networks
   * LSTM
   * Bidirectional RNNs
   * npRNN and IRNN (vanilla RNNs with ReLU activations)
 * Training setups
   * Vector-to-vector (standard feed-forward)
   * Sequence-to-sequence (standard RNN)
   * Sequence-to-vector
   * Connectionist Temporal Classification
 * Miscellaneous
   * Gumbel Softmax

Plenty of stuff is missing from the above list. Luckily, it's easy to write new APIs on top of *anynet*. Here is a non-exhaustive list of packages that work with *anynet*:

 * [unixpickle/anyrl](https://github.com/unixpickle/anyrl) - deep reinforcement learning
 * [unixpickle/lazyseq](https://github.com/unixpickle/lazyseq) - memory-efficient RNNs
 * [unixpickle/attention](https://github.com/unixpickle/attention) - attention mechanisms
 * [unixpickle/rwa](https://github.com/unixpickle/rwa) - a new attention-based RNN architecture

# TODO

Here are some minor things I'd like to get done at some point. None of these are very urgent, as *anynet* is already complete for the most part.

 * anyrnn
   * Tests comparing LSTM outputs to another implementation
   * GRU (gated recurrent units)
 * anysgd
   * Gradient clipping
   * Marshalling for RMSProp
   * Marshalling for Momentum
