# Text_Summarization
Simple Tensorflow implementation of text summarization using [seq2seq library](https://www.tensorflow.org/api_guides/python/contrib.seq2seq).

#### Model
Encoder-Decoder model with attention mechanism.

#### Word Embedding
Used [Glove pre-trained vectors](https://nlp.stanford.edu/projects/glove/) to initialize word embedding.

#### Encoder
Used LSTM cell with [stack_bidirectional_dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn).

#### Decoder
Used LSTM [BasicDecoder](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder) for training, and [BeamSearchDecoder](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BeamSearchDecoder) for inference.

#### Attention Mechanism
Used [BahdanauAttention](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BahdanauAttention) with weight normalization.


