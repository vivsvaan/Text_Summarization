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


## Usage
#### Prepare data
Dataset is available at [harvardnlp/sent-summary](https://github.com/harvardnlp/sent-summary). Locate the summary.tar.gz file in project root directory. Then,
```
$ python prep_data.py
```
To use Glove pre-trained embedding, download it via
```
$ python prep_data.py --glove
```

#### Train
We used ```sumdata/train/train.article.txt``` and ```sumdata/train/train.title.txt``` for training data. To train the model, use
```
$ python train.py
```
To use Glove pre-trained vectors as initial embedding, use
```
$ python train.py --glove
```

#### Additional Hyperparamters
```
$ python train.py -h
usage: train.py [-h] [--num_hidden NUM_HIDDEN] [--num_layers NUM_LAYERS]
                [--beam_width BEAM_WIDTH] [--glove]
                [--embedding_size EMBEDDING_SIZE]
                [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                [--num_epochs NUM_EPOCHS] [--keep_prob KEEP_PROB] [--toy]

optional arguments:
  -h, --help            show this help message and exit
  --num_hidden NUM_HIDDEN
                        Network size.
  --num_layers NUM_LAYERS
                        Network depth.
  --beam_width BEAM_WIDTH
                        Beam width for beam search decoder.
  --glove               Use glove as initial word embedding.
  --embedding_size EMBEDDING_SIZE
                        Word embedding size.
  --learning_rate LEARNING_RATE
                        Learning rate.
  --batch_size BATCH_SIZE
                        Batch size.
  --num_epochs NUM_EPOCHS
                        Number of epochs.
  --keep_prob KEEP_PROB
                        Dropout keep prob.
  --toy                 Use only 5K samples of data

```


