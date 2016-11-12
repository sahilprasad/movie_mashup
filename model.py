import tensorflow as tf
import numpy as np

from tensorflow.python.ops import rnn_cell, seq2seq

class LSTM(object):

    DEFAULT_VOCAB_SIZE = 5000
    DEFAULT_CELL_SIZE = 128
    DEFAULT_LAYER_DIM = 3
    DEFAULT_BATCH_SIZE = 4
    DEFAULT_SEQ_LEN = 4
    DEFAULT_GRAD_CLIP = 5
    DEFAULT_LEARNING_RATE = 0.002

    def __init__(self, args, test=False):
        self.args = args
        self.test = test

        if self.test:
            self.args.batch_size = 1
            self.args.seq_len = 1

        self._build()

    def _settle_constants(self):
        self.grad_clip = self.args.grad_clip or self.DEFAULT_GRAD_CLIP
        self.vocab_size = self.args.vocab_size or self.DEFAULT_VOCAB_SIZE
        self.cell_size = self.args.cell_size or self.DEFAULT_CELL_SIZE
        self.batch_size = self.args.batch_size or self.DEFAULT_BATCH_SIZE
        self.layer_dim = self.args.layer_dim or self.DEFAULT_LAYER_DIM
        self.seq_len = self.args.seq_len or self.DEFAULT_SEQ_LEN
        self.learning_rate = self.args.learning_rate or self.DEFAULT_LEARNING_RATE

    def _build(self):
        self._settle_constants()

        lstm_cell = rnn_cell.BasicLSTMCell(self.cell_size)
        self.cell = cell = rnn_cell.MultiRNNCell([lstm_cell] * self.layer_dim)

        self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('lstm'):
            self.weights = tf.get_variable('weights', [self.cell_size,
                self.vocab_size])
            self.biases = tf.get_variable('biases', [self.vocab_size])

            with tf.device('/cpu:0'):
                self.embedding = tf.get_variable('embedding', [
                    self.vocab_size, self.cell_size,
                ])
                embeds = tf.nn.embedding_lookup(self.embedding, self.inputs)

                embeds = tf.split(1, self.seq_len, embeds)
                embeds = [tf.squeeze(_input, [1]) for _input in embeds]


        def loop(prev, _):
            prev = tf.matmul(prev, self.weights) + self.biases
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs_split, last_state = seq2seq.rnn_decoder(
            embeds,
            self.initial_state,
            self.cell,
            loop_function=loop if self.test else None,
            scope='lstm'
        )

        output = tf.reshape(tf.concat(1, outputs_split), [-1, self.cell_size]) # TODO

        self.logits = tf.matmul(output, self.weights) + self.biases
        self.probs = tf.nn.softmax(self.logits)

        loss_func = seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([self.batch_size * self.seq_len])],
            self.vocab_size,
        )

        self.cost = tf.reduce_sum(loss_func) / self.batch_size / self.seq_len
        self.final_state = last_state

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
            self.grad_clip) # TODO

        self.lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
            name='optimizer')
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), name='train_op')

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.inputs: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(t, np.random.rand(1) * s))

        ret = prime
        char = prime[-1]

        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.inputs: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)

            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred

        return ret
