import os
import codecs

import collections
import numpy as np
import tensorflow as tf

from six.moves import cPickle

class DataLoader(object):

    def __init__(self, direc, input_file, vocab_file, tensor_file,
        batch_size, seq_len, encoding='utf-8'):
        self.direc = direc
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.encoding = encoding

        print "vocab: " + str(os.path.exists(vocab_file))
        print "tensor: " + str(os.path.exists(tensor_file))

        input_file = os.path.join(direc, input_file)
        vocab_file = os.path.join(direc, vocab_file)
        tensor_file = os.path.join(direc, tensor_file)

        print "vocab: " + str(os.path.exists(vocab_file))
        print "tensor: " + str(os.path.exists(tensor_file))

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            self.process(input_file, vocab_file, tensor_file)
        else:
            self.load(vocab_file, tensor_file)

        self.create_batches()
        self.reset()

    def process(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, encoding=self.encoding) as inp:
            data = inp.read()

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])

        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)

        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        with open(vocab_file, 'wb') as voc:
            cPickle.dump(self.chars, voc)

        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load(self, vocab_file, tensor_file):
        with open(vocab_file, 'wb') as voc:
            self.chars = cPickle.load(voc)

        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        self.tensor = np.load(tensor_file)

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_len))
        if self.num_batches == 0:
            raise ValueError("Tune batch_size and seq_len. Not enough data.")

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_len]

        xdata = self.tensor
        ydata = np.copy(self.tensor)

        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]

        self.pointer = 0

        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset(self):
        self.pointer = 0
