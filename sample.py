import tensorflow as tf

import numpy as np
from six.moves import cPickle

import os
import argparse

from model import LSTM

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', type=str, default='save',
        help='save dir for checkpoint-satisfying models')
    parser.add_argument('-n', type=int, default=1000,
        help='number of chars to sample')
    parser.add_argument('--prime', type=str, default='Q',
        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
        help='0 for max at each step, 1 to sample each step, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)

def sample(args):
    with open(os.path.join(args.save, 'config.pkl'), 'rb') as config:
        saved_args = cPickle.load(config)

    with open(os.path.join(args.save, 'chars_vocab.pkl'), 'rb') as vocab_f:
        chars, vocab = cPickle.load(vocab_f)

    model = LSTM(saved_args, test=True)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()

        init.run()

        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, chars, vocab, args.n, prime=args.prime,
                sampling_type=args.sample))

if __name__ == '__main__':
    main()
