import tensorflow as tf

import os
import time
import argparse
import io

from model import LSTM
from load import DataLoader
from six.moves import cPickle

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='',
        help='the location of the data to train the model on')
    parser.add_argument('--save', type=str, default='save',
        help='the location to save the model at certain checkpoints')
    parser.add_argument('--cell_size', type=int, help='the size of the rnn cell')
    parser.add_argument('--layer_dim', type=int, help='number of layers')
    parser.add_argument('--batch_size', type=int, default=32,
        help='the size of each of the generated batches')
    parser.add_argument('--seq_len', type=int, default=10, help='RNN seq length')
    parser.add_argument('--epochs', type=int, default=5, help='the number of \
        epochs to use')
    parser.add_argument('--save_every', type=int, default=10000, help='the frequency \
        at which to save the model to a designated save directory')
    parser.add_argument('--grad_clip', type=float, default=5., help='the value at \
        which to clip the gradient')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='the learning rate')
    parser.add_argument('--init_from', type=str, default=None, help='continue from \
        saved model at given directory, which must contain files: config.pkl, \
        chars_vocab.pkl, checkpoint, model.cpkt-*.')
    parser.add_argument('--decay_rate', type=float, default=0.97,
        help='decay rate value')
    parser.add_argument('--input_file', type=str, default='input.txt', help='the input to the LSTM')

    args = parser.parse_args()
    train(args)

def train(args):
    loader = DataLoader(args.data, args.input_file, 'vocab.pkl', 'data.npy', \
        args.batch_size, args.seq_len)

    print "loader initialized"

    args.vocab_size = loader.vocab_size

    if args.init_from is not None:
        # TODO: allow continued training
        pass
    else:
        with open(os.path.join(args.save, 'config.pkl'), 'wb') as conf:
            cPickle.dump(args, conf)

        with open(os.path.join(args.save, 'chars_vocab.pkl'), 'wb') as voc:
            cPickle.dump((loader.words, loader.vocab), voc)


    model = LSTM(args)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        init.run()

        saver = tf.train.Saver(tf.all_variables())
        if args.init_from is not None:
            pass # this will fail for sure until implemented
            saver.restore(sess, cpkt.model_checkpoint_path)
        else:
            for e in range(args.epochs):
                sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
                loader.reset()
                state = sess.run(model.initial_state)

                for b in range(loader.num_batches):
                    start = time.time()

                    x, y = loader.next_batch()
                    feed = {model.inputs: x, model.targets: y}

                    for i, (c, h) in enumerate(model.initial_state):
                        feed[c] = state[i].c
                        feed[h] = state[i].h

                    train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                    end = time.time()

                    print("{}/{} (epoch {}), train loss = {:.3f}, time/batch = {:.3f}").format(\
                        e * loader.num_batches + b, args.epochs * loader.num_batches, e, train_loss, end - start)

                    if (e * loader.num_batches + b) % args.save_every == 0 or \
                        (e == args.epochs - 1 and b == loader.num_batches - 1):
                        checkpoint_path = os.path.join(args.save, 'model.cpkt')
                        saver.save(sess, checkpoint_path, global_step=e * loader.num_batches + b)
                        print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
