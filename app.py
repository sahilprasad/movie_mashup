import os
import tensorflow as tf
from model import LSTM
from six.moves import cPickle
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('home.html')

@app.route('/action')
def get_text():
    # takes the most recently saved model checkpoint of futurama dataset, and
    # uses it to generate fixed-size text to be sent back to FE
    with open(os.path.join('save/futurama', 'config.pkl'), 'rb') as config:
        saved_args = cPickle.load(config)

    with open(os.path.join('save/futurama', 'chars_vocab.pkl'), 'rb') as vocab_f:
        words, vocab = cPickle.load(vocab_f)

    model = LSTM(saved_args, test=True)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()

        init.run()

        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state('save/futurama')

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            ret = model.sample(sess, words, vocab, 500, prime=' ',
                sampling_type=1)
            print ret
            return ret

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
