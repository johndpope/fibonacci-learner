#!/usr/bin/python
import numpy as np
import tensorflow as tf 

np.random.seed()

# Constants
batch_size = 25
sequence_length = 50
learning_rate = 0.1
n_fib_terms = 250
n_training_sequences = 1000
n_batches = n_training_sequences / batch_size
training_iterations = 100000

# Generate the dataset
fib_ref = np.zeros(n_fib_terms, dtype=np.float64)
fib_ref[0] = 1.0
fib_ref[1] = 1.0

for i in xrange(2, n_fib_terms):
    fib_ref[i] = fib_ref[i-1] + fib_ref[i-2]

# The dimensions below make it easier to split the input for the RNN
training_fib_sequences_x = np.zeros([n_batches,sequence_length,batch_size])
training_fib_sequences_y = np.zeros([n_batches,batch_size])

for batch in xrange(n_batches):
    for ex_in_batch in xrange(batch_size):
        fib_index_offset = np.random.randint(0, n_fib_terms-sequence_length-1)
        for i in xrange(sequence_length):
            training_fib_sequences_x[batch][i][ex_in_batch] = fib_ref[i+fib_index_offset]
        training_fib_sequences_y[batch][ex_in_batch] = fib_ref[i+fib_index_offset+1]

# Build the computational graph
x = tf.placeholder(tf.float64, [sequence_length,batch_size])
y = tf.placeholder(tf.float64, [batch_size])

lstm = tf.nn.rnn_cell.BasicLSTMCell(1, forget_bias=0.0, state_is_tuple=True)
outputs, states = tf.nn.rnn(lstm, tf.split(0,sequence_length,x), dtype=tf.float64)
print tf.shape(outputs)

cost = tf.reduce_mean(tf.nn.l2_loss(outputs, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

def main(_):
    # Set up the graph
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        step = 1

        while batch_size*step < training_iterations:
            batch_index = ((step-1)*batch_size) % n_batches
            batch_x = np.take(training_fib_sequences_x, range(0,sequence_length), axis=0)#[batch_index]
            batch_y = np.take(training_fib_sequences_y, range(0,batch_size), axis=0)#[batch_index]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            step += 1

if __name__ == '__main__':
    tf.app.run()

# RESOURCES
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py
# https://www.tensorflow.org/versions/r0.10/tutorials/recurrent/index.html
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
# http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/6_lstm.ipynb
