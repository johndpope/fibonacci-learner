#!/usr/bin/python
from keras.preprocessing import sequence
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense
from keras.layers.core import Activation
from keras.regularizers import l2
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dropout
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import sys

# The model takes 2 numbers and predicts their sum
# Samples are modelled as 2 sequences of vectors
# X vector has 2 vectors per frame (one for each operand)
# vector is a one-hot embedding of the digit ID

# Notes
# - performs better than sequence of raw digits and sequence of normed raw digits
# - more stable and realistic than predicting real output at each frame
# - gets to 20-35% accuracy after 80 epochs

# Todo
# - train on more data (limited by int64)
# - try binary representation (limited by int64)

################################################################################

np.random.seed(42)

MAX_DIGITS = 105
N_FIB_TERMS = 400
N_3_WINDOWS = N_FIB_TERMS - 3 + 1

def int_to_digits (n, n_digits=MAX_DIGITS):
    digits = np.zeros(n_digits)
    n_str = "{0:0.0f}".format(n)
    for i in xrange(len(n_str)):
        digits[N_DIGITS-1-i] = int(n_str[len(n_str)-1-i])
    return digits

# Read the fibonacci series into memory
fibonacci_ref = pd.read_csv("fibonacci_sequence.csv", header=None)

# Generate the dataset
# Duplicate each sample, with t-1 and t-2 terms swapped 
# TODO: does this have an affect?
X_fib_all = np.zeros([2*N_3_WINDOWS, MAX_DIGITS, 20], dtype=np.float64)
Y_fib_all = np.zeros([2*N_3_WINDOWS, MAX_DIGITS, 10], dtype=np.float64)

for i in xrange(N_3_WINDOWS):

    first = fibonacci_ref.ix[i][1][::-1]
    second = fibonacci_ref.ix[i+1][1][::-1]
    third = fibonacci_ref.ix[i+2][1][::-1]

    # Fill digit of t-2 term (X)
    for j in xrange(len(first)):
        k = int(first[j])
        X_fib_all[i][j][k] = 1.0
        X_fib_all[i+N_3_WINDOWS][j][k+10] = 1.0

    # Fill digit of t-1 term (X)
    for j in xrange(len(second)):
        k = int(second[j])
        X_fib_all[i][j][k+10] = 1.0
        X_fib_all[i+N_3_WINDOWS][j][k] = 1.0

    # Fill digit of sum / t term (Y)
    for j in xrange(len(third)):
        k = int(third[j])
        Y_fib_all[i][j][k] = 1.0
        Y_fib_all[i+N_3_WINDOWS][j][k] = 1.0

# Generate extra data
N_EXTRA_DATA_POINTS = 10000
EXTRA_DATA_MAX = np.iinfo(np.int64).max / 2

X_extra_all = np.zeros([N_EXTRA_DATA_POINTS, MAX_DIGITS,20])
Y_extra_all = np.zeros([N_EXTRA_DATA_POINTS, MAX_DIGITS,10])

for i in xrange(N_EXTRA_DATA_POINTS):

    first_n = np.random.randint(0,EXTRA_DATA_MAX)
    second_n = np.random.randint(0,EXTRA_DATA_MAX)
    third_n = first_n + second_n

    first = "{0:0.0f}".format(first_n)[::-1]
    second = "{0:0.0f}".format(second_n)[::-1]
    third = "{0:0.0f}".format(third_n)[::-1]

    # Fill digit of t-2 term (X)
    for j in xrange(len(first)):
        k = int(first[j])
        X_extra_all[i][j][k] = 1.0

    # Fill digit of t-1 term (X)
    for j in xrange(len(second)):
        k = int(second[j])
        X_extra_all[i][j][k+10] = 1.0

    # Fill digit of sum / t term (Y)
    for j in xrange(len(third)):
        k = int(third[j])
        Y_extra_all[i][j][k] = 1.0

# Create training and test sets by randomly sampling the whole data set
test_split = 0.2
#test_split_mask = np.random.rand(X_extra_all.shape[0]) < (1-test_split)
#X_train = X_extra_all[test_split_mask]
#Y_train = Y_extra_all[test_split_mask]
#X_test = X_extra_all[~test_split_mask]
#Y_test = Y_extra_all[~test_split_mask]

# Define and train the model
model = Sequential()
dropout = 0.33
model.add(LSTM(150, return_sequences=True, input_shape=(MAX_DIGITS, 20)))
model.add(Dropout(dropout))
#model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(10, activation='softmax')))

opt = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print model.summary()

model.fit(X_extra_all, Y_extra_all, nb_epoch=50, batch_size=500)

scores = model.evaluate(X_fib_all, Y_fib_all, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)
