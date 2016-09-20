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
# vector is a one-hot embedding of the bit (one or zero)
# seems to get to 96% after 10 epochs

################################################################################

np.random.seed(42)

MAX_DIGITS = 19 # sys.maxint has 
MAX_BITS = 63
N_FIB_TERMS = 92 # up to an including 92nd fib. has <= 19 digits
N_3_WINDOWS = N_FIB_TERMS - 3 + 1

# Read the fibonacci series into memory
fibonacci_ref = pd.read_csv("fibonacci_sequence.csv", header=None)

# Generate the dataset
# Duplicate each sample, with t-1 and t-2 terms swapped 
# TODO: does this have an affect?
X_fib_all = np.zeros([2*N_3_WINDOWS, MAX_BITS, 2], dtype=np.int64)
Y_fib_all = np.zeros([2*N_3_WINDOWS, MAX_BITS, 1], dtype=np.int64)

for i in xrange(N_3_WINDOWS):

    first = "{0:b}".format(int(fibonacci_ref.ix[i][1]))[::-1]
    second = "{0:b}".format(int(fibonacci_ref.ix[i+1][1]))[::-1]
    third = "{0:b}".format(int(fibonacci_ref.ix[i+2][1]))[::-1]

    # Fill digit of t-2 term (X)
    for j in xrange(len(first)):
        X_fib_all[i][j][0] = float(first[j])
        X_fib_all[i+N_3_WINDOWS][j][1] = float(first[j])

    # Fill digit of t-1 term (X)
    for j in xrange(len(second)):
        X_fib_all[i][j][1] = float(second[j])
        X_fib_all[i+N_3_WINDOWS][j][0] = float(second[j])

    # Fill digit of sum / t term (Y)
    for j in xrange(len(third)):
        Y_fib_all[i][j] = float(third[j])
        Y_fib_all[i+N_3_WINDOWS][j] = float(third[j])

# Generate extra data
N_EXTRA_DATA_POINTS = 5000
EXTRA_DATA_MAX = np.iinfo(np.int64).max / 2

X_extra_all = np.zeros([N_EXTRA_DATA_POINTS, MAX_BITS,2], dtype=np.int64)
Y_extra_all = np.zeros([N_EXTRA_DATA_POINTS, MAX_BITS,1], dtype=np.int64)

for i in xrange(N_EXTRA_DATA_POINTS):

    first_n = np.random.randint(0,EXTRA_DATA_MAX)
    second_n = np.random.randint(0,EXTRA_DATA_MAX)
    third_n = first_n + second_n

    first = "{0:b}".format(first_n)[::-1]
    second = "{0:b}".format(second_n)[::-1]
    third = "{0:b}".format(third_n)[::-1]

    # Fill digit of t-2 term (X)
    for j in xrange(len(first)):
        X_extra_all[i][j][0] = float(first[j])

    # Fill digit of t-1 term (X)
    for j in xrange(len(second)):
        X_extra_all[i][j][1] = float(second[j])

    # Fill digit of sum / t term (Y)
    for j in xrange(len(third)):
        Y_extra_all[i][j] = float(third[j])

# Create training and test sets by randomly sampling the whole data set
X_train_all = np.vstack(tuple([X_extra_all]+[X_fib_all]*10))
Y_train_all = np.vstack(tuple([Y_extra_all]+[Y_fib_all]*10))
test_split = 0.2
test_split_mask = np.random.rand(X_train_all.shape[0]) < (1-test_split)
X_train = X_train_all[test_split_mask]
Y_train = Y_train_all[test_split_mask]
X_test = X_train_all[~test_split_mask]
Y_test = Y_train_all[~test_split_mask]

# Define and train the model
model = Sequential()
dropout = 0.33
model.add(LSTM(150, return_sequences=True, input_shape=(MAX_BITS, 2)))
model.add(Dropout(dropout))
#model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

opt = Adam(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
print model.summary()

model.fit(X_train, Y_train, nb_epoch=10, batch_size=100)

print "~~~ Test Set Scores ~~~"
scores = model.evaluate(X_test, Y_test, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)
print "~~~ All Fibonacci Scores ~~~"
scores = model.evaluate(X_fib_all, Y_fib_all, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)

model.save("fib_adder_10ep.h5")
