#!/usr/bin/python
from keras.preprocessing import sequence
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.core import Activation
from keras.regularizers import l2
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dropout
import numpy as np
import pandas as pd
import sys

np.random.seed(42)

MAX_DIGITS = 105
N_FIB_TERMS = 400
N_3_WINDOWS = N_FIB_TERMS - 3 + 1

def int_to_digits (n):
    digits = np.zeros(N_DIGITS)
    n_str = "{0:0.0f}".format(n)
    for i in xrange(len(n_str)):
        digits[N_DIGITS-1-i] = int(n_str[len(n_str)-1-i])
    return digits

# Read the fibonacci series into memory
fibonacci_ref = pd.read_csv("fibonacci_sequence.csv", header=None)

# Generate the dataset
# Duplicate each sample, with t-1 and t-2 terms swapped 
# TODO: does this have an affect?
X_all = np.zeros([2*N_3_WINDOWS, MAX_DIGITS, 2], dtype=np.float64)
Y_all = np.zeros([2*N_3_WINDOWS, MAX_DIGITS, 10], dtype=np.float64)

for i in xrange(N_3_WINDOWS):

    first = fibonacci_ref.ix[i][1][::-1]
    second = fibonacci_ref.ix[i+1][1][::-1]
    third = fibonacci_ref.ix[i+2][1][::-1]

    # Fill digit of t-2 term (X)
    for j in xrange(len(first)):
        k = float(first[j])
        X_all[i][j][0] = (k-5)/10
        X_all[i+N_3_WINDOWS][j][1] = (k-5)/10

    # Fill digit of t-1 term (X)
    for j in xrange(len(second)):
        k = float(second[j])
        X_all[i][j][1] = (k-5)/10
        X_all[i+N_3_WINDOWS][0] = (k-5)/10

    # Fill digit of sum / t term (Y)
    for j in xrange(len(third)):
        k = int(third[j])
        Y_all[i][j][k] = 1.0
        Y_all[i+N_3_WINDOWS][j][k] = 1.0

# Create training and test sets by randomly sampling the whole data set
test_split = 0.2
test_split_mask = np.random.rand(X_all.shape[0]) < (1-test_split)
X_train = X_all[test_split_mask]
Y_train = Y_all[test_split_mask]
X_test = X_all[~test_split_mask]
Y_test = Y_all[~test_split_mask]

# Define and train the model
model = Sequential()
dropout = 0.33
model.add(LSTM(150, return_sequences=True, input_shape=(MAX_DIGITS, 2)))
model.add(Dropout(dropout))
#model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(10, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print model.summary()

model.fit(X_train, Y_train, nb_epoch=100, batch_size=100)

scores = model.evaluate(X_test, Y_test, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)
