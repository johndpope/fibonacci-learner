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
# 105 digits for first 500 fib #s

WINDOW_SIZE = 25
N_DIGITS = 105
N_FIB_TERMS = 100

def number_to_digits (n):
    digits = np.zeros(N_DIGITS)
    n_str = "{0:0.0f}".format(n)
    for i in xrange(len(n_str)):
        digits[N_DIGITS-1-i] = int(n_str[len(n_str)-1-i])
    return digits

# Generate a big fibonacci sequence to sample
fib_ref = np.empty(N_FIB_TERMS, dtype=np.float64)
fib_ref[0] = 1.0
fib_ref[1] = 1.0

for i in xrange(2, N_FIB_TERMS):
    fib_ref[i] = fib_ref[i-1] + fib_ref[i-2]

# Generate a data set using slices of the reference data set of the same length
n_all_sequences = N_FIB_TERMS - (WINDOW_SIZE+1)
X_all = np.empty([n_all_sequences, WINDOW_SIZE, 2*N_DIGITS], dtype=np.float64)
y_all = np.empty([n_all_sequences, WINDOW_SIZE, 1*N_DIGITS], dtype=np.float64)

for i in xrange(n_all_sequences):
    for j in xrange(WINDOW_SIZE):

        # Fill first term of X sample
        T_n_2 = number_to_digits(fib_ref[i+j])
        for k in xrange(N_DIGITS):
            X_all[i][j][k] = T_n_2[k]

        # Fill second term of X sample
        T_n_1 = number_to_digits(fib_ref[i+j+1])
        for k in xrange(N_DIGITS):
            X_all[i][j][k+N_DIGITS] = T_n_1[k]

        # Fill Y with third term
        T_n = number_to_digits(fib_ref[i+j+2])
        for k in xrange(N_DIGITS):
            y_all[i][j][k] = T_n[k]

# Create training and test sets by randomly sampling the whole data set
test_split = 0.2
test_split_mask = np.random.rand(X_all.shape[0]) < (1-test_split)
X_train = X_all[test_split_mask]
y_train = y_all[test_split_mask]
X_test = X_all[~test_split_mask]
y_test = y_all[~test_split_mask]

# Define and train the model
model = Sequential()
dropout = 0.0
model.add(Bidirectional(LSTM(150, return_sequences=True), input_shape=(WINDOW_SIZE,2*N_DIGITS)))
model.add(TimeDistributed(Dense(N_DIGITS, activation='relu')))
#model.add(Dropout(dropout))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print model.summary()

model.fit(X_train, y_train, nb_epoch=500, batch_size=50)

scores = model.evaluate(X_test, y_test, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)
