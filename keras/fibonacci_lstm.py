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

# Generate a big fibonacci sequence to sample
n_terms = 100
fib_ref = np.empty(n_terms, dtype=np.float64)
fib_ref[0] = 1.0
fib_ref[1] = 1.0

for i in xrange(2, n_terms):
    fib_ref[i] = fib_ref[i-1] + fib_ref[i-2]

# Generate a data set using slices of the reference data set of the same length
window_size = 25
n_all_sequences = n_terms - (window_size+1)
X_all = np.empty([n_all_sequences, window_size, 2], dtype=np.float64)
y_all = np.empty([n_all_sequences, window_size, 1], dtype=np.float64)

for i in xrange(n_all_sequences):
    for j in xrange(window_size):
        X_all[i][j][0] = fib_ref[i+j]
        X_all[i][j][1] = fib_ref[i+j+1]
        y_all[i][j] = fib_ref[i+j+2]

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
model.add(LSTM(150, return_sequences=True, input_shape=(window_size,2)))
model.add(TimeDistributed(Dense(1, activation='relu')))
#model.add(Dropout(dropout))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print model.summary()

model.fit(X_train, y_train, nb_epoch=10, batch_size=25)

scores = model.evaluate(X_test, y_test, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)
