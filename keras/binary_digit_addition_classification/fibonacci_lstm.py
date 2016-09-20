#!/usr/bin/python
# -*- coding: UTF-8 -*-
################################################################################
# fibonacci_lstm.py

# The model takes 2 decimal numbers and predicts their sum.
# - Samples are modelled as sequences of 2 bits.
# - X vector has 2 bits for frame, one per operand.
# - Y vector has one bit

# Model is trained heavily on first 92 Fibonacci terms to ensure it can generate
# the Fibonacci sequence.

# Gets to 99% validation accuracy after 10 epochs, 100% train and test accuracy.
################################################################################

from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dropout
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import sys

np.random.seed(42)

# Constants
MAX_BITS = 63 # for sys.maxint
N_FIB_TERMS = 92 # up to an including 92nd fib. has <= 19 digits
N_3_WINDOWS = N_FIB_TERMS - 3 + 1
N_EXTRA_DATA_POINTS = 5000
EXTRA_DATA_MAX = np.iinfo(np.int64).max / 2
TEST_SPLIT = 0.2
FIB_REPEAT_FACTOR = 10

# Model & Training Parameters
DROPOUT = 0.33
N_HIDDEN_UNITS = 10
LEARNING_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 100
LOAD_MODEL_FILEPATH = ""
SAVE_MODEL_FILEPATH = ""

# Utility functions
def encode_decimal (n, n_bits=MAX_BITS):
    n = int(n)
    rev_bit_string = "{0:b}".format(n)[::-1]
    x = np.zeros((n_bits,1), dtype=np.int64)
    for i in xrange(len(rev_bit_string)):
        x[i] = int(rev_bit_string[i])
    return x

def make_bits_from_sigmoid_out (arr):
    bit_array = np.zeros(arr.shape, dtype=np.int64)
    for i in xrange(arr.shape[0]):
        if arr[i] >= 0.5:
            bit_array[i] = 1
        else:
            bit_array[i] = 0
    return bit_array

def decode_bits (arr):
    rev_bit_string = ""
    for i in xrange(arr.shape[0]):
        rev_bit_string += str(arr[i][0])
    bit_string = rev_bit_string[::-1]
    return int(bit_string, 2)

# Read the fibonacci series into memory
fibonacci_ref = pd.read_csv("fibonacci_sequence.csv", header=None)

# Generate the dataset
# Duplicate each sample, with t-1 and t-2 terms swapped 
X_fib_all = np.zeros([2*N_3_WINDOWS, MAX_BITS, 2], dtype=np.int64)
Y_fib_all = np.zeros([2*N_3_WINDOWS, MAX_BITS, 1], dtype=np.int64)

for i in xrange(N_3_WINDOWS):

    first = encode_decimal(fibonacci_ref.ix[i][1])
    second = encode_decimal(fibonacci_ref.ix[i+1][1])
    third = encode_decimal(fibonacci_ref.ix[i+2][1])

    X = np.hstack((first, second))
    X_swap = np.hstack((second, first))

    np.copyto(X_fib_all[i], X)
    np.copyto(Y_fib_all[i], third)
    np.copyto(X_fib_all[i+N_3_WINDOWS], X_swap)
    np.copyto(Y_fib_all[i+N_3_WINDOWS], third)

# Generate extra data
X_extra_all = np.zeros([N_EXTRA_DATA_POINTS, MAX_BITS,2], dtype=np.int64)
Y_extra_all = np.zeros([N_EXTRA_DATA_POINTS, MAX_BITS,1], dtype=np.int64)

for i in xrange(N_EXTRA_DATA_POINTS):

    first_n = np.random.randint(0,EXTRA_DATA_MAX)
    second_n = np.random.randint(0,EXTRA_DATA_MAX)
    third_n = first_n + second_n

    first = encode_decimal(first_n)
    second = encode_decimal(second_n)
    third = encode_decimal(third_n)

    X = np.hstack((first, second))
    np.copyto(X_extra_all[i], X)
    np.copyto(Y_extra_all[i], third)

# Create training and test sets by randomly sampling the whole data set
X_train_all = np.vstack(tuple([X_extra_all]+[X_fib_all]*FIB_REPEAT_FACTOR))
Y_train_all = np.vstack(tuple([Y_extra_all]+[Y_fib_all]*FIB_REPEAT_FACTOR))

test_split_mask = np.random.rand(X_train_all.shape[0]) < (1-TEST_SPLIT)

X_train = X_train_all[test_split_mask]
Y_train = Y_train_all[test_split_mask]
X_test = X_train_all[~test_split_mask]
Y_test = Y_train_all[~test_split_mask]

# Define and train the model
model = Sequential()
model.add(LSTM(N_HIDDEN_UNITS, return_sequences=True, input_shape=(MAX_BITS,), input_dim=2))
#model.add(LSTM(150, return_sequences=True, input_shape=(MAX_BITS,2))) # doesn't work with smal <150 LSTM layer
model.add(Dropout(DROPOUT))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

opt = Adam(lr=LEARNING_RATE)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

print model.summary()

model.fit(X_train, Y_train, nb_epoch=10, batch_size=100, verbose=2)

# Evaluate the model
print "~~~ Test Set Scores ~~~"
scores = model.evaluate(X_test, Y_test, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)
print "~~~ All Fibonacci Scores ~~~"
scores = model.evaluate(X_fib_all, Y_fib_all, verbose=0)
print "Accuracy: %.2f%%" % (scores[1]*100)

if SAVE_MODEL_FILEPATH:
    model.save(SAVE_MODEL_FILEPATH)

# Use the model for prediction
def predict_sum (x1, x2):
    x1_encoded = encode_decimal(x1)
    x2_encoded = encode_decimal(x2)
    X = np.hstack((x1_encoded, x2_encoded)).reshape((1,x1_encoded.shape[0],2))
    pred_encoded_raw = model.predict(X, batch_size=1)[0]
    pred_encoded = make_bits_from_sigmoid_out(pred_encoded_raw)
    return decode_bits(pred_encoded)

#print predict_sum(1,1)

# Use the model for generate fibonacci sequence
def generate_fibonacci (n):
    if n > 92:
        print "ERROR: only first 92 fibs supported because of int64 limits! Sorry :("
        return []

    fibs = [1,1]
    for i in xrange(len(fibs), n):
        fibs.append(predict_sum(fibs[i-1], fibs[i-2]))

    return fibs

#print generate_fibonacci(10)

print "~~~ Generating Fibonacci Terms ~~~"
while True:
    print "\nFirst `how many` terms? "
    try:
        raw_s = raw_input().strip()
    except EOFError as e:
        sys.exit(0)

    try:
        n = int(raw_s)
    except ValueError as e:
        print "Please enter an integer!"
        continue

    fibs = generate_fibonacci(n)
    if fibs:
        print ""
        for f in fibs:
            print f
