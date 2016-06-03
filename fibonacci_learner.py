#!/usr/bin/python
import numpy as np
import tensorflow as tf 

# Generate a big fibonnaci sequence to 
N_TERMS = 100
fib_ref = np.empty(N_TERMS, dtype=np.int64)
fib_ref[0] = 0
fib_ref[1] = 1

for i in xrange(2, N_TERMS):
    fib_ref[i] = fib_ref[i-1] + fib_ref[i-2]

# Generate a data set using slices of the reference data set of the same length
WINDOW_SIZE = 20
n_all_seqs = N_TERMS - WINDOW_SIZE + 1
all_seqs = np.empty([n_all_seqs, WINDOW_SIZE], dtype=np.int64)

for i in xrange(0, n_all_seqs):
    for j in xrange(0, WINDOW_SIZE):
        all_seqs[i][j] = fib_ref[i+j]

# Create training and test sets by randomly sampling the whole data set
TEST_SPLIT = 0.2
n_test_seqs = int(n_all_seqs * TEST_SPLIT)
n_train_seqs = n_all_seqs - n_test_seqs
test_seqs = np.empty([n_test_seqs, WINDOW_SIZE], dtype=np.int64)
train_seqs = np.empty([n_train_seqs, WINDOW_SIZE], dtype=np.int64)

test_indices = np.random.choice(xrange(0, n_all_seqs), n_test_seqs, replace=False)
train_i = 0
test_i = 0

for i in xrange(0, n_all_seqs):
    if i in test_indices:
        np.copyto(test_seqs[test_i], all_seqs[i])
        test_i += 1
    else:
        np.copyto(train_seqs[train_i], all_seqs[i])
        train_i += 1
