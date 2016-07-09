#!/usr/bin/python
import numpy as np

################################################################################
# GENERATE THE DATA
################################################################################

np.random.seed()

# Generate a big fibonacci sequence to 
N_TERMS = 500
fib_ref = np.empty(N_TERMS, dtype=np.float64)
fib_ref[0] = 1.0
fib_ref[1] = 1.0

for i in xrange(2, N_TERMS):
    fib_ref[i] = fib_ref[i-1] + fib_ref[i-2]

# Generate a data set using slices of the reference data set of the same length
WINDOW_SIZE = 25
n_all_seqs = N_TERMS - WINDOW_SIZE
all_seqs = np.empty([n_all_seqs, 2, WINDOW_SIZE], dtype=np.float64)

# X data = all_seqs[sequence][0][t]
# Y data = all_seqs[sequence][1][t+1]

for i in xrange(0, n_all_seqs):
    for j in xrange(0, WINDOW_SIZE):
        all_seqs[i][0][j] = fib_ref[i+j]
        all_seqs[i][1][j] = fib_ref[i+j+1]

# Create training and test sets by randomly sampling the whole data set
TEST_SPLIT = 0.2
n_test_seqs = int(n_all_seqs * TEST_SPLIT)
n_train_seqs = n_all_seqs - n_test_seqs
test_seqs = np.empty([n_test_seqs, 2, WINDOW_SIZE], dtype=np.float32)
train_seqs = np.empty([n_train_seqs, 2, WINDOW_SIZE], dtype=np.float64)

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

################################################################################
# DEFINE THE MODEL
################################################################################

def sigmoid (x):
    return 1.0 / (1.0 + np.exp(-x))

def dtanh_dx (x):
    return 1.0 - np.power(np.tanh(x), 2)

class LSTMcell:

    def __init__ (self):
        # States
        self.h_t = 2*np.random.random() # hypothesis output
        self.c_t = 2*np.random.random() # internal memory

        # Parameters
        self.W_a = 2*np.random.random([1,2])
        self.W_i = 2*np.random.random([1,2])
        self.W_f = 2*np.random.random([1,2])
        self.W_o = 2*np.random.random([1,2])
        self.U_a = 2*np.random.random()
        self.U_i = 2*np.random.random()
        self.U_f = 2*np.random.random()
        self.U_o = 2*np.random.random()

    def forward_step (self, x_t):
        a_t = np.tanh( np.dot(self.W_c, x_t) + np.dot(self.U_c, self.h_t) )
        i_t = sigmoid( np.dot(self.W_i, x_t) + np.dot(self.U_i, self.h_t) )
        f_t = sigmoid( np.dot(self.W_f, x_t) + np.dot(self.U_f, self.h_t) )
        o_t = sigmoid( np.dot(self.W_o, x_t) + np.dot(self.U_o, self.h_t) )

        self.c_t = (i_t * a_t) + (f_t * self.c_t)
        self.h_t = o_t * np.tanh(self.c_t)

    def backward_step (self):
        pass

    def train (self, dataX, dataY):
        pass

################################################################################
# TRAIN THE MODEL
################################################################################

################################################################################
# TEST THE MODEL
################################################################################
