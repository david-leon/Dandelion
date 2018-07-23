# coding:utf-8
# Test for softmax/log_softmax for ndim > 2.
# This test is added because occasionally Theano compiled function would raise shape mismatch error for reshap op. inside softmax/log_softmax when ndim >2.
# Created   :   7, 23, 2018
# Revised   :   7, 23, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, numpy as np
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import softmax, log_softmax
from dandelion.objective import categorical_crossentropy, categorical_crossentropy_log
from dandelion.util import one_hot

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

# ndim = 2
def test_case_0():
    x = tensor.fmatrix()  # (B, N)
    y = softmax(x)  # (B, N)
    f = theano.function([x] , y)

    for i in range(10):
        B = 1
        N = 5
        x = np.random.rand(B, N).astype(np.float32)
        y = f(x)
        print(y.shape)

# ndim = 3
def test_case_1():
    x = tensor.ftensor3()  # (B, T, N)
    y = softmax(x)  # (B, T, N)
    f = theano.function([x] , y)

    for i in range(20):
        B = np.random.randint(1, 10)
        T = np.random.randint(1, 10)
        N = np.random.randint(2, 8)
        x = np.random.rand(B, T, N).astype(np.float32)
        y = f(x)
        print(y.shape)

# ndim = 4
def test_case_2():
    x = tensor.ftensor4()  # (B, H, W, N)
    y = softmax(x)  # (B, H, W, N)
    f = theano.function([x], y)

    for i in range(20):
        B = np.random.randint(1, 10)
        H = np.random.randint(1, 100)
        W = np.random.randint(1, 100)
        N = np.random.randint(2, 8)
        x = np.random.rand(B, H, W, N).astype(np.float32)
        y = f(x)
        print(y.shape)

if __name__ == '__main__':

    test_case_0()

    test_case_1()

    test_case_2()

    print('Test passed')



