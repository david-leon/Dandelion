# coding:utf-8
# Test for categorical_crossentropy_log
# Created   :   7,  3, 2018
# Revised   :   7,  3, 2018
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

def test_case_0():
    x = tensor.fmatrix()  # (B, N)
    # y = tensor.ivector()  # (B,)
    y = tensor.fmatrix()  # (B, N)

    x1 = softmax(x)  # (B, N)
    r1 = categorical_crossentropy(x1, y, eps=0.0)

    x2 = log_softmax(x)
    r2 = categorical_crossentropy_log(x2, y)

    f1 = theano.function([x, y], r1)
    f2 = theano.function([x, y], r2)

    for i in range(100):
        B, N = np.random.randint(low=1, high=32), np.random.randint(low=2, high=100)
        X = np.random.rand(B, N)
        Y = np.random.randint(low=0, high=N, size=(B,))
        Y = np.eye(N)[Y]

        X = X.astype('float32')
        Y = Y.astype('float32')

        r1 = f1(X, Y)
        r2 = f2(X, Y)
        dif = np.sum(np.abs(r1 - r2))
        if dif > 1e-7:
            print(r1)
            print(r2)
            raise ValueError('r1 != r2')

def test_case_1():
    N = np.random.randint(low=2, high=100)
    x = tensor.fmatrix()  # (B, N)
    y = tensor.ivector()  # (B,)

    x1 = softmax(x)  # (B, N)
    r1 = categorical_crossentropy(x1, y, eps=0.0)

    x2 = log_softmax(x)
    r2 = categorical_crossentropy_log(x2, y, m=N)

    f1 = theano.function([x, y], r1)
    f2 = theano.function([x, y], r2)

    for i in range(100):
        # B, N = np.random.randint(low=1, high=32), np.random.randint(low=2, high=100)
        B = np.random.randint(low=1, high=32)
        X = np.random.rand(B, N)
        Y = np.random.randint(low=0, high=N, size=(B,))

        X = X.astype('float32')
        Y = Y.astype('int32')

        r1 = f1(X, Y)
        # print('r1=', r1)
        r2 = f2(X, Y)
        # print('r2=', r2)
        dif = np.max(np.abs(r1 - r2))
        if dif > 1e-6:
            print('r1=', r1)
            print('r2=', r2)
            raise ValueError('r1 != r2')

if __name__ == '__main__':

    test_case_0()

    test_case_1()

    print('Test passed')



