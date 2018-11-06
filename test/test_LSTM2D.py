# coding:utf-8
'''
Unitest for LSTM2D class
Created   :  11,  6, 2018
Revised   :  11,  6, 2018
All rights reserved
'''
# ------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import os
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
import theano.tensor as tensor
from dandelion.module import LSTM2D
from dandelion.objective import *
from dandelion.update import *
import numpy as np, time

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

def test_case_0():
    # input_dim, hidden_dim, B, H, W = 5, 4, 2, 5, 7
    input_dim, hidden_dim, B, H, W = 8, 8, 2, 32, 32
    input = tensor.ftensor4()
    gt    = tensor.ftensor4()
    model = LSTM2D(input_dims=[input_dim],hidden_dim=hidden_dim, peephole=True)
    output = model.forward(input, backward=False)
    loss = aggregate(squared_error(output, gt))
    params = model.collect_params()
    updates = sgd(loss, params, 1e-4)
    updates.update(model.collect_self_updates())
    print('compiling function ...')
    f = theano.function([input, gt], [output, loss], updates=updates, no_default_updates=False)

    print('run function ...')
    X = np.random.rand(H, W, B, input_dim).astype('float32')
    GT = np.random.rand(H, W, B, hidden_dim).astype('float32')
    time0 = time.time()
    Y, loss = f(X, GT)
    time_used = time.time() - time0
    print('time_used = ', time_used)
    print('Y=', Y)
    print('Y.shape=', Y.shape)
    print('loss=', loss)

if __name__ == '__main__':

    test_case_0()

    print('Test passed')