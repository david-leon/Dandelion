# coding:utf-8
# Unit test for Dense class
# Created   :   1, 30, 2018
# Revised   :   1, 30, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from lasagne.layers import InputLayer, DenseLayer, get_output
import lasagne.nonlinearities as LACT

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

class build_model_D(Module):
    def __init__(self, in_dim=3, out_dim=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dense = Dense(input_dims=self.in_dim, output_dim=self.out_dim)
        self.predict = self.forward

    def forward(self, x):
        x = self.dense.forward(x)
        x = relu(x)
        return x

def build_model_L(in_dim=3, out_dim=3):
    input_var = tensor.fmatrix('x')
    input0 = InputLayer(shape=(None, in_dim), input_var=input_var, name='input0')
    dense0 = DenseLayer(input0, num_units=out_dim, nonlinearity=LACT.rectify, name='dense0')
    return dense0

def test_case_0(in_dim=1, out_dim=1):
    import numpy as np
    from lasagne_ext.utils import get_layer_by_name


    model_D = build_model_D(in_dim=in_dim, out_dim=out_dim)
    model_L = build_model_L(in_dim=in_dim, out_dim=out_dim)

    W = np.random.rand(in_dim, out_dim).astype(np.float32)
    b = np.random.rand(out_dim).astype(np.float32)
    model_D.dense.W.set_value(W)
    model_D.dense.b.set_value(b)
    get_layer_by_name(model_L, 'dense0').W.set_value(W)
    get_layer_by_name(model_L, 'dense0').b.set_value(b)

    X = get_layer_by_name(model_L, 'input0').input_var
    y_D = model_D.forward(X)
    y_L = get_output(model_L)

    fn_D = theano.function([X], y_D, no_default_updates=True)
    fn_L = theano.function([X], y_L, no_default_updates=True)

    for i in range(20):
        x = np.random.rand(16, in_dim).astype(np.float32)
        y_D = fn_D(x)
        y_L = fn_L(x)
        diff = np.sum(np.abs(y_D - y_L))
        print('i=%d, diff=%0.6f' % (i, diff))
        if diff>1e-4:
            raise ValueError('diff is too big')

if __name__ == '__main__':

    test_case_0(3, 2)

    print('Test passed')



