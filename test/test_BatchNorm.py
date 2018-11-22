# coding:utf-8
# Test for BatchNorm class
# Created   :   2, 27, 2018
# Revised   :   2, 27, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from dandelion.objective import *
from dandelion.update import *
from lasagne.layers import InputLayer, BatchNormLayer_DV, get_output, get_all_updates
import lasagne.nonlinearities as LACT
import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

class build_model_D(Module):
    def __init__(self, input_shape=None, axes='auto'):
        super().__init__()
        self.input_shape = input_shape
        self.axes = axes
        self.bn = BatchNorm(input_shape=self.input_shape, axes=self.axes)

    def forward(self, x):
        x = self.bn.forward(x)
        return x

    def predict(self, x):
        return self.bn.predict(x)

def build_model_L(input_shape=None, axes='auto'):
    input_var = tensor.ftensor4('x')
    input0 = InputLayer(shape=input_shape, input_var=input_var, name='input0')
    result = BatchNormLayer_DV(input0, axes=axes, name='bn0')
    return result

def fix_update_bcasts(updates):
    for param, update in updates.items():
        if param.broadcastable != update.broadcastable:
            updates[param] = tensor.patternbroadcast(update, param.broadcastable)
    return updates

def test_case_0():
    import numpy as np
    from lasagne_ext.utils import get_layer_by_name

    B, C, H, W = 2, 1, 8, 8
    input_shape = (None, C, H, W)
    axes = 'auto'

    model_D = build_model_D(input_shape=input_shape, axes=axes)
    model_L = build_model_L(input_shape=input_shape, axes=axes)

    X = get_layer_by_name(model_L, 'input0').input_var
    #--- predict ---#
    if 0:
        y_D = model_D.predict(X)
        y_L = get_output(model_L, deterministic=True)
        fn_L = theano.function([X], y_L, no_default_updates=True)
        fn_D = theano.function([X], y_D, no_default_updates=True)

    #--- train ---#
    if 1:
        y_D = model_D.forward(X)
        y_L = get_output(model_L, deterministic=False)

    update_L = fix_update_bcasts(get_all_updates(model_L))
    update_D = fix_update_bcasts(model_D.collect_self_updates())

    fn_L = theano.function([X], y_L, updates=update_L, no_default_updates=True)
    fn_D = theano.function([X], y_D, updates=update_D, no_default_updates=False)
    # fn_L = theano.function([X], y_L, no_default_updates=True)
    # fn_D = theano.function([X], y_D, no_default_updates=True)


    for i in range(20):
        x = np.random.rand(B, C, H, W).astype(np.float32)
        y_D = fn_D(x)
        y_L = fn_L(x)
        diff = np.sum(np.abs(y_D - y_L))
        print('i=%d, diff=%0.6f' % (i, diff))
        if diff>1e-4:
            print(y_D)
            print(y_L)
            raise ValueError('diff is too big')
    print('test_case_0 passed')

def test_case_1():
    B, C, H, W = 4, 1, 256, 256
    x = tensor.ftensor4('x')
    z = tensor.ftensor4('gt')
    bn   = BatchNorm(input_shape=(B, C, H, W), beta=None, gamma=None)
    conv = Conv2D(in_channels=C, out_channels=C, kernel_size=(3,3), pad='same')
    model = Sequential([bn, conv], activation=relu)
    y = model.forward(x)
    loss = aggregate(squared_error(y, z))
    updates = adadelta(loss, model.collect_params())
    updates.update(model.collect_self_updates())
    # f = theano.function([x], y, no_default_updates=False, updates=fix_update_bcasts(bn.collect_self_updates()))
    f = theano.function([x, z], [y, loss], no_default_updates=False, updates=updates)
    x = np.random.rand(B, C, H, W).astype('float32')
    z = np.random.rand(B, C, H, W).astype('float32')
    y, loss = f(x, z)
    assert y.shape ==(B, C, H, W)
    print('test_case_1 passed')

if __name__ == '__main__':

    # test_case_0()

    test_case_1()

    print('Test passed')