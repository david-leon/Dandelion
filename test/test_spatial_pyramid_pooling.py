# coding:utf-8
# Test for spatial_pyramid pooling
# Created   :   7,  5, 2018
# Revised   :   7,  5, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys
# os.environ['THEANO_FLAGS'] = "floatX=float64, mode=FAST_RUN"
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.functional import spatial_pyramid_pooling
from lasagne.layers import InputLayer, get_output, SpatialPyramidPoolingLayer
import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

class build_model_D(Module):
    def __init__(self, pyramid_dims=[6, 4, 2, 1]):
        super().__init__()
        self.pyramid_dims = pyramid_dims
        self.predict = self.forward

    def forward(self, x):
        """

        :param x: (B, C, H, W)
        :return:
        """
        x = spatial_pyramid_pooling(x, pyramid_dims=self.pyramid_dims, implementation='fast_ls')
        return x

def build_model_L(pyramid_dims=[6, 4, 2, 1]):
    input_var = tensor.ftensor4('x')  # (B, C, H, W)
    input0 = InputLayer(shape=(None, None, None, None), input_var=input_var, name='input0')
    x = SpatialPyramidPoolingLayer(input0, pool_dims=pyramid_dims, implementation='fast')
    return x

def test_case_0():
    import numpy as np
    from lasagne_ext.utils import get_layer_by_name
    np.random.seed(0)

    pyramid_dims = [3, 2, 1]

    model_D = build_model_D(pyramid_dims=pyramid_dims)
    model_L = build_model_L(pyramid_dims=pyramid_dims)

    X = get_layer_by_name(model_L, 'input0').input_var
    y_D = model_D.forward(X)
    y_L = get_output(model_L)

    fn_D = theano.function([X], y_D, no_default_updates=True, on_unused_input='ignore')
    fn_L = theano.function([X], y_L, no_default_updates=True, on_unused_input='ignore')

    for i in range(20):
        B = np.random.randint(low=1, high=33)
        C = np.random.randint(low=1, high=32)
        H = np.random.randint(low=5, high=512)
        W = np.random.randint(low=5, high=513)
        x = np.random.rand(B, C, H, W).astype(np.float32) - 0.5
        y_D= fn_D(x)
        y_L = fn_L(x)
        # print(y_D)
        diff = np.max(np.abs(y_D - y_L))
        print('i=%d, diff=%0.6f' % (i, diff))
        if diff>1e-4:
            print(y_D.shape)
            print(y_L.shape)
            print('B=%d, C=%d, H=%d, W=%d\n' % (B, C, H, W))
            print('y_D=\n', y_D)
            print('y_L=\n', y_L)
            raise ValueError('diff is too big')


class build_model_D_1(Module):
    def __init__(self, pyramid_dims=[6, 4, 2, 1]):
        super().__init__()
        self.pyramid_dims = pyramid_dims
        self.predict = self.forward

    def forward(self, x):
        """

        :param x: (B, C, H, W)
        :return:
        """
        x = spatial_pyramid_pooling(x, pyramid_dims=self.pyramid_dims, implementation='fast')
        return x

def build_model_L_1(pyramid_dims=[6, 4, 2, 1]):
    input_var = tensor.ftensor4('x')  # (B, C, H, W)
    input0 = InputLayer(shape=(None, None, None, None), input_var=input_var, name='input0')
    x = SpatialPyramidPoolingLayer(input0, pool_dims=pyramid_dims, implementation='fast')
    return x


def test_case_1():
    import numpy as np
    from lasagne_ext.utils import get_layer_by_name
    np.random.seed(0)

    pyramid_dims = [4]

    model_D = build_model_D_1(pyramid_dims=pyramid_dims)
    model_L = build_model_L_1(pyramid_dims=pyramid_dims)

    X = get_layer_by_name(model_L, 'input0').input_var
    y_D = model_D.forward(X)
    # y_L = get_output(model_L)

    fn_D = theano.function([X], y_D, no_default_updates=True, on_unused_input='ignore')
    # fn_L = theano.function([X], y_L, no_default_updates=True, on_unused_input='ignore')

    x = np.array([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]])
    x = x.astype(np.float32)
    # y = fn_L(x)  #  Lasagne will crash
    y = fn_D(x)
    y_expect = np.array([[[[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.6, 0.0], [0.7, 0.8, 0.9, 0.0], [0.0, 0.0, 0.0, 0.0]]]])
    y_expect = y_expect.astype(np.float32)
    y_expect = y_expect.reshape((1, 1, -1))
    diff = np.max(np.abs(y - y_expect))
    print('diff=%0.6f' % diff)
    if diff > 1e-4:
        raise ValueError('diff is too big')


class build_model_D_2(Module):
    def __init__(self, pyramid_dims=[6, 4, 2, 1]):
        super().__init__()
        self.pyramid_dims = pyramid_dims
        self.predict = self.forward

    def forward(self, x):
        """

        :param x: (B, C, H, W)
        :return:
        """
        x = spatial_pyramid_pooling(x, pyramid_dims=self.pyramid_dims, implementation='stretch')
        return x

def build_model_L_2(pyramid_dims=[6, 4, 2, 1]):
    input_var = tensor.ftensor4('x')  # (B, C, H, W)
    input0 = InputLayer(shape=(None, None, None, None), input_var=input_var, name='input0')
    x = SpatialPyramidPoolingLayer(input0, pool_dims=pyramid_dims, implementation='kaiming')
    return x


def test_case_2():
    import numpy as np
    from lasagne_ext.utils import get_layer_by_name
    np.random.seed(0)

    pyramid_dims = [4]

    model_D = build_model_D_2(pyramid_dims=pyramid_dims)
    model_L = build_model_L_2(pyramid_dims=pyramid_dims)

    X = get_layer_by_name(model_L, 'input0').input_var
    y_D = model_D.forward(X)
    # y_L = get_output(model_L)

    fn_D = theano.function([X], y_D, no_default_updates=True, on_unused_input='ignore')
    # fn_L = theano.function([X], y_L, no_default_updates=True, on_unused_input='ignore')

    x = np.array([[[[0.1, 0.2], [0.3, 0.4]]]])
    x = x.astype(np.float32)
    # y = fn_L(x)  #  Lasagne will use float64
    y = fn_D(x)
    y_expect = np.array([[[[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4, 0.4]]]])
    y_expect = y_expect.astype(np.float32)
    y_expect = y_expect.reshape((1, 1, -1))
    diff = np.max(np.abs(y - y_expect))
    print('diff=%0.6f' % diff)
    if diff > 1e-4:
        raise ValueError('diff is too big')

    x = np.array([[[[0.7, 0.2, 0.3], [0.5, 0.6, 0.4], [0.1, 0.9, 0.8]]]])
    x = x.astype(np.float32)
    y = fn_D(x)
    y_expect = np.array([[[[0.7, 0.7, 0.3, 0.3], [0.7, 0.7, 0.6, 0.4], [0.5, 0.9, 0.9, 0.8], [0.1, 0.9, 0.9, 0.8]]]])
    y_expect = y_expect.astype(np.float32)
    y_expect = y_expect.reshape((1, 1, -1))
    diff = np.max(np.abs(y - y_expect))
    print('diff=%0.6f' % diff)
    if diff > 1e-4:
        raise ValueError('diff is too big')


if __name__ == '__main__':

    test_case_0()

    test_case_1()

    test_case_2()

    print('Test passed')



