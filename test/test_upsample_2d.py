# coding:utf-8
# Test for upsample_2d
# Created   :   7,  5, 2018
# Revised   :   7,  5, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.functional import upsample_2d, upsample_2d_bilinear
from lasagne.layers import InputLayer, get_output, Upscale2DLayer
import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

class build_model_D(Module):
    def __init__(self, ratio=[2, 3], mode='repeat'):
        super().__init__()
        self.ratio = ratio
        self.mode = mode
        self.predict = self.forward

    def forward(self, x):
        """

        :param x: (B, C, H, W)
        :return:
        """
        x = upsample_2d(x, ratio=self.ratio, mode=self.mode)
        # x = relu(x)
        return x

def build_model_L(ratio=[2,3], mode='repeat'):
    input_var = tensor.ftensor4('x')  # (B, C, H, W)
    input0 = InputLayer(shape=(None, None, None, None), input_var=input_var, name='input0')
    x  = Upscale2DLayer(input0, scale_factor=ratio, mode=mode)
    return x

def test_case_0():
    import numpy as np
    from lasagne_ext.utils import get_layer_by_name

    ratio = [1, 2]
    mode = 'dilate'

    model_D = build_model_D(ratio=ratio, mode=mode)
    model_L = build_model_L(ratio=ratio, mode=mode)


    X = get_layer_by_name(model_L, 'input0').input_var
    y_D = model_D.forward(X)
    y_L = get_output(model_L)

    fn_D = theano.function([X], y_D, no_default_updates=True, on_unused_input='ignore')
    fn_L = theano.function([X], y_L, no_default_updates=True, on_unused_input='ignore')

    for i in range(20):
        B = np.random.randint(low=1, high=16)
        C = np.random.randint(low=1, high=32)
        H = np.random.randint(low=5, high=256)
        W = np.random.randint(low=5, high=255)
        x = np.random.rand(B, C, H, W).astype(np.float32) - 0.5
        y_D = fn_D(x)
        y_L = fn_L(x)
        # print(y_D)
        diff = np.max(np.abs(y_D - y_L))
        print('i=%d, diff=%0.6f' % (i, diff))
        if diff>1e-4:
            print('y_D=\n', y_D)
            print('y_L=\n', y_L)
            raise ValueError('diff is too big')

if __name__ == '__main__':

    test_case_0()

    print('Test passed')



