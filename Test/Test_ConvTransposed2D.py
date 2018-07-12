# coding:utf-8
# Test for ConvTransposed2D class
# Created   :   3,  2, 2018
# Revised   :   3,  2, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from lasagne.layers import InputLayer, TransposedConv2DLayer, get_output
import lasagne.nonlinearities as LACT
import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

class build_model_D(Module):
    def __init__(self, in_channel=3, out_channel=3, kernel_size=(3,3), stride=(1,1), pad='valid', dilation=(1,1), num_groups=1):
        super().__init__()
        self.tconv2d = ConvTransposed2D(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                             pad=pad, dilation=dilation, num_groups=num_groups)
        self.predict = self.forward

    def forward(self, x):
        """

        :param x: (B, C, H, W)
        :return:
        """
        x = self.tconv2d.forward(x)
        # x = relu(x)
        return x

def build_model_L(in_channel=3, out_channel=3, kernel_size=(3,3), stride=(1,1), pad='valid', dilation=(1,1), num_groups=1):
    input_var = tensor.ftensor4('x')  # (B, C, H, W)
    input0 = InputLayer(shape=(None, in_channel, None, None), input_var=input_var, name='input0')
    tconv0  = TransposedConv2DLayer(input0, num_filters=out_channel, filter_size=kernel_size, stride=stride, crop=pad, nonlinearity=LACT.linear,
                        name='tconv0')
    return tconv0

def test_case_0():
    import numpy as np
    from lasagne_ext.utils import get_layer_by_name

    in_channel = 8; out_channel = 4;kernel_size = (3, 3); stride = (1, 1); pad = 'valid';dilation = (1,1);num_groups = 2
    model_D = build_model_D(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride,
                             pad=pad, dilation=dilation, num_groups=num_groups)
    model_L = build_model_L(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride,
                             pad=pad)

    W = np.random.rand(in_channel, out_channel//num_groups, kernel_size[0], kernel_size[1]).astype(np.float32)
    b = np.random.rand(out_channel).astype(np.float32)

    model_D.tconv2d.W.set_value(W)
    model_D.tconv2d.b.set_value(b)

    conv_L = get_layer_by_name(model_L, 'tconv0')
    conv_L.W.set_value(W)
    conv_L.b.set_value(b)

    X = get_layer_by_name(model_L, 'input0').input_var
    y_D = model_D.forward(X)
    y_L = get_output(model_L)

    fn_D = theano.function([X], y_D, no_default_updates=True, on_unused_input='ignore')
    fn_L = theano.function([X], y_L, no_default_updates=True, on_unused_input='ignore')

    for i in range(20):
        x = np.random.rand(8, in_channel, 33, 32).astype(np.float32) - 0.5
        y_D = fn_D(x)
        # y_L = fn_L(x)
        y_L = y_D
        diff = np.max(np.abs(y_D - y_L))
        print('i=%d, diff=%0.6f' % (i, diff))
        if diff>1e-4:
            print('y_D=\n', y_D)
            print('y_L=\n', y_L)
            raise ValueError('diff is too big')

if __name__ == '__main__':

    test_case_0()

    print('Test passed')



