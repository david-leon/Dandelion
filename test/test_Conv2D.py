# coding:utf-8
# Test for Conv2D class
# Created   :   1, 31, 2018
# Revised   :   1, 31, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from lasagne.layers import InputLayer, Conv2DLayer, get_output
import lasagne.nonlinearities as LACT
import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

class build_model_D(Module):
    def __init__(self, in_channel=3, out_channel=3, kernel_size=(3,3), stride=(1,1), pad='valid', dilation=(1,1), num_groups=1):
        super().__init__()
        self.conv2d = Conv2D(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                             pad=pad, dilation=dilation, num_groups=num_groups)
        self.predict = self.forward

    def forward(self, x):
        """

        :param x: (B, C, H, W)
        :return:
        """
        x = self.conv2d.forward(x)
        x = relu(x)
        return x

def build_model_L(in_channel=3, out_channel=3, kernel_size=(3,3), stride=(1,1), pad='valid', dilation=(1,1), num_groups=1):
    input_var = tensor.ftensor4('x')  # (B, C, H, W)
    input0 = InputLayer(shape=(None, in_channel, None, None), input_var=input_var, name='input0')
    conv0  = Conv2DLayer(input0, num_filters=out_channel, filter_size=kernel_size, stride=stride, pad=pad, nonlinearity=LACT.rectify,
                        name='conv0')
    return conv0

def test_case_0():
    import numpy as np
    from lasagne_ext.utils import get_layer_by_name

    in_channel = 1; out_channel = 3;kernel_size = (3, 3); stride = (1, 1); pad = 'valid';dilation = (1,1);num_groups = 1
    model_D = build_model_D(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride,
                             pad=pad, dilation=dilation, num_groups=num_groups)
    model_L = build_model_L(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride,
                             pad=pad)

    W = np.random.rand(out_channel, in_channel, kernel_size[0], kernel_size[1]).astype(np.float32)
    b = np.random.rand(out_channel).astype(np.float32)

    model_D.conv2d.W.set_value(W)
    model_D.conv2d.b.set_value(b)

    conv_L = get_layer_by_name(model_L, 'conv0')
    conv_L.W.set_value(W)
    conv_L.b.set_value(b)

    X = get_layer_by_name(model_L, 'input0').input_var
    y_D = model_D.forward(X)
    y_L = get_output(model_L)

    fn_D = theano.function([X], y_D, no_default_updates=True, on_unused_input='ignore')
    fn_L = theano.function([X], y_L, no_default_updates=True, on_unused_input='ignore')

    for i in range(20):
        x = np.random.rand(3, in_channel, 32, 32).astype(np.float32) - 0.5
        y_D = fn_D(x)
        y_L = fn_L(x)
        diff = np.max(np.abs(y_D - y_L))
        print('i=%d, diff=%0.6f' % (i, diff))
        if diff>1e-4:
            print('y_D=\n', y_D)
            print('y_L=\n', y_L)
            raise ValueError('diff is too big')


if __name__ == '__main__':

    test_case_0()

    print('Test passed')



