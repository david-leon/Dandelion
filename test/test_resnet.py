# coding:utf-8
# Test for ResNet, partial.
# Created   :   7,  6, 2018
# Revised   :   7,  6, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from dandelion.model.resnet import ResNet_bottleneck

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

def test_case_0():
    model = ResNet_bottleneck(outer_channel=16, inner_channel=4, border_mode='same', batchnorm_mode=0, activation=relu)
    x = tensor.ftensor4('x')
    y = model.forward(x)
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(4, 16, 32, 33).astype(np.float32)
    output = fn(input)
    print(output)
    print(output.shape)

if __name__ == '__main__':

    test_case_0()

    print('Test passed')



