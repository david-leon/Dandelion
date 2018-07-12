# coding:utf-8
# Test for feature pyramid net, partial.
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
from dandelion.model.feature_pyramid_net import model_FPN

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

def test_case_0():
    model = model_FPN(input_channel=1, batchnorm_mode=2, base_n_filters=32)
    x = tensor.ftensor4('x')
    y = model.forward(x)
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(2, 1, 64, 63).astype(np.float32)
    output = fn(input)
    for r in output:
        print(r.shape)
    # print(output.shape)
    print(output)

if __name__ == '__main__':

    test_case_0()

    print('Test passed')



