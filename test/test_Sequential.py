# coding:utf-8
# Test for Sequential container
# Created   :   7, 12, 2018
# Revised   :   7, 12, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

def test_case_0():
    conv1 = Conv2D(in_channels=3, out_channels=3, stride=(2, 2))
    bn1   = BatchNorm(input_shape=(None, 3, None, None))
    conv2 = Conv2D(in_channels=3, out_channels=5)
    conv3 = Conv2D(in_channels=5, out_channels=8)
    model = Sequential([conv1, bn1, conv2, conv3], activation=relu, name='seq')
    model_weights = model.get_weights()
    for value, w_name in model_weights:
        print('name = %s, shape=' % w_name, value.shape)

    x = tensor.ftensor4('x')
    y = model.forward(x)
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(4, 3, 32, 33).astype(np.float32)
    output = fn(input)
    print(output)
    print(output.shape)

if __name__ == '__main__':

    test_case_0()

    print('Test passed')



