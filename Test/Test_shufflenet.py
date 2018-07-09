# coding:utf-8
# Test for Shuffle-net, partial.
# Created   :   7,  9, 2018
# Revised   :   7,  9, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"
# os.environ['THEANO_FLAGS'] = "floatX=float32, mode=DEBUG_MODE, warn_float64='raise', exception_verbosity=high"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from dandelion.model.shufflenet import ShuffleUnit

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)


if __name__ == '__main__':
    model = ShuffleUnit(outer_channel=16, inner_channel=4, border_mode='same', batchnorm_mode=0, activation=relu, group_num=4)
    x = tensor.ftensor4('x')
    y = model.forward(x)
    # y = tensor.nnet.conv2d()
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(4, 16, 33, 32).astype(np.float32)
    output = fn(input)
    print(output)
    print(output.shape)

    print('Test passed')



