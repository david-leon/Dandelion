# coding:utf-8
# Unit test for pooling functions
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
from dandelion.functional import *

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

def pool_1d_Lasagne(x, axis=1, mode='max'):
    """
    Lasagne requires x is 3D, and pooling is done on the last dimension
    :param x:
    :param axis:
    :return:
    """
    input_4d = tensor.shape_padright(x, 1)
    if axis == 1:
        input_4d = input_4d.dimshuffle((0, 2, 1, 3))
    pooled = pool_2d(input_4d,
                     ws=(2, 1),
                     stride=(2, 1),
                     ignore_border=True,
                     pad=(0, 0),
                     mode=mode,
                     )
    if axis == 1:  # [DV] add support for 'axis' para
        pooled = pooled.dimshuffle((0, 2, 1, 3))
    return pooled[:, :, :, 0]

def test_case_0():
    import numpy as np

    x_3d = tensor.ftensor3('x')
    y_3d_D = pool_1d(x_3d, axis=1)

    y_3d_L = pool_1d_Lasagne(x_3d, axis=1)

    fn_D = theano.function([x_3d], y_3d_D, no_default_updates=True, on_unused_input='ignore')
    fn_L = theano.function([x_3d], y_3d_L, no_default_updates=True, on_unused_input='ignore')


    for i in range(20):
        x = np.random.rand(7, 117, 27).astype(np.float32)
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



