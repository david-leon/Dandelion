# coding:utf-8
# Dandelion functional pool
# Created   :   2, 27, 2018
# Revised   :   2, 27, 2018
# All rights reserved
# ------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import theano
floatX = theano.config.floatX
from theano import tensor
from theano.tensor.signal import pool

def pool_1d(x, ws=2, ignore_border=True, stride=None, pad=0, mode='max', axis=-1):
    """
    Pooling 1 dimension along the given axis, support for any dimensional input
    :param x: input to be pooled, can have any dimension
    :param ws:
    :param ignore_border:
    :param stride: if None, same with ws
    :param pad:
    :param mode:
    :param axis: default the last dimension
    :return:
    """
    if stride is not None:
        stride = (stride, 1)
    ndim = x.ndim
    if axis < 0:
        axis += ndim
    pattern = []
    pattern_reverse = list(range(ndim))
    for i in range(ndim):
        if i != axis:
            pattern.append(i)
            if i != len(pattern)-1:
                pattern_reverse[i] = len(pattern)-1
    pattern.extend([axis, 'x'])
    pattern_reverse[axis] = ndim-1
    x = x.dimshuffle(pattern)
    x = pool.pool_2d(x, ws=(ws, 1), ignore_border=ignore_border, stride=stride, pad=(pad,0), mode=mode)[..., 0]
    x = x.dimshuffle(pattern_reverse)
    return x

def pool_2d(x, ws=(2,2), ignore_border=True, stride=None, pad=(0, 0), mode='max'):
    return pool.pool_2d(x, ws=ws, ignore_border=ignore_border, stride=stride, pad=pad, mode=mode)

def pool_3d(x, ws=(2,2,2), ignore_border=True, stride=None, pad=(0, 0, 0), mode='max'):
    return pool.pool_3d(x, ws=ws, ignore_border=ignore_border, stride=stride, pad=pad, mode=mode)



if __name__ == '__main__':
    INFO = ['Dandelion framework: module pool\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)




