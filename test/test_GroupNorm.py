# coding:utf-8
# Unit test for GroupNorm module
# Created   :  11, 22, 2018
# Revised   :  11, 22, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from dandelion.objective import *
from dandelion.update import *
import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

def fix_update_bcasts(updates):
    for param, update in updates.items():
        if param.broadcastable != update.broadcastable:
            updates[param] = tensor.patternbroadcast(update, param.broadcastable)
    return updates

def test_case_0():
    B, C, H, W = 4, 128, 256, 256
    x = tensor.ftensor4('x')
    z = tensor.ftensor4('gt')
    conv0 = Conv2D(in_channels=2*C, out_channels=C, kernel_size=(3,3), pad='same')
    # gn0   = GroupNorm(channel_num=C, group_num=32, beta=None, gamma=None)
    gn0   = GroupNorm(channel_num=C, group_num=32)
    model = Sequential([conv0, gn0], activation=relu)
    y = model.forward(x)
    loss = aggregate(squared_error(y, z))
    updates = adadelta(loss, model.collect_params())
    updates.update(model.collect_self_updates())
    # f = theano.function([x], y, no_default_updates=False, updates=fix_update_bcasts(bn.collect_self_updates()))
    f = theano.function([x, z], [y, loss], no_default_updates=False, updates=updates)
    x = np.random.rand(B, 2*C, H, W).astype('float32')
    z = np.random.rand(B, C, H, W).astype('float32')
    y, loss = f(x, z)
    assert y.shape ==(B, C, H, W)
    print('test_case_0 passed')

if __name__ == '__main__':

    # test_case_0()

    test_case_0()

    print('Test passed')