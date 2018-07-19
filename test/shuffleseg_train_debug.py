# coding:utf-8
# GPU DeBUG for ShuffleSeg: model_ShuffleSeg can be trained on CPU, but when trained on GPU, error raised as "images and kernel must have the same stack size"
# BUG fixed with PR: https://github.com/Theano/Theano/pull/6624
# Created   :   7, 18, 2018
# Revised   :   7, 18, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import os, sys
# os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise', exception_verbosity=high"
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"
sys.setrecursionlimit(40000)

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from dandelion.model.shufflenet import *
from dandelion.objective import *
from dandelion.update import adadelta

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

if __name__ == '__main__':
    import argparse
    import pygpu

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-device', default='cuda5', type=str)
    argparser.add_argument('-gn', default=1, type=int)
    arg = argparser.parse_args()

    # --- public paras ---#
    device    = arg.device
    group_num = arg.gn


    #--- (1) setup device ---#
    if sys.platform.startswith('linux'):
        if device.startswith('cuda'):
            print('Using NEW backend for training')
            import theano.gpuarray
            theano.gpuarray.use(device, force=True)
        elif device.startswith('gpu'):
            print('Using OLD backend for training')
            import theano.sandbox.cuda
            theano.sandbox.cuda.use(device, force=True)
        else:
            print('Using CPU for training')



    Nclass = 6
    in_channels = 1
    out_channels = 128
    model = model_ShuffleSeg(in_channels=in_channels, Nclass=Nclass, SF_group_num=group_num)
    # model = ShuffleUnit(in_channels=in_channels, group_num=4, stride=2)
    # model = ShuffleUnit_Stack(in_channels=in_channels, out_channels=out_channels, group_num=group_num, stack_size=4)
    print(model.__class__.__name__)
    x = tensor.ftensor4('x')
    y = model.forward(x)
    y_gt = tensor.ftensor4('y')
    loss = aggregate(squared_error(y, y_gt), mode='mean')
    params = model.collect_params()
    updates = adadelta(loss, params)
    updates.update(model.collect_self_updates())
    print('compiling train fn')
    train_fn = theano.function([x, y_gt], loss, updates=updates, no_default_updates=False)

    print('training...')
    for i in range(10):
        print('batch %d' % i)
        x = np.random.rand(4, in_channels, 256, 256).astype(np.float32)
        # y = np.random.rand(4, out_channels, 128, 128).astype(np.float32)
        y = np.random.rand(4, Nclass, 256, 256).astype(np.float32)
        loss = train_fn(x, y)
        print('loss = ', loss)

