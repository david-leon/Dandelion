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
from dandelion.model.shufflenet import *

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

def test_case_0():
    model = ShuffleUnit(in_channels=16, inner_channels=4, border_mode='same', batchnorm_mode=0, activation=relu, group_num=4, fusion_mode='add', dilation=2, stride=2)
    x = tensor.ftensor4('x')
    y = model.forward(x)
    # y = tensor.nnet.conv2d()
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(4, 16, 256, 256).astype(np.float32)
    output = fn(input)
    # print(output)
    print(output.shape)

def test_case_1():
    model = ShuffleUnit_Stack(in_channels=16, out_channels=32)
    x = tensor.ftensor4('x')
    y = model.forward(x)
    # y = tensor.nnet.conv2d()
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(4, 16, 256, 256).astype(np.float32)
    output = fn(input)
    # print(output)
    print(output.shape)


def test_case_2():
    model = model_ShuffleNet(in_channels=1, group_num=4, stage_channels=(24, 272, 544, 1088), stack_size=(3, 7, 3), batchnorm_mode=1, activation=relu)
    # model_weights = model.get_weights()
    # for value, w_name in model_weights:
    #     print('name = %s, shape='%w_name, value.shape)

    x = tensor.ftensor4('x')
    y = model.forward(x)
    # y = tensor.nnet.conv2d()
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(4, 1, 224, 224).astype(np.float32)
    output = fn(input)
    # print(output)
    print(output.shape)

def test_case_3():
    model = model_ShuffleSeg.ShuffleNet(in_channels=1, out_channels=6)
    x = tensor.ftensor4('x')
    y = model.forward(x)
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(4, 1, 224, 224).astype(np.float32)
    output = fn(input)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)

def test_case_4():
    model = model_ShuffleSeg()
    # from dandelion.util import gpickle
    # gpickle.dump((model.get_weights(), None), 'shuffleseg.gpkl')
    x = tensor.ftensor4('x')
    y = model.forward(x)
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False, on_unused_input='ignore')
    print('run fn...')
    input = np.random.rand(4, 1, 256, 256).astype(np.float32)
    output = fn(input)
    print(output.shape)



if __name__ == '__main__':

    # test_case_0()
    test_case_1()
    # test_case_2()
    # test_case_3()
    # test_case_4()

    print('Test passed')



