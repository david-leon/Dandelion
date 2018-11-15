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

import theano, sys
sys.setrecursionlimit(40000)
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from dandelion.model.shufflenet import *
from dandelion.objective import *
from dandelion.update import *

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

def test_case_0():
    print('test_case_0: ShuffleUnit')
    model = ShuffleUnit(in_channels=16, inner_channels=4, border_mode='same', batchnorm_mode=0, activation=relu, group_num=4, fusion_mode='add', dilation=2, stride=2)
    x = tensor.ftensor4('x')
    y = model.forward(x)
    # y = tensor.nnet.conv2d()
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(4, 16, 256, 256).astype(np.float32)
    output = fn(input)
    assert output.shape == (4, 16, 128, 128), 'incorrect output shape = %s' % str(output.shape)

def test_case_1():
    print('test_case_1: ShuffleUnit_Stack')
    model = ShuffleUnit_Stack(in_channels=16, out_channels=32)
    x = tensor.ftensor4('x')
    y = model.forward(x)
    # y = tensor.nnet.conv2d()
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(4, 16, 256, 256).astype(np.float32)
    output = fn(input)
    assert output.shape == (4, 32, 128, 128), 'incorrect output shape = %s' % str(output.shape)

def test_case_2():
    print('test_case_2: model_ShuffleNet')
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
    assert output.shape == (4, 1088, 7, 7), 'incorrect output shape = %s' % str(output.shape)

def test_case_3():
    print('test_case_3: model_ShuffleSeg.ShuffleNet')
    model = model_ShuffleSeg.ShuffleNet(in_channels=1, out_channels=6)
    x = tensor.ftensor4('x')
    y = model.forward(x)
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(4, 1, 224, 224).astype(np.float32)
    output = fn(input)
    assert output[0].shape == (4, 6, 7, 7), 'incorrect output[0] shape = %s' % str(output[0].shape)
    assert output[1].shape == (4, 544, 14, 14), 'incorrect output[1] shape = %s' % str(output[1].shape)
    assert output[2].shape == (4, 272, 28, 28), 'incorrect output[2] shape = %s' % str(output[2].shape)

def test_case_4():
    print('test_case_4: model_ShuffleSeg')
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
    assert output.shape == (4, 6, 256, 256), 'incorrect output shape = %s' % str(output.shape)

def test_case_5():
    print('test_case_5: ShuffleUnit_v2')
    args = [ [256, 128, 'same', 1, relu, 2, 2],
             [256, 256, 'same', 2, relu, 1, 1], ]
    shape_gt = [ (4, 128, 128, 128),
                 (4, 256, 256, 256)]
    for i, arg in enumerate(args):
        in_channels, out_channels, border_mode, batchnorm_mode, activation, stride, dilation = arg
        model = ShuffleUnit_v2(in_channels=in_channels, out_channels=out_channels, border_mode=border_mode, batchnorm_mode=batchnorm_mode,
                               activation=activation, dilation=dilation, stride=stride)
        x = tensor.ftensor4('x')
        y = model.forward(x)
        print('compiling fn...')
        fn = theano.function([x], y, no_default_updates=False)
        print('run fn...')
        input = np.random.rand(4, in_channels, 256, 256).astype(np.float32)
        output = fn(input)
        assert output.shape == shape_gt[i], 'incorrect output shape = %s' % str(output.shape)

def test_case_6():
    print('test_case_6: ShuffleUnit_v2_Stack')
    args = [ [16, 32, 1, relu, 3],
             [16, 16, 0, relu, 4],
             [32, 16, 2, relu, 2]]
    shape_gt = [ (4, 32, 128, 128),
                 (4, 16, 128, 128),
                 (4, 16, 128, 128)]
    for i, arg in enumerate(args):
        in_channels, out_channels, batchnorm_mode, activation, stack_size = arg
        model = ShuffleUnit_v2_Stack(in_channels=in_channels, out_channels=out_channels, batchnorm_mode=batchnorm_mode, activation=activation, stack_size=stack_size)
        x = tensor.ftensor4('x')
        y = model.forward(x)
        print('compiling fn...')
        fn = theano.function([x], y, no_default_updates=False)
        print('run fn...')
        input = np.random.rand(4, in_channels, 256, 256).astype(np.float32)
        output = fn(input)
        # print(output.shape)
        assert output.shape == shape_gt[i], 'incorrect output shape = %s' % str(output.shape)


def test_case_7():
    print('test_case_7: model_ShuffleNet_v2')
    args = [ [3, (24, 116, 232, 464, 1024), (3, 7, 3), 1, relu],
             [1, (24, 48, 96, 192, 1088), (2, 5, 2), 0, relu],
           ]
    shape_gt = [ (4, 1024, 7, 7),
                 (4, 1088, 7, 7),
                ]
    for i, arg in enumerate(args):
        in_channels, stage_channels, stack_size, batchnorm_mode, activation = arg

        model = model_ShuffleNet_v2(in_channels=in_channels, stage_channels=stage_channels, stack_size=stack_size, batchnorm_mode=batchnorm_mode, activation=activation)
        x = tensor.ftensor4('x')
        y = model.forward(x)
        # y = tensor.nnet.conv2d()
        print('compiling fn...')
        fn = theano.function([x], y, no_default_updates=False)
        print('run fn...')
        input = np.random.rand(4, in_channels, 224, 224).astype(np.float32)
        output = fn(input)
        # print(output.shape)
        assert output.shape == shape_gt[i], 'incorrect output shape = %s' % str(output.shape)

def test_case_8():
    print('test_case_8: grad of model_ShuffleNet_v2')
    args = [ [3, (24, 116, 232, 464, 1024), (3, 7, 3), 1, relu],
             [1, (24, 48, 96, 192, 1088), (2, 5, 2), 2, relu],
           ]
    shape_gt = [ (4, 1024, 7, 7),
                 (4, 1088, 7, 7),
                ]
    for i, arg in enumerate(args):
        in_channels, stage_channels, stack_size, batchnorm_mode, activation = arg

        model = model_ShuffleNet_v2(in_channels=in_channels, stage_channels=stage_channels, stack_size=stack_size, batchnorm_mode=batchnorm_mode, activation=activation)
        x  = tensor.ftensor4('x')
        gt = tensor.ftensor4('gt')
        y  = model.forward(x)
        loss = aggregate(squared_error(gt, y))
        params = model.collect_params()
        updates = sgd(loss, params, 1e-4)
        updates.update(model.collect_self_updates())
        print('compiling fn...')
        fn = theano.function([x, gt], [y, loss], updates=updates, no_default_updates=False)
        print('run fn...')
        input = np.random.rand(4, in_channels, 224, 224).astype(np.float32)
        gt    = np.random.rand(4, stage_channels[4], 7, 7).astype(np.float32)
        y, loss = fn(input, gt)
        # print(output.shape)
        print('loss = ', loss)
        assert y.shape == shape_gt[i], 'incorrect output shape = %s' % str(y.shape)



if __name__ == '__main__':

    # test_case_0()
    # test_case_1()
    # test_case_2()
    # test_case_3()
    # test_case_4()
    # test_case_5()
    # test_case_6()
    # test_case_7()
    test_case_8()

    print('Test passed')



