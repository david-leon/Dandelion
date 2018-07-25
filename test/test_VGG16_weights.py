# coding:utf-8
# Test VGG16 weights transferred from Lasagne
# Created   :  mm, dd, yyyy
# Revised   :  mm, dd, yyyy
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import os
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"
import theano
import lasagne
from lasagne.layers import InputLayer, get_output
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax

from dandelion.model.vgg import model_VGG16
from dandelion.util import gpickle
import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)


def build_model_L():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

build_model_D = model_VGG16

# '_' prefix means no need for unit test
def _test_case_0():
    import numpy as np, pickle
    from lasagne_ext.utils import get_layer_by_name, set_weights, get_all_layers

    model_D = build_model_D()
    model_L = build_model_L()

    weight_file = r"C:\Users\dawei\Work\Code\Git\Reference Codes\Lasagne_Recipes\modelzoo\vgg16.pkl"
    with open(weight_file, mode='rb') as f:
        weights = pickle.load(f, encoding='latin1')
    lasagne.layers.set_all_param_values(model_L['prob'], weights['param values'])
    for layer_name in [
        'conv1_1',
        'conv1_2',
        'conv2_1',
        'conv2_2',
        'conv3_1',
        'conv3_2',
        'conv3_3',
        'conv4_1',
        'conv4_2',
        'conv4_3',
        'conv5_1',
        'conv5_2',
        'conv5_3',
    ]:
        print('processing layer = ', layer_name)
        W = model_L[layer_name].W.get_value()
        b = model_L[layer_name].b.get_value()
        print('W.shape=', W.shape)
        print('b.shape=', b.shape)
        model_D.__getattribute__(layer_name).W.set_value(W)
        model_D.__getattribute__(layer_name).b.set_value(b)

    for layer_name in ['fc6', 'fc7', 'fc8']:
        print('processing layer = ', layer_name)
        W = model_L[layer_name].W.get_value()
        b = model_L[layer_name].b.get_value()
        print('W.shape=', W.shape)
        print('b.shape=', b.shape)
        model_D.__getattribute__(layer_name).W.set_value(W)
        model_D.__getattribute__(layer_name).b.set_value(b)

    gpickle.dump((model_D.get_weights(), None), 'VGG16_weights.gpkl')
    print('compiling...')
    X = model_L['input'].input_var
    y_D = model_D.predict(X)
    y_L = get_output(model_L['prob'], deterministic=True)

    fn_D = theano.function([X], y_D, no_default_updates=True)
    fn_L = theano.function([X], y_L, no_default_updates=True)
    print('run test...')
    for i in range(20):
        B, C, H, W = 4, 3, 224, 224
        x = np.random.rand(B, C, H, W).astype('float32')
        y_D = fn_D(x)
        y_L = fn_L(x)
        diff = np.sum(np.abs(y_D - y_L))
        print('i=%d, diff=%0.6f' % (i, diff))
        if diff > 1e-4:
            raise ValueError('diff is too big')









if __name__ == '__main__':

    _test_case_0()

    print('Test pass ~')
