# coding:utf-8
# Test for unet, partial.
# Created   :   5, 25, 2018
# Revised   :   5, 25, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys, psutil
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from dandelion.model.unet import model_Unet

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

def test_case_0():
    im_height, im_width = 65, 63
    model = model_Unet(im_height=im_height, im_width=im_width)
    x = tensor.ftensor4('x')
    y = model.forward(x)
    print('compiling fn...')
    fn = theano.function([x], y, no_default_updates=False)
    print('run fn...')
    input = np.random.rand(7, 1, im_height, im_width).astype(np.float32)
    output = fn(input)
    print(output)
    print(output.shape)

if __name__ == '__main__':

    test_case_0()

    print('Test passed')



