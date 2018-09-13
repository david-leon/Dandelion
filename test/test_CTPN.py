# coding:utf-8
'''
Test for CTPN implementation
Created   :   7, 27, 2018
Revised   :   7, 27, 2018
All rights reserved
'''
__author__ = 'dawei.leng'

import os
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"
# os.environ['THEANO_FLAGS'] = "floatX=float32, mode=DEBUG_MODE, warn_float64='raise', exception_verbosity=high"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from dandelion.model.ctpn import model_CTPN
from dandelion.objective import *

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

def test_case_0():
    model = model_CTPN(k=10, do_side_refinement_regress=False,
                 batchnorm_mode=1, channel=3, im_height=None, im_width=None,
                 kernel_size=3, border_mode=(1, 1), VGG_flip_filters=False,
                 im2col=None)
    x = tensor.ftensor4('x')
    y1 = tensor.ftensor5('y1')
    y2 = tensor.ftensor4('y2')
    class_score, bboxs = model.forward(x)
    #--- check back-prop ---#
    loss = aggregate(squared_error(y1, class_score)) + aggregate(squared_error(y2, bboxs))
    grad = theano.grad(loss, model.collect_params())
    print('back-prop test pass')


    print('compiling fn...')
    fn = theano.function([x], [class_score, bboxs], no_default_updates=False, on_unused_input='ignore')
    print('run fn...')
    input = np.random.rand(4, 3, 256, 256).astype(np.float32)
    class_score, bboxs = fn(input)
    assert class_score.shape == (4, 16, 16, 10, 2), 'class_score shape not correct'
    assert bboxs.shape == (4, 16, 16, 10, 2), 'bboxs shape not correct'

    # print(class_score.shape)
    # print(bboxs.shape)



if __name__ == '__main__':

    test_case_0()

    print('Test passed')



