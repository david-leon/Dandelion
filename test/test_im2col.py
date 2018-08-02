# coding:utf-8
'''
Test for functional.im2col
Created   :   7, 26, 2018
Revised   :   7, 26, 2018
All rights reserved
'''
__author__ = 'dawei.leng'

import os, sys
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.functional import _im2col as im2col
import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)


def _test_case_0():
    print('test case 0')
    import numpy as np
    x = tensor.ftensor4()
    y = im2col(x, nb_size=(1,1), border_mode='valid', merge_channel=False)
    f = theano.function([x], y, no_default_updates=True)

    # B, C, H, W = 1, 1, 3, 3
    for i in range(10):
        B, C, H, W = np.random.randint(1, 8), np.random.randint(1, 4), np.random.randint(3, 128), np.random.randint(3, 128)
        X = np.random.rand(B, C, H, W).astype('float32')
        Y = f(X)
        print('X.shape=', X.shape)
        print('Y.shape=', Y.shape)
        if np.all(X==Y[:,:,:,:,0]):
            pass
        else:
            diff = np.max(abs(X-Y[:,:,:,:,0]))
            print('max diff=', diff)
            print('X=', X.flatten())
            print('Y=', Y[:,:,:,:,0].flatten())
            raise ValueError('X!=Y')

def _test_case_1():
    print('test case 1')
    import numpy as np
    x = tensor.ftensor4()
    y = im2col(x, nb_size=(1,1), border_mode='valid', merge_channel=True)
    f = theano.function([x], y, no_default_updates=True)

    # B, C, H, W = 1, 1, 3, 3
    for i in range(10):
        B, C, H, W = np.random.randint(1, 8), np.random.randint(1, 4), np.random.randint(3, 128), np.random.randint(3, 128)
        X = np.random.rand(B, C, H, W).astype('float32')
        Y = f(X)
        Y2 = X.transpose((0, 2, 3, 1))
        # Y2 = np.reshape(Y2, (B, H, W, -1))
        print('Y.shape=', Y.shape)
        print('Y2.shape=', Y2.shape)
        if np.all(Y==Y2):
            pass
        else:
            diff = np.max(abs(Y-Y2))
            print('max diff=', diff)
            print('Y=',  Y.flatten())
            print('Y2=', Y2.flatten())
            raise ValueError('X!=Y')

def _test_case_2():
    print('test case 2')
    import numpy as np
    # B, C, H, W, nb_size, nb_step, border_mode = 2, 4, 6, 6, (3,3), (1,1), 'valid' # failed
    # B, C, H, W, nb_size, nb_step, border_mode = 2, 4, 6, 6, (2,2), (1,1), 'valid' # failed
    # B, C, H, W, nb_size, nb_step, border_mode = 2, 4, 5, 6, (3,3), (1,1), 'half' # pass, mode half need neighbour with odd shapes
    # B, C, H, W, nb_size, nb_step, border_mode = 2, 4, 6, 6, (2,2), (1,1), 'half' # failed
    # B, C, H, W, nb_size, nb_step, border_mode = 2, 4, 9, 9, (3,3), (1,1), 'ignore_borders' # failed
    # B, C, H, W, nb_size, nb_step, border_mode = 2, 4, 6, 6, (3,3), (1,1), 'full' # failed
    # B, C, H, W, nb_size, nb_step, border_mode = 2, 4, 5, 6, (2,2), (1,1), 'wrap_centered' # failed, mode wrap_centered need neighbour with odd shapes
    # B, C, H, W, nb_size, nb_step, border_mode = 2, 4, 5, 6, (3,3), (1,1), 'wrap_centered' # pass, mode wrap_centered need neighbour with odd shapes
    B, C, H, W, nb_size, nb_step, border_mode = 2, 4, 10, 9, (3,3), (1,1), 'wrap_centered' # pass, mode wrap_centered need neighbour with odd shapes

    x = tensor.ftensor4()
    y = im2col(x, nb_size=nb_size, border_mode=border_mode, merge_channel=False)
    f = theano.function([x], y, no_default_updates=True)

    X = np.random.rand(B, C, H, W).astype('float32')
    Y = f(X)
    print('X.shape=', X.shape)
    print('Y.shape=', Y.shape)
    if X.shape[0] == Y.shape[0] and X.shape[1] == Y.shape[1] and X.shape[2] == Y.shape[2] and  X.shape[3] == Y.shape[3] and Y.shape[-1] == nb_size[0] * nb_size[1]:
        pass
    else:
        raise ValueError('Shape not consistent')



if __name__ == '__main__':

    _test_case_0()
    _test_case_1()
    _test_case_2()

    print('Test passed')



