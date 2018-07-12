# coding:utf-8
# Unit test for GRU class
# Created   :   1, 31, 2018
# Revised   :   1, 31, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from lasagne.layers import InputLayer, GRULayer, get_output
import lasagne.nonlinearities as LACT

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

class build_model_D(Module):
    def __init__(self, in_dim=3, out_dim=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gru = GRU(input_dims=self.in_dim, hidden_dim=self.out_dim, learn_ini=True)
        self.predict = self.forward

    def forward(self, x):
        """

        :param x: (B, T, D)
        :return:
        """
        x = x.dimshuffle((1, 0, 2)) # ->(T, B, D)
        x = self.gru.forward(x, backward=False, only_return_final=False)
        x = x.dimshuffle((1, 0, 2)) # ->(B, T, D)
        # x = tanh(x)
        return x

def build_model_L(in_dim=3, out_dim=3):
    input_var = tensor.ftensor3('x')  # (B, T, D)
    input0 = InputLayer(shape=(None, None, in_dim), input_var=input_var, name='input0')
    gru0 = GRULayer(input0, num_units=out_dim, precompute_input=True,
                      backwards=False, only_return_final=False, learn_init=True,
                      name='gru0')
    return gru0

def test_case_0():
    import numpy as np
    from lasagne_ext.utils import get_layer_by_name

    in_dim, out_dim = 6, 5
    model_D = build_model_D(in_dim=in_dim, out_dim=out_dim)
    model_L = build_model_L(in_dim=in_dim, out_dim=out_dim)

    W_in = np.random.rand(in_dim, 3 * out_dim).astype(np.float32)
    b_in = np.random.rand(3 * out_dim).astype(np.float32)
    W_hid = np.random.rand(out_dim, 3 * out_dim).astype(np.float32)
    h_ini = np.random.rand(out_dim).astype(np.float32)

    model_D.gru.W_in.set_value(W_in)
    model_D.gru.b_in.set_value(b_in)
    model_D.gru.W_hid.set_value(W_hid)
    model_D.gru.h_ini.set_value(h_ini)

    gru_L = get_layer_by_name(model_L, 'gru0')
    gru_L.W_in_to_resetgate.set_value(W_in[:, :out_dim])
    gru_L.W_in_to_updategate.set_value(W_in[:, out_dim:2 * out_dim])
    gru_L.W_in_to_hidden_update.set_value(W_in[:, 2 * out_dim:3 * out_dim])

    gru_L.W_hid_to_resetgate.set_value(W_hid[:, :out_dim])
    gru_L.W_hid_to_updategate.set_value(W_hid[:, out_dim:2 * out_dim])
    gru_L.W_hid_to_hidden_update.set_value(W_hid[:, 2 * out_dim:3 * out_dim])

    gru_L.b_resetgate.set_value(b_in[:out_dim])
    gru_L.b_updategate.set_value(b_in[out_dim:2 * out_dim])
    gru_L.b_hidden_update.set_value(b_in[2 * out_dim:3 * out_dim])

    gru_L.hid_init.set_value(h_ini.reshape((1, out_dim)))

    X = get_layer_by_name(model_L, 'input0').input_var
    y_D = model_D.forward(X)
    y_L = get_output(model_L)

    fn_D = theano.function([X], y_D, no_default_updates=True, on_unused_input='ignore')
    fn_L = theano.function([X], y_L, no_default_updates=True, on_unused_input='ignore')

    for i in range(20):
        x = np.random.rand(2, 5, in_dim).astype(np.float32) - 0.5
        y_D = fn_D(x)
        y_L = fn_L(x)
        diff = np.max(np.abs(y_D - y_L))
        print('i=%d, diff=%0.6f' % (i, diff))
        if diff > 1e-4:
            print('y_D=\n', y_D)
            print('y_L=\n', y_L)
            raise ValueError('diff is too big')


if __name__ == '__main__':

    test_case_0()

    print('Test passed')



