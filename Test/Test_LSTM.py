# coding:utf-8
# Unit test for LSTM class
# Created   :   1, 30, 2018
# Revised   :   1, 30, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

import theano
from theano import tensor
from dandelion.module import *
from dandelion.activation import *
from lasagne.layers import InputLayer, DenseLayer, LSTMLayer, get_output, Upscale2DLayer, TransposedConv2DLayer
import lasagne.nonlinearities as LACT

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

class build_model_D(Module):
    def __init__(self, in_dim=3, out_dim=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lstm = LSTM(input_dims=self.in_dim, hidden_dim=self.out_dim, peephole=True, learn_ini=True)
        self.predict = self.forward

    def forward(self, x):
        """

        :param x: (B, T, D)
        :return:
        """
        x = x.dimshuffle((1, 0, 2)) # ->(T, B, D)
        x = self.lstm.forward(x, backward=True, only_return_final=True)
        # x = x.dimshuffle((1, 0, 2)) # ->(B, T, D)
        # x = tanh(x)
        return x

def build_model_L(in_dim=3, out_dim=3):
    input_var = tensor.ftensor3('x')  # (B, T, D)
    input0 = InputLayer(shape=(None, None, in_dim), input_var=input_var, name='input0')
    lstm0 = LSTMLayer(input0, num_units=out_dim, precompute_input=True, nonlinearity=LACT.tanh,
                      backwards=True, only_return_final=True, learn_init=True, consume_less='None',
                      name='lstm0')
    return lstm0

def test_case_0():
    import numpy as np
    from lasagne_ext.utils import get_layer_by_name

    in_dim, out_dim = 32, 3
    model_D = build_model_D(in_dim=in_dim, out_dim=out_dim)
    model_L = build_model_L(in_dim=in_dim, out_dim=out_dim)

    W_in = np.random.rand(in_dim, 4*out_dim).astype(np.float32)
    b_in = np.random.rand(4*out_dim).astype(np.float32)
    W_hid = np.random.rand(out_dim, 4*out_dim).astype(np.float32)
    h_ini = np.random.rand(out_dim).astype(np.float32)
    c_ini = np.random.rand(out_dim).astype(np.float32)
    w_cell_to_igate = np.random.rand(out_dim).astype(np.float32)
    w_cell_to_fgate = np.random.rand(out_dim).astype(np.float32)
    w_cell_to_ogate = np.random.rand(out_dim).astype(np.float32)

    model_D.lstm.W_in.set_value(W_in)
    model_D.lstm.b_in.set_value(b_in)
    model_D.lstm.W_hid.set_value(W_hid)
    model_D.lstm.h_ini.set_value(h_ini)
    model_D.lstm.c_ini.set_value(c_ini)
    model_D.lstm.w_cell_to_igate.set_value(w_cell_to_igate)
    model_D.lstm.w_cell_to_fgate.set_value(w_cell_to_fgate)
    model_D.lstm.w_cell_to_ogate.set_value(w_cell_to_ogate)

    lstm_L = get_layer_by_name(model_L, 'lstm0')
    lstm_L.W_in_to_ingate.set_value(W_in[:, :out_dim])
    lstm_L.W_in_to_forgetgate.set_value(W_in[:, out_dim:2*out_dim])
    lstm_L.W_in_to_cell.set_value(W_in[:, 2*out_dim:3*out_dim])
    lstm_L.W_in_to_outgate.set_value(W_in[:, 3*out_dim:])

    lstm_L.W_hid_to_ingate.set_value(W_hid[:, :out_dim])
    lstm_L.W_hid_to_forgetgate.set_value(W_hid[:, out_dim:2*out_dim])
    lstm_L.W_hid_to_cell.set_value(W_hid[:, 2*out_dim:3*out_dim])
    lstm_L.W_hid_to_outgate.set_value(W_hid[:, 3*out_dim:])

    lstm_L.b_ingate.set_value(b_in[:out_dim])
    lstm_L.b_forgetgate.set_value(b_in[out_dim:2*out_dim])
    lstm_L.b_cell.set_value(b_in[2*out_dim:3*out_dim])
    lstm_L.b_outgate.set_value(b_in[3*out_dim:])

    lstm_L.hid_init.set_value(h_ini.reshape((1, out_dim)))
    lstm_L.cell_init.set_value(c_ini.reshape((1, out_dim)))

    lstm_L.W_cell_to_ingate.set_value(w_cell_to_igate)
    lstm_L.W_cell_to_forgetgate.set_value(w_cell_to_fgate)
    lstm_L.W_cell_to_outgate.set_value(w_cell_to_ogate)

    X = get_layer_by_name(model_L, 'input0').input_var
    y_D = model_D.forward(X)
    y_L = get_output(model_L)

    fn_D = theano.function([X], y_D, no_default_updates=True, on_unused_input='ignore')
    fn_L = theano.function([X], y_L, no_default_updates=True, on_unused_input='ignore')

    for i in range(20):
        x = np.random.rand(4, 16, in_dim).astype(np.float32)
        y_D = fn_D(x)
        y_L = fn_L(x)
        diff = np.max(np.abs(y_D - y_L))
        print('i=%d, diff=%0.6f' % (i, diff))
        if diff>1e-4:
            print('y_D=\n', y_D)
            print('y_L=\n', y_L)
            raise ValueError('diff is too big')

if __name__ == '__main__':

    test_case_0()

    print('Test passed')



