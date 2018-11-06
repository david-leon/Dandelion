# coding:utf-8
'''
2D LSTM implementation trial
Created   :  10, 17, 2018
Revised   :  10, 17, 2018
All rights reserved
'''
# ------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import theano
theano.config.floatX = 'float32'
floatX = theano.config.floatX
from theano import tensor
from theano.gradient import grad_clip
# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse
from dandelion.module import Module, create_uneven_weight, LSTM
import dandelion.initialization as init
from dandelion.activation import *
from dandelion.objective import *
from dandelion.update import *


import numpy as np, time

class LSTM2D(Module):
    """
    2D-LSTM layer, input shape is (H, W, B, C)
    .step() can be used as LSTMCell by setting `process_input=True`
    h_ini will be learned during training.
    Note LSTM input shape is (T, B, D)
    """

    def __init__(self, input_dims, hidden_dim, peephole=True, initializer=init.Normal(0.1), grad_clipping=0, hidden_activation=tanh,
                 learn_ini=False, truncate_gradient=-1, name=None):
        """
        :param input_dims: integer or list of integers, dimension of the input for different part, allows to have different
                            initializations for different parts of the input.
        :param hidden_dim:
        :param initializer:
        :param peephole: whether add peephole connection
        :param grad_clipping: float. Hard clip the gradients at each time step. Only the gradient values
                              above this threshold are clipped to the threshold. This is done during backprop.
        :param hidden_activation: nonlinearity applied to hidden variable, i.e., h = out_gate * hidden_activation(cell). It's recommended to use `tanh` as default.
        :param learn_ini: If True, initial hidden values will be learned.
        :param truncate_gradient: if not -1, BPTT will be used, gradient back-propagation will be performed at most `truncate_gradient` steps
        """
        super().__init__(name=name)
        if not isinstance(input_dims, (tuple, list)):
            input_dims = [input_dims]

        self.input_dims    = input_dims
        self.input_dim     = sum(input_dims)
        self.hidden_dim    = hidden_dim
        self.output_dim    = self.hidden_dim     # same with `self.hidden_dim`, for API unification.
        self.grad_clipping = grad_clipping
        self.peephole      = peephole
        self.learn_ini     = learn_ini
        self.hidden_activation = hidden_activation
        self.truncate_gradient = truncate_gradient

        W_in        = create_uneven_weight(input_dims, 4 * hidden_dim, initializer)
        b_in        = np.zeros(4*self.hidden_dim, floatX)
        W_h_left    = initializer.sample((hidden_dim, 4 * hidden_dim))
        W_h_up      = initializer.sample((hidden_dim, 4 * hidden_dim))

        self.W_in   = self.register_param(W_in)  # (input_dim, 4*hidden_dim)
        self.b_in   = self.register_param(b_in)  # (4*hidden_dim)
        self.W_h_left = self.register_param(W_h_left)  # (hidden_dim, 4*hidden_dim)
        self.W_h_up   = self.register_param(W_h_up)    # (hidden_dim, 4*hidden_dim)
        if self.learn_ini:
            self.h_ini  = self.register_param(np.zeros(self.hidden_dim, floatX))  # (hidden_dim,)  hidden initial state
            self.c_ini  = self.register_param(np.zeros(self.hidden_dim, floatX))  # (hidden_dim,)  cell initial state

        if self.peephole:
            self.w_cell_to_igate_left = self.register_param(np.squeeze(initializer.sample((hidden_dim, 1)))) # (hidden_dim,) peephole weights
            self.w_cell_to_igate_up   = self.register_param(np.squeeze(initializer.sample((hidden_dim, 1)))) # (hidden_dim,) peephole weights
            self.w_cell_to_fgate_left = self.register_param(np.squeeze(initializer.sample((hidden_dim, 1)))) # (hidden_dim,) peephole weights
            self.w_cell_to_fgate_up   = self.register_param(np.squeeze(initializer.sample((hidden_dim, 1)))) # (hidden_dim,) peephole weights
            self.w_cell_to_ogate = self.register_param(np.squeeze(initializer.sample((hidden_dim, 1)))) # (hidden_dim,) peephole weights

        self.predict = self.forward                 # predict() is the same with forward() for this layer

    def _precompute_input(self, input):
        return tensor.dot(input, self.W_in) + self.b_in   # (H, W, B, C) -> (H, W, B, 4*hidden_dim)

    def step(self, input, h_left, c_left, x_pos, h_buffer, c_buffer, width, mask=None, process_input=False):
        """
        One time step. This function can be used as LSTMCell by setting `process_input=True`.
        :param input:   (B, input_dim)
        :param h_left:  (B, hidden_dim)
        :param c_left:  (B, hidden_dim)
        :param x_pos:   int64 scalar, width dimension
        :param h_buffer: (W, B, hidden_dim)
        :param c_buffer: (W, B, hidden_dim)
        :param width:   width for x_pos rounding
        :param mask:    (B,)
        :param process_input: If possible, it is better to process the whole input sequence beforehand.
                              But sometimes this is not suitable, for example at prediction time.
        :return: h, c, both (B, hidden_dim)
        """
        if process_input:
            input = self._precompute_input(input)    # (B, 4*hidden_dim)

        h_up   = h_buffer[x_pos, :, :]  # (B, hidden_dim)
        c_up   = c_buffer[x_pos, :, :]  # (B, hidden_dim)

        gates   = input + tensor.dot(h_left, self.W_h_left) + tensor.dot(h_up, self.W_h_up)   # (B, 4*hidden_dim)
        if self.grad_clipping > 0:
            gates = grad_clip(gates, -self.grad_clipping, self.grad_clipping)

        i_gate  = gates[:, :self.hidden_dim]                     # input gate, (B, hidden_dim)
        f_gate  = gates[:, self.hidden_dim:2*self.hidden_dim]    # forget gate, (B, hidden_dim)
        c_input = gates[:, 2*self.hidden_dim:3*self.hidden_dim]  # cell input, (B, hidden_dim)
        o_gate  = gates[:, 3*self.hidden_dim:]                   # output gate, (B, hidden_dim)

        if self.peephole:
            i_gate += (c_left * self.w_cell_to_igate_left + c_up * self.w_cell_to_igate_up)
            f_gate += (c_left * self.w_cell_to_fgate_left + c_up * self.w_cell_to_fgate_up)

        i_gate  = sigmoid(i_gate)
        f_gate  = sigmoid(f_gate)
        c_input = tanh(c_input)
        c       = f_gate * (c_up + c_left) * 0.5 + i_gate * c_input  # add 0.5 coefficient for numerical stability

        if self.peephole:
            o_gate += c * self.w_cell_to_ogate
        o_gate  = sigmoid(o_gate)
        h       = o_gate * self.hidden_activation(c)

        if mask:
            h = tensor.switch(mask[:, None], h, h_left)
            c = tensor.switch(mask[:, None], c, c_left)

        h_buffer = tensor.set_subtensor(h_buffer[x_pos, :, :], h)
        c_buffer = tensor.set_subtensor(c_buffer[x_pos, :, :], c)
        x_pos = x_pos + 1
        x_pos = tensor.mod(x_pos, width)
        return h, c, x_pos, h_buffer, c_buffer

    def forward(self, seq_input, h_ini=None, c_ini=None, seq_mask=None, backward=False, only_return_final=False, return_final_state=False):
        """
        Forward for train
        :param seq_input:    (H, W, B, input_dim)
        :param h_ini:        (B, hidden_dim) or None, if None, then learned self.h_ini will be used
        :param c_ini:        (B, hidden_dim) or None, if None, then learned self.c_ini will be used
        :param seq_mask:     (H, W, B)
        :param backward:
        :param only_return_final:  If True, only return the final sequential output
        :param return_final_state: If True, the final state of `hidden` and `cell` will be returned, both (B, hidden_dim)
        :return: seq_h: (H, W, B, hidden_dim)
        """
        seq_input = self._precompute_input(seq_input)
        height, width, B, D = seq_input.shape
        if h_ini is None:
            if self.learn_ini:
                h_ini = tensor.ones((B, 1)) * self.h_ini
            else:
                h_ini = tensor.zeros((B, self.hidden_dim))
        if c_ini is None:
            if self.learn_ini:
                c_ini = tensor.ones((B, 1)) * self.c_ini
            else:
                c_ini = tensor.zeros((B, self.hidden_dim))

        def LSTM_step_no_mask(input, h_pre, c_pre, x_pos, h_buffer, c_buffer, width):
            return self.step(input, h_pre, c_pre, x_pos, h_buffer, c_buffer, width, mask=None, process_input=False)

        def LSTM_step_mask(input, mask, h_pre, c_pre, x_pos, h_buffer, c_buffer, width):
            return self.step(input, h_pre, c_pre, x_pos, h_buffer, c_buffer, width, mask=mask, process_input=False)

        seq_input = seq_input.reshape((-1, B, D))
        if seq_mask is not None:
            LSTM_step = LSTM_step_mask
            sequences = [seq_input, seq_mask.reshape(-1, B)]  # (H, W, B, C) -> (H*W, B, C)
        else:
            LSTM_step = LSTM_step_no_mask
            sequences = [seq_input]

        h_buffer, c_buffer = tensor.zeros(shape=(width, B, self.hidden_dim)), tensor.zeros(shape=(width, B, self.hidden_dim))
        seq_h, seq_c, x_poss, h_buffers, c_buffers = theano.scan(
            fn=LSTM_step,
            sequences=sequences,
            outputs_info=[h_ini, c_ini, tensor.constant(0, dtype='int64'), h_buffer, c_buffer],
            non_sequences=[width],
            go_backwards=backward,
            truncate_gradient=self.truncate_gradient)[0]

        final_state = (seq_h[-1], seq_c[-1]) # both (B, hidden_dim)

        if only_return_final:
            seq_h = seq_h[-1]    # (B, hidden_dim)
        else:
            if backward:
                seq_h = seq_h[::-1, :]  # if backward, result should be reverted by time
            seq_h = seq_h.reshape((height, width, B, self.hidden_dim))

        if return_final_state:
            return (seq_h, *final_state)
        else:
            return seq_h  # (H, W, B, hidden_dim) or (B, hidden_dim)


def test_1():
    def step(x, h, c, x_pos, y_pos, H, C):
        print('x =', x)
        print('h =', h)
        print('c =', c)
        print('x_pos = ', x_pos)
        tmp = H[x_pos, y_pos]
        print('tmp=', tmp)

        h = x + tmp
        c = x + c
        x_pos = x_pos + 1
        y_pos = ifelse(x_pos >= height, y_pos+1, y_pos)
        x_pos = tensor.mod(x_pos, height)
        y_pos = tensor.mod(y_pos, width)
        # H = tensor. H[x_pos, y_pos] = hheano
        return h, c, x_pos, y_pos

    input = tensor.fmatrix()
    sequences = [input.flatten()]
    height, width = input.shape
    h_ini, c_ini = tensor.cast(theano.shared(0.0), 'float32'), tensor.cast(theano.shared(0.0), 'float32')
    H = tensor.zeros_like(input)
    C = tensor.zeros_like(input)
    print('input=', input)
    print('H=', H)

    seq_h, seq_c, xs, ys = theano.scan(
        fn=step,
        sequences=sequences,
        outputs_info=[h_ini, c_ini, tensor.constant(0, dtype='int64'), tensor.constant(0, dtype='int64')],
        non_sequences=[H, C]
        )[0]
    print('compiling function ...')
    f = theano.function([input], [seq_h, seq_c, xs, ys])
    print('run function ...')

    X = np.random.rand(5, 6).astype('float32')
    o1, o2, o3, o4 = f(X)
    print('X=',X)
    print('seq_h=',o1)
    print('seq_c=',o2)
    print('xs=',o3)
    print('ys=',o4)

def test_2():
    def step(x, h, c, x_pos, y_pos, H, C):
        print('x =', x)
        print('h =', h)
        print('c =', c)
        print('x_pos = ', x_pos)
        tmp = H[y_pos, x_pos]
        print('tmp=', tmp)

        h = x + tmp
        c = x + c

        H = tensor.set_subtensor(H[y_pos, x_pos], h)
        C = tensor.set_subtensor(C[y_pos, x_pos], c)

        x_pos = x_pos + 1
        y_pos = ifelse(x_pos >=width, y_pos+1, y_pos)
        x_pos = tensor.mod(x_pos, width)
        y_pos = tensor.mod(y_pos, height)

        return h, c, x_pos, y_pos, H, C

    input = tensor.fmatrix()
    sequences = [input.flatten()]
    height, width = input.shape
    h_ini, c_ini = tensor.cast(theano.shared(0.0), 'float32'), tensor.cast(theano.shared(0.0), 'float32')
    H = tensor.zeros_like(input)
    C = tensor.zeros_like(input)
    print('input=', input)
    print('H=', H)

    seq_h, seq_c, xs, ys, Hs, Cs = theano.scan(
        fn=step,
        sequences=sequences,
        outputs_info=[h_ini, c_ini, tensor.constant(0, dtype='int64'), tensor.constant(0, dtype='int64'), H, C],
        # non_sequences=[H, C]
        )[0]
    print('compiling function ...')
    f = theano.function([input], [seq_h, seq_c, xs, ys, Hs, Cs])
    print('run function ...')

    X = np.random.rand(3, 5).astype('float32')
    o1, o2, o3, o4, o5, o6 = f(X)
    print('X=',X)
    print('seq_h=',o1)
    print('seq_c=',o2)
    print('xs=',o3)
    print('ys=',o4)
    print('H=', o5[-1])
    print('C=', o6[-1].flatten())
    print('X.accsum=', X.flatten().cumsum())

def test_3():
    input = tensor.fmatrix()
    sequences = [input.flatten()]
    height, width = input.shape
    h_ini, c_ini = tensor.cast(theano.shared(0.0), 'float32'), tensor.cast(theano.shared(0.0), 'float32')
    # H = tensor.zeros_like(input)
    # C = tensor.zeros_like(input)

    H = tensor.zeros((height, width))
    C = tensor.zeros_like(input)

    def step(x, h, c, x_pos, y_pos):
        global H, C
        print('x =', x)
        print('h =', h)
        print('c =', c)
        print('x_pos = ', x_pos)
        tmp = H[y_pos, x_pos]
        print('tmp=', tmp)

        h = x + tmp
        c = x + c

        H = tensor.inc_subtensor(H[y_pos, x_pos], h, tolerate_inplace_aliasing=True)
        C = tensor.set_subtensor(C[y_pos, x_pos], c)

        x_pos = x_pos + 1
        y_pos = ifelse(x_pos >=width, y_pos+1, y_pos)
        x_pos = tensor.mod(x_pos, width)
        y_pos = tensor.mod(y_pos, height)

        return h, c, x_pos, y_pos


    print('input=', input)
    print('H=', H)

    seq_h, seq_c, xs, ys = theano.scan(
        fn=step,
        sequences=sequences,
        outputs_info=[h_ini, c_ini, tensor.constant(0, dtype='int64'), tensor.constant(0, dtype='int64')],
        # non_sequences=[H, C]
        )[0]
    print('compiling function ...')
    f = theano.function([input], [seq_h, seq_c, xs, ys, H, C])
    print('run function ...')

    X = np.random.rand(3, 5).astype('float32')
    o1, o2, o3, o4, o5, o6 = f(X)
    print('X=',X)
    print('seq_h=',o1)
    print('seq_c=',o2)
    print('xs=',o3)
    print('ys=',o4)
    print('H=', o5)
    print('C=', o6.flatten())
    print('X.accsum=', X.flatten().cumsum())

def test_4():
    input_dim, hidden_dim, B, H, W = 5, 4, 2, 5, 7
    input = tensor.ftensor4()
    model = LSTM2D(input_dims=[input_dim],hidden_dim=hidden_dim, peephole=True)
    output = model.forward(input, backward=True)
    print('compiling function ...')
    f = theano.function([input], output)
    theano.printing.pydotprint(f, outfile="2dlstm_graph.png", var_with_name_simple=True)

    print('run function ...')
    X = np.random.rand(H, W, B, input_dim).astype('float32')
    Y = f(X)
    print('Y=', Y)
    print('Y.shape=', Y.shape)

def test_5():
    # input_dim, hidden_dim, B, H, W = 5, 4, 2, 5, 7
    input_dim, hidden_dim, B, H, W = 8, 8, 2, 256, 256
    input = tensor.ftensor4()
    gt    = tensor.ftensor4()
    model = LSTM2D(input_dims=[input_dim],hidden_dim=hidden_dim, peephole=True)
    output = model.forward(input, backward=False)
    loss = aggregate(squared_error(output, gt))
    params = model.collect_params()
    updates = sgd(loss, params, 1e-4)
    updates.update(model.collect_self_updates())
    print('compiling function ...')
    f = theano.function([input, gt], [output, loss], updates=updates, no_default_updates=False)

    print('run function ...')
    X = np.random.rand(H, W, B, input_dim).astype('float32')
    GT = np.random.rand(H, W, B, hidden_dim).astype('float32')
    time0 = time.time()
    Y, loss = f(X, GT)
    time_used = time.time() - time0
    print('time_used = ', time_used)
    print('Y=', Y)
    print('Y.shape=', Y.shape)
    print('loss=', loss)

def test_5_LSTM():
    # input_dim, hidden_dim, B, H, W = 5, 4, 2, 5, 7
    input_dim, hidden_dim, B, H, W = 8, 8, 2, 256, 256
    x  = tensor.ftensor3()
    gt = tensor.ftensor3()
    model = LSTM(input_dims=input_dim,hidden_dim=hidden_dim, peephole=True)
    y = model.forward(x, backward=False)
    loss = aggregate(squared_error(y, gt))
    params = model.collect_params()
    updates = sgd(loss, params, 1e-4)
    updates.update(model.collect_self_updates())
    print('compiling function ...')
    f = theano.function([x, gt], [y, loss], updates=updates, no_default_updates=False)

    print('run function ...')
    X = np.random.rand(H*W, B, input_dim).astype('float32')
    GT = np.random.rand(H* W, B, hidden_dim).astype('float32')
    time0 = time.time()
    Y, loss = f(X, GT)
    time_used = time.time() - time0
    print('time_used = ', time_used)
    print('Y=', Y)
    print('Y.shape=', Y.shape)
    print('loss=', loss)

if __name__ == '__main__':
    test_5()
    test_5_LSTM()
