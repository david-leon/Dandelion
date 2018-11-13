# coding:utf-8
"""
LSTM2D implementation by alternating LSTM along different dimensions
Created   :  11,  9, 2018
Revised   :  11,  9, 2018
All rights reserved
"""
# ------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import theano.tensor as tensor
from ..module import *
from ..functional import *
from ..activation import *

class Alternate_2D_LSTM(Module):
    """
    2D LSTM implementaton by alternating 1D LSTM along different input dimensions
    Input shape = (H, W, B, C)

    """
    def __init__(self, input_dims, hidden_dim, peephole=True, initializer=init.Normal(0.1), grad_clipping=0, hidden_activation=tanh,
                 learn_ini=False, truncate_gradient=-1, mode=2):
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
        :param mode: {0|1|2}. 0: 1D LSTM results from horizontal and vertical dimensions are concatenated along the `C` dimension; 1: horizontal and vertical dimensions are processed
               sequentially, i.e., the final result = horizontal_LSTM(vertical_LSTM(input)); 2: mixed mode: final result = horizontal_LSTM(concat(input, vertical_LSTM(input)))

        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mode = mode
        if not isinstance(input_dims, (tuple, list)):
            input_dims = [input_dims]
        self.lstm_h = LSTM(input_dims=input_dims, hidden_dim=hidden_dim, peephole=peephole, initializer=initializer, grad_clipping=grad_clipping, hidden_activation=hidden_activation,
                           learn_ini=learn_ini, truncate_gradient=truncate_gradient)
        if self.mode == 0:
            lstm_w_inputdim = input_dims
        elif self.mode == 1:
            lstm_w_inputdim = hidden_dim
        elif self.mode == 2:
            lstm_w_inputdim = input_dims + [hidden_dim]
        else:
            raise ValueError('Invalid mode input: should be among {0, 1, 2}')
        self.lstm_w = LSTM(input_dims=lstm_w_inputdim, hidden_dim=hidden_dim, peephole=peephole, initializer=initializer, grad_clipping=grad_clipping, hidden_activation=hidden_activation,
                           learn_ini=learn_ini, truncate_gradient=truncate_gradient)
        self.predict = self.forward

    def forward(self, seq_input, h_ini=(None, None), c_ini=(None, None), seq_mask=None, backward=(False, False), return_final_state=False):
        """
        :param seq_input:    (H, W, B, input_dim)
        :param h_ini:        tuple of matrix (B, hidden_dim) or None, if None, then learned self.h_ini will be used
        :param c_ini:        tuple of matrix (B, hidden_dim) or None, if None, then learned self.c_ini will be used
        :param seq_mask:     (H, W, B)
        :param backward:     tuple of False/True
        :param only_return_final:  If True, only return the final sequential output
        :param return_final_state: If True, the final state of `hidden` and `cell` will be returned, both (B, hidden_dim)
        :return: (H, W, B, hidden_dim) if mode= 1 or 2, (H, W, B, 2*hidden_dim) if mode = 0
        """
        H, W, B, C = seq_input.shape
        h_ini, c_ini, backward = as_tuple(h_ini, 2), as_tuple(c_ini, 2), as_tuple(backward, 2)
        x = tensor.reshape(seq_input, (H, W*B, C))
        output_h = self.lstm_h.forward(x, h_ini=h_ini[0], c_ini=c_ini[0], seq_mask=seq_mask, backward=backward[0], return_final_state=return_final_state)

        if self.mode == 0:
            x = seq_input.dimshuffle(1, 0, 2, 3)  # (W, H, B, C)
            x = tensor.reshape(x, (W, H*B, C))
            output_w = self.lstm_w.forward(x, h_ini=h_ini[1], c_ini=c_ini[1], seq_mask=seq_mask, backward=backward[1], return_final_state=return_final_state)
            if return_final_state:
                x_h, x_w = output_h[0], output_w[0]
            else:
                x_h, x_w = output_h, output_w
            x_h, x_w = tensor.reshape(x_h, (H, W, B, -1)), tensor.reshape(x_w, (W, H, B, -1))
            x_w = x_w.dimshuffle(1, 0, 2, 3)
            x = tensor.concatenate([x_h, x_w], axis=3)
            if return_final_state:
                return x, output_h[1:], output_w[1:]
            else:
                return x

        elif self.mode == 1:
            if return_final_state:
                x = output_h[0]  # (H, W*B, hidden_dim)
            else:
                x = output_h
            x = tensor.reshape(x, (H, W, B, self.hidden_dim))
            x = x.dimshuffle(1, 0, 2, 3)
            x = tensor.reshape(x, (W, H*B, self.hidden_dim))
            output_w = self.lstm_w.forward(x, h_ini=h_ini[1], c_ini=c_ini[1], seq_mask=seq_mask, backward=backward[1], return_final_state=return_final_state)
            if return_final_state:
                x = tensor.reshape(output_w[0], (W, H, B, -1))
                x = x.dimshuffle(1, 0, 2, 3)  # (H, W, B, hidden_dim)
                return x, output_h[1:], output_w[1:]
            else:
                x = tensor.reshape(output_w, (W, H, B, -1))
                x = x.dimshuffle(1, 0, 2, 3)  # (H, W, B, hidden_dim)
                return x

        else:
            if return_final_state:
                x = output_h[0]  # (H, W*B, hidden_dim)
            else:
                x = output_h
            x = tensor.reshape(x, (H, W, B, self.hidden_dim))  # (H, W, B, hidden_dim)
            x = tensor.concatenate([seq_input, x], axis=3)
            x = x.dimshuffle(1, 0, 2, 3)
            x = tensor.reshape(x, (W, H*B, -1))
            output_w = self.lstm_w.forward(x, h_ini=h_ini[1], c_ini=c_ini[1], seq_mask=seq_mask, backward=backward[1], return_final_state=return_final_state)
            if return_final_state:
                x = tensor.reshape(output_w[0], (W, H, B, -1))
                x = x.dimshuffle(1, 0, 2, 3)  # (H, W, B, hidden_dim)
                return x, output_h[1:], output_w[1:]
            else:
                x = tensor.reshape(output_w, (W, H, B, -1))
                x = x.dimshuffle(1, 0, 2, 3)  # (H, W, B, hidden_dim)
                return x


