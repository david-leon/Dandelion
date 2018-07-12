# coding:utf-8
# Test ChainCRF() class with anago's implementation
# Created   :   2, 12, 2018
# Revised   :   2, 12, 2018
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os, sys
os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine import Layer, InputSpec
import theano, numpy as np
import theano.tensor as tensor

import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)

def log_sum_exp(x, axis=None, keepdims=False):
    """
    Stable log of a sum of exponentials
    """
    x_max = tensor.max(x, axis=axis, keepdims=True)
    z = tensor.log(tensor.sum(tensor.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis, keepdims=keepdims)

def path_energy(y, x, U, b_start=None, b_end=None, mask=None):
    """Calculates the energy of a tag path y for a given input x (with mask),
    transition energies U and boundary energies b_start, b_end."""
    x = add_boundary_energy(x, b_start, b_end, mask)
    return path_energy0(y, x, U, mask)


def path_energy0(y, x, U, mask=None):
    """Path energy without boundary potential handling."""
    n_classes = K.shape(x)[2]                   # x.shape = (B, T, N)
    y_one_hot = K.one_hot(y, n_classes)         # convert integer 'y' to one-hot encoded 'y': (B, T) -> (B, T, N)

    # Tag path energy
    energy = K.sum(x * y_one_hot, 2)    # (B, T, N) -> (B, T)
    energy = K.sum(energy, 1)           # (B, T) -> (B,)

    # Transition energy
    y_t = y[:, :-1]                     # y_t, (B, T-1)
    y_tp1 = y[:, 1:]                    # y_(t+1), (B, T-1)
    U_flat = K.reshape(U, [-1])         # (N, N) -> (N*N,)
    # Convert 2-dim indices (y_t, y_tp1) of U to 1-dim indices of U_flat:
    flat_indices = y_t * n_classes + y_tp1
    U_y_t_tp1 = K.gather(U_flat, flat_indices)

    if mask is not None:
        mask = K.cast(mask, K.floatx())
        y_t_mask = mask[:, :-1]
        y_tp1_mask = mask[:, 1:]
        U_y_t_tp1 *= y_t_mask * y_tp1_mask

    energy += K.sum(U_y_t_tp1, axis=1)

    return energy  #(B,)


def sparse_chain_crf_loss(y, x, U, b_start=None, b_end=None, mask=None):
    """Given the true sparsely encoded tag sequence y, input x (with mask),
    transition energies U, boundary energies b_start and b_end, it computes
    the loss function of a Linear Chain Conditional Random Field:
    loss(y, x) = NLL(P(y|x)), where P(y|x) = exp(E(y, x)) / Z.
    So, loss(y, x) = - E(y, x) + log(Z)
    Here, E(y, x) is the tag path energy, and Z is the normalization constant.
    The values log(Z) is also called free energy.
    """
    x = add_boundary_energy(x, b_start, b_end, mask)
    energy = path_energy0(y, x, U, mask)
    energy -= free_energy0(x, U, mask)
    return K.expand_dims(-energy, -1)


def chain_crf_loss(y, x, U, b_start=None, b_end=None, mask=None):
    """Variant of sparse_chain_crf_loss but with one-hot encoded tags y."""
    y_sparse = K.argmax(y, -1)
    y_sparse = K.cast(y_sparse, 'int32')
    return sparse_chain_crf_loss(y_sparse, x, U, b_start, b_end, mask)


def add_boundary_energy(x, b_start=None, b_end=None, mask=None):
    """Given the observations x, it adds the start boundary energy b_start (resp.
    end boundary energy b_end on the start (resp. end) elements and multiplies
    the mask."""
    if mask is None:
        if b_start is not None:
            x = K.concatenate([x[:, :1, :] + b_start, x[:, 1:, :]], axis=1)  # dim_1 is T
        if b_end is not None:
            x = K.concatenate([x[:, :-1, :], x[:, -1:, :] + b_end], axis=1)
    else:
        mask = K.cast(mask, K.floatx())
        mask = K.expand_dims(mask, 2)
        x *= mask
        if b_start is not None:
            mask_r = K.concatenate([K.zeros_like(mask[:, :1]), mask[:, :-1]], axis=1)
            start_mask = K.cast(K.greater(mask, mask_r), K.floatx())
            x = x + start_mask * b_start
        if b_end is not None:
            mask_l = K.concatenate([mask[:, 1:], K.zeros_like(mask[:, -1:])], axis=1)
            end_mask = K.cast(K.greater(mask, mask_l), K.floatx())
            x = x + end_mask * b_end
    return x


def viterbi_decode(x, U, b_start=None, b_end=None, mask=None):
    """Computes the best tag sequence y for a given input x, i.e. the one that
    maximizes the value of path_energy."""
    x = add_boundary_energy(x, b_start, b_end, mask)

    alpha_0 = x[:, 0, :]   # (B, N)
    gamma_0 = K.zeros_like(alpha_0)
    initial_states = [gamma_0, alpha_0]
    # the following ``` lambda B: [K.cast(K.argmax(B, axis=1), K.floatx()), K.max(B, axis=1)], ``` means [idx_max, value_max]
    _, gamma = _forward(x,
                        lambda B: [K.cast(K.argmax(B, axis=1), K.floatx()), K.max(B, axis=1)],
                        initial_states,
                        U,
                        mask)
    # gamma: (B, T, N)
    y = _backward(gamma, mask)
    return y


def free_energy(x, U, b_start=None, b_end=None, mask=None):
    """Computes efficiently the sum of all path energies for input x, when
    runs over all possible tag sequences."""
    x = add_boundary_energy(x, b_start, b_end, mask)
    return free_energy0(x, U, mask)


def free_energy0(x, U, mask=None):
    """Free energy without boundary potential handling.
    x: (B, T, N)
    U: (N, N)
    """
    initial_states = [x[:, 0, :]]         # [B, N]
    last_alpha, _ = _forward(x,
                             lambda B: [K.logsumexp(B, axis=1)],
                             initial_states,
                             U,
                             mask)
    return last_alpha[:, 0]


def _forward(x, reduce_step, initial_states, U, mask=None):
    """Forward recurrence of the linear chain crf."""

    def _forward_step(energy_matrix_t, states):  # (B, N, N), [(N,), (B, N)]
        alpha_tm1 = states[-1]
        new_states = reduce_step(K.expand_dims(alpha_tm1, 2) + energy_matrix_t)
        return new_states[0], new_states

    U_shared = K.expand_dims(K.expand_dims(U, 0), 0)   # (N, N) -> (1, 1, N, N)

    if mask is not None:
        mask = K.cast(mask, K.floatx())
        mask_U = K.expand_dims(K.expand_dims(mask[:, :-1] * mask[:, 1:], 2), 3)
        U_shared = U_shared * mask_U

    inputs = K.expand_dims(x[:, 1:, :], 2) + U_shared     # (B, T-1, 1, N) + (1, 1, N, N) -> (B, T-1, N, N)
    inputs = K.concatenate([inputs, K.zeros_like(inputs[:, -1:, :, :])], axis=1) # (B, T-1, N, N) -> (B, T, N, N)

    last, values, _ = K.rnn(_forward_step, inputs, initial_states)
    return last, values


def batch_gather(reference, indices):
    """

    :param reference: (B, N)
    :param indices: (B,)
    :return:
    """
    ref_shape = K.shape(reference)
    batch_size = ref_shape[0]
    n_classes = ref_shape[1]
    flat_indices = K.arange(0, batch_size) * n_classes + K.flatten(indices)
    return K.gather(K.flatten(reference), flat_indices)


def _backward(gamma, mask):
    """Backward recurrence of the linear chain crf."""
    gamma = K.cast(gamma, 'int32')   # (B, T, N)

    def _backward_step(gamma_t, states):
        # print('len(states)=', len(states))
        # print(type(states))
        # y_tm1 = K.squeeze(states[0], 0)
        y_tm1 = states[0]
        y_t = batch_gather(gamma_t, y_tm1)
        # return y_t, [K.expand_dims(y_t, 0)]
        # return K.expand_dims(y_t, 0), [K.expand_dims(y_t, 0)]
        return y_t, [y_t]

    # initial_states = [K.expand_dims(K.zeros_like(gamma[:, 0, 0]), 0)]  # (1, B)
    initial_states = [K.zeros_like(gamma[:, 0, 0])]  # (1, B)
    _, y_rev, _ = K.rnn(_backward_step,
                        gamma,
                        initial_states,
                        go_backwards=True)
    y = K.reverse(y_rev, 1)

    if mask is not None:
        mask = K.cast(mask, dtype='int32')
        # mask output
        y *= mask
        # set masked values to -1
        y += -(1 - mask)
    return y

#--------- ChainCRF() -------------------#
def CRF_forward(observations, transitions):
    """

    :param observations: (B, T, N)
    :param transitions:  (N, N)
    :param viterbi:
    :param return_alpha:
    :param return_best_sequence:
    :return:
    """
    U = transitions.dimshuffle('x', 'x', 0, 1)  # (N, N) -> (1, 1, N, N)
    initial = observations[:, 0, :] # (B, N)
    tensor.unbroadcast(initial, 0, 1)
    x = observations[:, 1:, :]  # (B, T-1, N)
    x = x.dimshuffle(0, 1, 'x', 2) #(B, T-1, N) -> (B, T-1, 1, N)
    x = x + U
    # x = tensor.concatenate([x, tensor.zeros_like(x[:, -1:, :, :])], axis=1)  # (B, T-1, N, N) -> (B, T, N, N)
    x = x.dimshuffle(1, 0, 2, 3) # (T, B, N, N)

    def recurrence(energy_matrix_t, states):
        """
        :param energy_matrix_t: (B, N, N)

        :return:
        """
        alpha_tm1 = states # (B,N)
        alpha_tm1 = alpha_tm1.dimshuffle(0, 1, 'x') # (B, N, 1)
        x = alpha_tm1 + energy_matrix_t  # (B, N, N)
        new_states = log_sum_exp(x, axis=1)  # (B, N)
        return new_states

    # alpha: (T, B, N)
    alpha, _ = theano.scan(
        fn=recurrence,
        sequences=x,
        outputs_info=[initial]
    )
    # return alpha[-1, :, :]  # (B,)
    return alpha.dimshuffle(1, 0, 2)

def CRF_decode(observations, transitions):
    """

    :param observations: (B, T, N)
    :param transitions:  (N, N)
    :param viterbi:
    :param return_alpha:
    :param return_best_sequence:
    :return:
    """
    alpha_0 = observations[:, 0, :] # (B, N)
    gamma_0 = tensor.zeros_like(alpha_0, dtype='int64')

    U = transitions.dimshuffle('x', 'x', 0, 1)   # (N, N) -> (1, 1, N, N)
    x = observations[:, 1:, :]  # (B, T-1, N)
    x = x.dimshuffle(0, 1, 'x', 2) #(B, T-1, N) -> (B, T-1, 1, N)
    x = x + U    # (B, T-1 N, N)
    x = tensor.concatenate([x, tensor.zeros_like(x[:, -1:, :, :])], axis=1)  # (B, T-1, N, N) -> (B, T, N, N)
    x = x.dimshuffle(1, 0, 2, 3) # (T, B, N, N)

    def recurrence(energy_matrix_t, index_tm1, score_tm1):
        """
        :param energy_matrix_t: (B, N, N)

        :return:
        """
        score_tm1 = score_tm1.dimshuffle(0, 1, 'x') # (B, N, 1)
        x = score_tm1 + energy_matrix_t  # (B, N, N)
        index = tensor.argmax(x, axis=1) # (B, N)
        score = tensor.max(x, axis=1)    # (B, N)
        return index, score

    # gamma, alpha: (T, B, N)
    result, _ = theano.scan(
        fn=recurrence,
        sequences=x,
        outputs_info=[gamma_0, alpha_0]
    )
    gamma, alpha = result

    def backward_step(gamma_t, y_tm1):
        y_t   = batch_gather(gamma_t, y_tm1)
        return y_t

    T, B, N = gamma.shape
    initial = tensor.zeros(shape=(B,), dtype='int64')
    # y : (T, B)
    y, _ = theano.scan(
        fn=backward_step,
        sequences=gamma,
        outputs_info=[initial],
        go_backwards=True
    )
    y = y.dimshuffle(1, 0)
    y = y[:, ::-1]
    y = tensor.cast(y, 'int32')
    return y  # (B, T)
    # return gamma.dimshuffle(1, 0, 2)

#--------- ChainCRF_Lample() ------------#
def CRF_forward_nobatch(observations, transitions, viterbi=False, return_alpha=False, return_best_sequence=False):
    """
    Takes as input:
        [DV]  (T+2, N+2), (N+2, N+2)
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities must be given in the log space.
    Compute alpha, matrix of size (n_steps, n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
    """
    assert not return_best_sequence or (viterbi and not return_alpha)

    def recurrence(obs, previous, transitions):
        """

        :param obs: (N+2,)
        :param previous: (N+2,)
        :param transitions: (N+2, N+2)
        :return:
        """
        previous = previous.dimshuffle(0, 'x')  # (N+2,) -> (N+2, 1)
        obs = obs.dimshuffle('x', 0)  # (N+2,) -> (1, N+2)
        if viterbi:
            scores = previous + obs + transitions
            out = scores.max(axis=0)
            if return_best_sequence:
                out2 = scores.argmax(axis=0)
                return out, out2
            else:
                return out
        else:
            return log_sum_exp(previous + obs + transitions, axis=0)

    initial = observations[0]  # = b_s = [[small] * T, 0, small]
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=(initial, None) if return_best_sequence else initial,
        sequences=[observations[1:]],
        non_sequences=transitions
    )

    if return_alpha:
        return alpha
    elif return_best_sequence:
        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=tensor.cast(tensor.argmax(alpha[0][-1]), 'int32'),
            sequences=tensor.cast(alpha[1][::-1], 'int32')
        )
        sequence = tensor.concatenate([sequence[::-1], [tensor.argmax(alpha[0][-1])]])
        return sequence
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1],
                               axis=0)  # p(x). Here alpha is equivalent to beta in "CRF as NN Layer", Page10.

def disabled_test_case_0():
    y = tensor.imatrix('y')
    x = tensor.ftensor3('x')
    U = tensor.fmatrix('U')
    x_nobatch = tensor.fmatrix('x_nobatch')

    cost1 = free_energy0(x, U)
    cost2 = CRF_forward(x, U)
    cost3 = CRF_forward_nobatch(x_nobatch, U)
    seq1 = viterbi_decode(x, U)
    seq2 = CRF_decode(x, U)
    print('compiling f1 & f2 ...')
    f1 = theano.function([x, U], cost1)
    f2 = theano.function([x, U], cost2)
    print('compiling f3 ...')
    f3 = theano.function([x, U], seq1)
    print('compiling f4 ...')
    f4 = theano.function([x, U], seq2)
    print('compiling f5 ...')
    f5 = theano.function([x_nobatch, U], cost3)

    B, T, N = 7, 100, 20
    for i in range(1):
        x = np.random.rand(B, T, N).astype(np.float32)
        U = np.random.rand(N, N).astype(np.float32)

        c1 = f1(x, U)
        c2 = f2(x, U)
        s1 = f3(x, U)
        s2 = f4(x, U)
        print(c1)
        print(c2)
        print(c1 == c2)
        if np.all(c1 == c2):
            print('c pass')
        else:
            raise ValueError('c not same!')

        print(s1)
        print(s2)
        print(s1.shape)
        print(s2.shape)
        print(s1 == s2)
        if np.all(s1 == s2):
            print('s pass')
        else:
            raise ValueError('s not same!')

def win_test_case_1():
    """
    This test case passed on Tensorflow 1.8.0 + Win64, but failed on Tensorflow 1.8.0 + Ubuntu 14.04
    :return:
    """
    y = tensor.imatrix('y')
    x = tensor.ftensor3('x')
    U = tensor.fmatrix('U')
    x_nobatch = tensor.fmatrix('x_nobatch')

    cost2 = CRF_forward(x, U)
    cost3 = CRF_forward_nobatch(x_nobatch, U, return_alpha=True)
    seq2 = CRF_decode(x, U)
    seq3 = CRF_forward_nobatch(x_nobatch, U, viterbi=True, return_best_sequence=True)
    print('compiling f2 ...')
    f2 = theano.function([x, U], cost2)
    print('compiling f4 ...')
    f4 = theano.function([x, U], seq2)
    print('compiling f5 ...')
    f5 = theano.function([x_nobatch, U], cost3)
    print('compiling f6 ...')
    f6 = theano.function([x_nobatch, U], seq3)

    B, T, N = 2, 5, 3
    for i in range(10):
        x = np.random.rand(B, T, N).astype(np.float32)
        U = np.random.rand(N, N).astype(np.float32)

        c2 = f2(x, U)
        s2 = f4(x, U)

        c3_list = []
        s3_list = []
        for j in range(B):
            c3_nobatch = f5(x[j, :, :], U)
            s3_nobatch = f6(x[j, :, :], U)
            c3_list.append(np.expand_dims(c3_nobatch, 0))
            s3_list.append(np.expand_dims(s3_nobatch, 0))
        c3 = np.concatenate(c3_list, axis=0)
        s3 = np.concatenate(s3_list, axis=0)

        # if np.all(c2 == c3):
        #     print('c pass')
        # else:
        print(c2)
        print(c3)
        # raise ValueError('c not same!')

        print(s2)
        print(s3)
        if np.all(s2 == s3):
            print('s pass')
        else:

            raise ValueError('s not same!')

if __name__ == '__main__':

    # test_case_0()

    win_test_case_1()

    print('Test passed~')



