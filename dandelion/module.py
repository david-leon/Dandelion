# coding:utf-8
'''
  Dandelion layer pool
  Created   :   9, 20, 2017
  Revised   :   2, 24, 2018
                3,  2, 2018  add ConvTransposed2D, GRUCell, LSTMCell
                3, 21, 2018  add support of BPTT for LSTM & GRU module
                3, 22, 2018  add `use_input_mean` argument to BatchNorm's forword() interface
                3, 27, 2018  add important documentation text to `Dropout` class; change random generator to MRG_RandomStreams
                4,  8, 2018  add `trainable` flag to all modules except for `BatchNorm` and `Center`;
                             add `collect_trainable_params()` to `Module` class to get trainable parameters
                4,  9, 2018  add exception handling code to `Dropout`
                4, 10, 2018  1) add auto-naming feature to root class `Module`: if a sub-module is unnamed yet, it'll be auto-named
                             by its instance name, from now on you don't need to name a sub-module manually any more.
                             2) redesign `Module`'s parameter collecting mechanism, add `.collect_params()` and modify
                             `.collect_self_updates()`. Now `self.params` and `self.self_updating_variables` do not include
                             sub-modules' parameters any more. You'll need to call `.collect_params()` to get all the trainable
                             parameters during training function compiling.
                             3) Rewind all `trainable` flags, you're now expected to use the `include` and `exclude`
                             arguments in `.collect_params()` and `.collect_self_updates()` to enable/disable training for
                             certain module's parameters
                             4) Now `.get_weights()` and `.set_weights()` traverse the parameters in the same order of sub-modules,
                             so they're not compatible with previous version
                             5) add `.set_weights_by_name()` to `Module` class, you can use this function to set module weights saved
                             by previous version of Dandelion
                4, 11, 2018  intercept name change of `Module` class and re-name its parameters accordingly

  Note      :
    1) GRU & LSTM and their cell version have built-in activation (tanh), other modules have no built-in activations
    2)
  All rights reserved
'''
__author__ = 'dawei.leng'

import theano
theano.config.floatX = 'float32'
floatX = theano.config.floatX
from theano import tensor
from theano.gradient import grad_clip
# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import initialization as init
from .util import create_param, one_hot, batch_gather
from .activation import *
import warnings

import numpy as np
from collections import OrderedDict

def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    input_length : int or None
        The size of the input.

    filter_size : int
        The size of the filter.

    stride : int
        The stride of the convolution operation.

    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.

    Returns
    -------
    int or None
        The output size corresponding to the given convolution parameters, or
        ``None`` if `input_size` is ``None``.

    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.
    """
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length

def conv_input_length(output_length, filter_size, stride, pad=0):
    """Helper function to compute the input size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    output_length : int or None
        The size of the output.

    filter_size : int
        The size of the filter.

    stride : int
        The stride of the convolution operation.

    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.

    Returns
    -------
    int or None
        The smallest input size corresponding to the given convolution
        parameters for the given output size, or ``None`` if `output_size` is
        ``None``. For a strided convolution, any input size of up to
        ``stride - 1`` elements larger than returned will still give the same
        output size.

    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.

    Notes
    -----
    This can be used to compute the output size of a convolution backward pass,
    also called transposed convolution, fractionally-strided convolution or
    (wrongly) deconvolution in the literature.
    """
    if output_length is None:
        return None
    if pad == 'valid':
        pad = 0
    elif pad == 'full':
        pad = filter_size - 1
    elif pad == 'same':
        pad = filter_size // 2
    if not isinstance(pad, int):
        raise ValueError('Invalid pad: {0}'.format(pad))
    return (output_length - 1) * stride - 2 * pad + filter_size

def create_uneven_weight(n_ins, n_out, initializer):
    """
    This allows to have different initial scales for different parts of the input.
    [From raccoon opensourced project]
    """
    w_ins = []
    for n_in in n_ins:
        w_ins.append(initializer.sample((n_in, n_out)))
    return np.concatenate(w_ins, axis=0) / len(n_ins)

def log_sum_exp(x, axis=None, keepdims=False):
    """
    Stable log of a sum of exponentials
    """
    x_max = tensor.max(x, axis=axis, keepdims=True)
    z = tensor.log(tensor.sum(tensor.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis, keepdims=keepdims)

def gradient_normalization(grads, norm_threshold=5):
    n = tensor.sqrt(sum([tensor.sum(tensor.square(g)) for g in grads]))
    return [tensor.switch(n >= norm_threshold, g * norm_threshold / n, g) for g in grads]

class Module(object):
    """
    Base class for all neural network modules, your class should inherit this class
    """
    def __init__(self, name=None, work_mode='inference'):
        """

        :param name:
        :param work_mode: 'train' | 'inference'
        """
        self.params                  = []  # contains all the parameters which should be updated by optimizer (submodule excluded)
        self.self_updating_variables = []  # contains all the parameters which are updated by user specified expression (submodule excluded)
        self.sub_modules             = OrderedDict()
        self.name                    = name
        self.work_mode               = work_mode
        if self.work_mode not in {'train', 'inference'}:
            raise ValueError('work_mode must be set to "train" or "inference"')

    def register_param(self, x, shape=None, name=None):
        """
         Register and possibly initialize a parameter tensor.
         Parameters to be updated by optimizer should be registered here.
         All the registered parameters can be accessed via self.params attribute.

         Parameters
         ----------
         x : Theano shared variable, expression, numpy array or callable
             initial value, expression or initializer for this parameter.

         shape : tuple of int (optional)
             a tuple of integers representing the desired shape of the
             parameter tensor.

         name : str (optional)
             a descriptive name for the parameter variable. This will be passed
             to ``theano.shared`` when the variable is created, prefixed by the
             layer's name if any (in the form ``'layer_name.param_name'``). If
             ``spec`` is already a shared variable or expression, this parameter
             will be ignored to avoid overwriting an existing name.

         Returns
         -------
         Theano shared variable or Theano expression
             the resulting parameter variable or parameter expression
         """
        if name is not None:
            name = "%s@%s" % (name, self.name)
        # create shared variable, or pass through given variable/expression
        param = create_param(x, shape, name)
        self.params.append(param)
        return param

    def register_self_updating_variable(self, x, shape=None, name=None):
        """
         Register and possibly initialize a parameter tensor.
         Parameters self-updated should be registerd here.
         All the registered parameters can be accessed via self.self_updating_variables attrribute.
        :param x:
        :param shape:
        :param name:
        :return: Theano shared variable or Theano expression the resulting parameter variable or parameter expression
        """
        if name is not None:
            name = "%s@%s" % (name, self.name)
        # create shared variable, or pass through given variable/expression
        param = create_param(x, shape, name)
        self.self_updating_variables.append(param)
        return param

    def __setattr__(self, key, value):
        if isinstance(value, Module):
            if key in self.sub_modules:
                warnings.warn('Duplicate assigning to sub-module %s' % key)
            self.sub_modules[key] = value
            if self.sub_modules[key].name is None:
                self.sub_modules[key].name = key  # if the sub-module is unnamed yet, name it with it instance name

        elif isinstance(value, theano.Variable):
            if key in self.__dict__.keys():
                warnings.warn('Duplicate assigning to variable %s @%s' % (key, self.name))
            if value.name is None:
                value.name = '%s_%s@%s' % (key, self.__class__.__name__, self.name)  # a valid variable name has 3 part: [1]_[2]@[3], in which 1 is the variable instance name, 2 is the class name, and 3 is the module instance name
        
        elif key == 'name':
            for variable in self.params + self.self_updating_variables:
                substrs = variable.name.split('@')
                variable.name = substrs[0] + '@' + value
        
        object.__setattr__(self, key, value)

    def __call__(self, *args, **kwargs):
        """
        Provide a simplified & unified interface for calling a module class
        Utilization of this mechanism may reduce code clarity, so it's not recommended officially.
        :param args:
        :param kwargs:
        :return:
        """
        if self.work_mode == 'train':
            return self.forward(*args, **kwargs)
        elif self.work_mode == 'inference':
            return self.predict(*args, **kwargs)
        else:
            raise ValueError('work_mode must be set to "train" or "inference"')

    def forward(self, *args, **kwargs):
        """
        Forward function for training
        Should be overriden by all subclasses.
        """
        raise NotImplementedError('abstract function, must be defined by subclasses')

    def predict(self, *args, **kwargs):
        """
        Predict function for inference
        Should be overriden by all subclasses.
        """
        raise NotImplementedError('abstract function, must be defined by subclasses')

    def collect_params(self, include=None, exclude=None, include_self=True):
        """
        Collect parameters to be updated by optimizer
        :param include: sub-module keys, means which sub-module to include
        :param exclude: sub-module keys, means which sub-module to exclude
        :param include_self: whether include self.params
        :return: list of parameters, in the same order of sub-modules
        """
        params = []
        if include_self:
            params.extend(self.params)
        if include is None:
            include = list(self.sub_modules.keys())
        if exclude is not None:
            for key in exclude:
                include.remove(key)
        for key in include:
            params.extend(self.sub_modules[key].collect_params())
        return params

    def _collect_self_updating_variables(self, include=None, exclude=None, include_self=True):
        """
        Collect all self_updating_variables
        :param include: sub-module keys, means which sub-module to include
        :param exclude: sub-module keys, means which sub-module to exclude
        :param include_self: whether include self.params
        :return: list of self_updating variables, in the same order of sub-modules
        """
        self_updating_variables = []
        if include_self:
            self_updating_variables.extend(self.self_updating_variables)
        if include is None:
            include = list(self.sub_modules.keys())
        if exclude is not None:
            for key in exclude:
                include.remove(key)
        for key in include:
            self_updating_variables.extend(self.sub_modules[key]._collect_self_updating_variables())
        return self_updating_variables

    def _collect_params_and_self_updating_variables(self, include=None, exclude=None, include_self=True):
        """
        Collect both parameters to be updated by optimizer and self_updating_variables
        :param include: sub-module keys, means which sub-module to include
        :param exclude: sub-module keys, means which sub-module to exclude
        :param include_self: whether include self.params
        :return: list of theano tensor variables, in the same order of sub-modules
        """
        variables = []
        if include_self:
            variables.extend(self.params)
            variables.extend(self.self_updating_variables)
        if include is None:
            include = list(self.sub_modules.keys())
        if exclude is not None:
            for key in exclude:
                include.remove(key)
        for key in include:
            variables.extend(self.sub_modules[key]._collect_params_and_self_updating_variables())
        return variables

    def collect_self_updates(self, include=None, exclude=None, include_self=True):
        """
        Collect all `update` from self_updating_variables
        :param include: sub-module keys, means which sub-module to include
        :param exclude: sub-module keys, means which sub-module to exclude
        :param include_self: whether include self.self_updating_variables
        :return: update dict, in the same order of sub-modules
        """
        updates = OrderedDict()
        self_updating_variables = self._collect_self_updating_variables(include=include, exclude=exclude, include_self=include_self)
        for variable in self_updating_variables:
            updates[variable] = variable.update
        return updates

    def get_weights(self):
        """
        Collect all module weights (including submodules)
        :return: list of tuples with format [variable.value, variable.name]
        """
        module_weights = []
        variables = self._collect_params_and_self_updating_variables()
        for variable in variables:
            if hasattr(variable, 'name'):
                module_weights.append([variable.get_value(), variable.name])
            else:
                module_weights.append([variable.get_value(), None])
        return module_weights

    def set_weights(self, module_weights, check_name='ignore'):
        """
        Set module weights by default order
        :param module_weights:
        :param check_name: 'ignore'|'warn'|'raise'
        :return:
        """
        variables = self._collect_params_and_self_updating_variables()
        for weight, variable in zip(module_weights, variables):
            value, name = weight
            if name != variable.name and check_name == 'warn':
                warnings.warn('variable name inconsistent: %s <-> %s' % (variable.name, name))
            elif name != variable.name and check_name == 'raise':
                raise ValueError('variable name inconsistent: %s <-> %s' % (variable.name, name))
            variable.set_value(value)

    def set_weights_by_name(self, module_weights, unmatched='ignore', name_map=dict()):
        """
        Set module weights by matching name
        :param module_weights:
        :param unmatched: 'ignore'|'warn'|'raise', what to do if unmatched name encountered
        :param name_map: dict{weight_name: variable_name}, weight name will be mapped to the corresponding variable name, optional
        :return:
        """
        variables = self._collect_params_and_self_updating_variables()
        for value, w_name in module_weights:
            if w_name in name_map:
                w_name = name_map[w_name]
            name_match = False
            for variable in variables:
                if variable.name == w_name:
                    name_match = True
                    variable.set_value(value)
                    variables.remove(variable)
                    break
            if not name_match:
                if unmatched == 'warn':
                    warnings.warn('weight %s has no matching variable' % w_name)
                elif unmatched == 'raise':
                    raise ValueError('weight %s has no matching variable' % w_name)
        if len(variables) > 0:
            for variable in variables:
                if unmatched == 'warn':
                    warnings.warn('variable %s has no matching weight' % variable.name)
                elif unmatched == 'raise':
                    raise ValueError('variable %s has no matching weight' % variable.name)

class Dropout(Module):
    """
    Note: Theano uses `self_update` mechanism to implement pseudo randomness, so to use `Dropout` class, the followings are
          recommened:
          1) define different instance for each droput layer
          2) compiling function with `no_default_updates=False`

    .__init__(seed)
        `seed`: integer, random seed

    .forward(input, p=0.5, shared_axes=(), rescale=True)
        `p`: float, probability to drop a value (set to 0)
        `shared_axes`: tuple of int, axes to share the dropout mask over. By default, each value is dropped individually.
                       shared_axes=(0,) uses the same mask across the batch. shared_axes=(2, 3) uses the same mask across
                       the spatial dimensions of 2D feature maps.
    This class has no predict function, since there is no parameter to learn from training.
    """
    def __init__(self, seed=None, name=None):
        super().__init__(name=name)
        if seed is None:
            seed = np.random.randint(1, 2147462579)
        self.srng = RandomStreams(seed=seed)  # same with Lasagne

    def forward(self, input, p=0.5, shared_axes=(), rescale=True):
        if p <= 0:
            return input
        elif p > 1.0:
            raise ValueError('p must be within [0, 1.0]')
        one = tensor.constant(1)   # from Lasagne, using theano constant to prevent upcasting
        retain_p = one - p
        if rescale:
            input /= retain_p
        mask_shape = input.shape

        # apply dropout, respecting shared axes
        if shared_axes:
            shared_axes = tuple(a if a >= 0 else a + input.ndim
                                for a in shared_axes)
            mask_shape = tuple(1 if a in shared_axes else s
                               for a, s in enumerate(mask_shape))
        mask = self.srng.binomial(mask_shape, p=retain_p, dtype=input.dtype)
        if shared_axes:
            bcast = tuple(bool(s == 1) for s in mask_shape)
            mask = tensor.patternbroadcast(mask, bcast)
        return input * mask

    def predict(self, input, *args, **kwargs):
        """
        Here *args and **kwargs are set only for function interface unity with .forward()
        :param input:
        :param args:
        :param kwargs:
        :return:
        """
        return input

class GRU(Module):
    """
    GRU layer, basically same with Lasagne's implementation.
    .step() can be used as GRUCell by setting `process_input=True`
    h_ini will be learned during training.
    """

    def __init__(self, input_dims, hidden_dim, initializer=init.Normal(0.1), grad_clipping=0, hidden_activation=tanh,
                 learn_ini=False, truncate_gradient=-1, name=None):
        """
        Initialization same with Lasagne
        :param input_dims:  integer or list of integers, dimension of the input for different part, allows to have different
                            initializations for different parts of the input.
        :param hidden_dim:
        :param initializer:
        :param grad_clipping: float. Hard clip the gradients at each time step. Only the gradient values
                              above this threshold are clipped to the threshold. This is done during backprop.
                              Some works report that using grad_normalization is better than grad_clipping
        :param hidden_activation: nonlinearity applied to hidden variable, i.e., h = hidden_activation(cell). It's recommended to use `tanh` as default.
        :param learn_ini: If True, initial hidden values will be learned.
        :param truncate_gradient: if not -1, BPTT will be used, gradient back-propagation will be performed at most `truncate_gradient` steps
        """
        super().__init__(name=name)
        if not isinstance(input_dims, (tuple, list)):
            input_dims = [input_dims]

        self.input_dims        = input_dims
        self.input_dim         = sum(input_dims)
        self.hidden_dim        = hidden_dim
        self.output_dim        = hidden_dim  # same with `self.hidden_dim`, for API unification.
        self.grad_clipping     = grad_clipping
        self.hidden_activation = hidden_activation
        self.truncate_gradient = truncate_gradient
        self.learn_ini         = learn_ini

        W_in         = create_uneven_weight(input_dims, 3 * hidden_dim, initializer)
        b_in         = np.zeros(3*self.hidden_dim, floatX)
        W_hid        = initializer.sample((hidden_dim, 3 * hidden_dim))
        
        self.W_in    = self.register_param(W_in,  name='W_in_GRU')  # (input_dim, 3*hidden_dim)
        self.b_in    = self.register_param(b_in,  name='b_in_GRU')  # (3*hidden_dim)
        self.W_hid   = self.register_param(W_hid, name='W_hid_GRU') # (hidden_dim, 3*hidden_dim)
        if self.learn_ini:
            self.h_ini   = self.register_param(np.zeros(self.hidden_dim, floatX), name='h_ini_GRU') # (hidden_dim, )

        self.predict = self.forward                # predict() is the same with forward() for this layer

    def _precompute_input(self, input):
        return tensor.dot(input, self.W_in) + self.b_in

    def step(self, input, h_pre, mask=None, process_input=False):
        """
        One time step. This function can be used as GRUCell by setting `process_input=True`.
        :param input:   (B, input_dim)
        :param h_pre:   (B, hidden_dim)
        :param mask:    (B,)
        :param process_input: If possible, it is better to process the whole input sequence beforehand.
                              But sometimes this is not suitable, for example at prediction time.
        :return:
        """
        if process_input:
            input = self._precompute_input(input)  # (..., 3*hidden_dim)

        h_input = tensor.dot(h_pre, self.W_hid)  # (B, 3*hidden_dim), no bias here
        gates = sigmoid(input[:, :2 * self.hidden_dim] + h_input[:, :2 * self.hidden_dim])
        r_gate = gates[:, :self.hidden_dim]  # (B, hidden_dim)
        u_gate = gates[:, self.hidden_dim:2 * self.hidden_dim]  # (B, hidden_dim)
        c = self.hidden_activation(input[:, 2 * self.hidden_dim:] + r_gate * h_input[:, 2 * self.hidden_dim:])  # (B, hidden_dim)
        h = (1 - u_gate) * h_pre + u_gate * c
        if mask:
            h = tensor.switch(mask[:, None], h, h_pre)
        if self.grad_clipping > 0:
            h = grad_clip(h, -self.grad_clipping, self.grad_clipping)
        return h

    def forward(self, seq_input, h_ini=None, seq_mask=None, backward=False, only_return_final=False, return_final_state=False):
        """
        Forward for train
        :param seq_input:    (T, B, input_dim)
        :param h_ini:        (B, hidden_dim) or None, if None, then learned self.h_ini will be used
        :param seq_mask:     (T, B)
        :param backward:
        :param only_return_final:  If True, only return the final sequential output
        :param return_final_state: If True, the final state of `hidden` will be returned, (B, hidden_dim)
        :return: seq_h: (T, B, hidden_dim)
        """
        seq_input = self._precompute_input(seq_input)
        B = seq_input.shape[1]
        if h_ini is None:
            if self.learn_ini:
                h_ini = tensor.ones((B, 1)) * self.h_ini
            else:
                h_ini = tensor.zeros((B, self.hidden_dim))

        def GRU_step_no_mask(input, h_pre):
            return self.step(input, h_pre, mask=None, process_input=False)

        def GRU_step_mask(input, mask, h_pre):
            return self.step(input, h_pre, mask=mask, process_input=False)

        if seq_mask is not None:
            GRU_step = GRU_step_mask
            sequences = [seq_input, seq_mask]
        else:
            GRU_step = GRU_step_no_mask
            sequences = [seq_input]

        seq_h, _ = theano.scan(
            fn=GRU_step,
            sequences=sequences,
            outputs_info=[h_ini],
            go_backwards=backward,
            truncate_gradient=self.truncate_gradient)

        final_state = seq_h[-1]

        if only_return_final:
            seq_h = seq_h[-1]    # (B, hidden_dim)
        else:
            if backward:
                seq_h = seq_h[::-1, :]  # if backward, result should be reverted by time

        if return_final_state:
            return seq_h, final_state
        else:
            return seq_h  # (T, B, hidden_dim) or (B, hidden_dim)

class GRUCell(GRU):
    """
    Convenient wrap for GRUCell
    """
    def __init__(self, input_dims, hidden_dim, initializer=init.Normal(0.1), grad_clipping=0, hidden_activation=tanh, name=None):
        super().__init__(input_dims=input_dims, hidden_dim=hidden_dim, initializer=initializer, grad_clipping=grad_clipping,
                         hidden_activation=hidden_activation, learn_ini=False, name=name)
        self.predict = self.forward

    def forward(self, input, h_pre, mask=None):
        """

        :param input: (B, input_dim)
        :param h_pre: (B, hidden_dim)
        :param mask:  (B, input_dim)
        :return: h (B, hidden_dim)
        """
        return self.step(input, h_pre, mask, process_input=True)

class LSTM(Module):
    """
    LSTM layer, basically same with Lasagne's implementation
    .step() can be used as LSTMCell by setting `process_input=True`
    h_ini will be learned during training.
    Note LSTM input shape is (T, B, D)
    """

    def __init__(self, input_dims, hidden_dim, peephole=True, initializer=init.Normal(0.1), grad_clipping=0, hidden_activation=tanh,
                 learn_ini=False, truncate_gradient=-1, name=None):
        """
        Initialization same with Lasagne
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

        W_in         = create_uneven_weight(input_dims, 4 * hidden_dim, initializer)
        b_in         = np.zeros(4*self.hidden_dim, floatX)
        W_hid        = initializer.sample((hidden_dim, 4 * hidden_dim))

        self.W_in   = self.register_param(W_in, name='W_in_LSTM')  # (input_dim, 4*hidden_dim)
        self.b_in   = self.register_param(b_in, name='b_in_LSTM')  # (4*hidden_dim)
        self.W_hid  = self.register_param(W_hid, name='W_hid_LSTM')  # (hidden_dim, 4*hidden_dim)
        if self.learn_ini:
            self.h_ini  = self.register_param(np.zeros(self.hidden_dim, floatX), name='h_ini_LSTM')  # (hidden_dim,)  hidden initial state
            self.c_ini  = self.register_param(np.zeros(self.hidden_dim, floatX), name='c_ini_LSTM')  # (hidden_dim,)  cell initial state

        if self.peephole:
            self.w_cell_to_igate = self.register_param(np.squeeze(initializer.sample([hidden_dim, 1])), name='w_cell2igate_LSTM') # (hidden_dim,) peephole weights
            self.w_cell_to_fgate = self.register_param(np.squeeze(initializer.sample([hidden_dim, 1])), name='w_cell2fgate_LSTM') # (hidden_dim,) peephole weights
            self.w_cell_to_ogate = self.register_param(np.squeeze(initializer.sample([hidden_dim, 1])), name='w_cell2ogate_LSTM') # (hidden_dim,) peephole weights

        self.predict = self.forward                 # predict() is the same with forward() for this layer

    def _precompute_input(self, input):
        return tensor.dot(input, self.W_in) + self.b_in

    def step(self, input, h_pre, c_pre, mask=None, process_input=False):
        """
        One time step. This function can be used as LSTMCell by setting `process_input=True`.
        :param input:   (B, input_dim)
        :param h_pre:   (B, hidden_dim)
        :param c_pre:   (B, hidden_dim)
        :param mask:    (B,)
        :param process_input: If possible, it is better to process the whole input sequence beforehand.
                              But sometimes this is not suitable, for example at prediction time.
        :return: h, c, both (B, hidden_dim)
        """
        if process_input:
            input = self._precompute_input(input)    # (B, 4*hidden_dim)

        gates   = input + tensor.dot(h_pre, self.W_hid)   # (B, 4*hidden_dim)
        if self.grad_clipping > 0:
            gates = grad_clip(gates, -self.grad_clipping, self.grad_clipping)

        i_gate  = gates[:, :self.hidden_dim]                     # input gate, (B, hidden_dim)
        f_gate  = gates[:, self.hidden_dim:2*self.hidden_dim]    # forget gate, (B, hidden_dim)
        c_input = gates[:, 2*self.hidden_dim:3*self.hidden_dim]  # cell input, (B, hidden_dim)
        o_gate  = gates[:, 3*self.hidden_dim:]                   # output gate, (B, hidden_dim)

        if self.peephole:
            i_gate += c_pre * self.w_cell_to_igate
            f_gate += c_pre * self.w_cell_to_fgate

        i_gate  = sigmoid(i_gate)
        f_gate  = sigmoid(f_gate)
        c_input = tanh(c_input)
        c       = f_gate * c_pre + i_gate * c_input

        if self.peephole:
            o_gate += c * self.w_cell_to_ogate
        o_gate  = sigmoid(o_gate)
        h       = o_gate * self.hidden_activation(c)

        if mask:
            h = tensor.switch(mask[:, None], h, h_pre)
            c = tensor.switch(mask[:, None], c, c_pre)

        return h, c

    def forward(self, seq_input, h_ini=None, c_ini=None, seq_mask=None, backward=False, only_return_final=False, return_final_state=False):
        """
        Forward for train
        :param seq_input:    (T, B, input_dim)
        :param h_ini:        (B, hidden_dim) or None, if None, then learned self.h_ini will be used
        :param c_ini:        (B, hidden_dim) or None, if None, then learned self.c_ini will be used
        :param seq_mask:     (T, B)
        :param backward:
        :param only_return_final:  If True, only return the final sequential output
        :param return_final_state: If True, the final state of `hidden` and `cell` will be returned, both (B, hidden_dim)
        :return: seq_h: (T, B, hidden_dim)
        """
        seq_input = self._precompute_input(seq_input)
        B = seq_input.shape[1]
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

        def LSTM_step_no_mask(input, h_pre, c_pre):
            return self.step(input, h_pre, c_pre, mask=None, process_input=False)

        def LSTM_step_mask(input, mask, h_pre, c_pre):
            return self.step(input, h_pre, c_pre, mask=mask, process_input=False)

        if seq_mask is not None:
            LSTM_step = LSTM_step_mask
            sequences = [seq_input, seq_mask]
        else:
            LSTM_step = LSTM_step_no_mask
            sequences = [seq_input]

        seq_h, seq_c = theano.scan(
            fn=LSTM_step,
            sequences=sequences,
            outputs_info=[h_ini, c_ini],
            go_backwards=backward,
            truncate_gradient=self.truncate_gradient)[0]

        final_state = (seq_h[-1], seq_c[-1]) # both (B, hidden_dim)

        if only_return_final:
            seq_h = seq_h[-1]    # (B, hidden_dim)
        else:
            if backward:
                seq_h = seq_h[::-1, :]  # if backward, result should be reverted by time

        if return_final_state:
            return (seq_h, *final_state)
        else:
            return seq_h  # (T, B, hidden_dim) or (B, hidden_dim)

class LSTMCell(LSTM):
    """
    Convenient wrap for LSTMCell
    """

    def __init__(self, input_dims, hidden_dim, peephole=True, initializer=init.Normal(0.1), grad_clipping=0, hidden_activation=tanh, name=None):
        super().__init__(input_dims=input_dims, hidden_dim=hidden_dim, peephole=peephole, initializer=initializer,
                         grad_clipping=grad_clipping, hidden_activation=hidden_activation, learn_ini=False, name=name)
        self.predict = self.forward

    def forward(self, input, h_pre, c_pre, mask=None):
        return self.step(input, h_pre, c_pre, mask, process_input=True)

class Conv2D(Module):
    """
    Convolution 2D.
    Input shape (B, C_in, H_in, W_in)
    Ouput shape (B, C_out, H_out, W_out)
    when stride not equal to 1, input image dimension will be different from output image dimension
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), pad='valid', dilation=(1,1), num_groups=1,
                 W=init.GlorotUniform(), b=init.Constant(0.), flip_filters=True, convOP=tensor.nnet.conv2d, input_shape=(None,None), untie_bias=False,
                 name=None):
        """
        Initialization same with Lasagne.
        :param in_channels:  int, required
        :param out_channels: int, required
        :param kernel_size: 2-element tuple of int
        :param stride:  2-element tuple of int
        :param pad: 'same'/'valid'/'full'/2-element tuple of int
        :param dilation: 2-element tuple of int
        :param num_groups: refer to tensor.nnet.conv2d's documentation.
        :param W: filter bank initialization
        :param b: bias initialization, you can set it to None to disable biasing.
        :param flip_filters: convolution or correlation
        :param convOP:
        :param input_shape: 2-element tuple describing input image height and width (any dimension can be None, except when `untie_bias` = True)
        :param untie_bias: if True, `input_shape` must be given
        """
        super().__init__(name=name)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * 2
        self.flip_filters = flip_filters
        self.stride       = stride
        if isinstance(stride, int):
            self.stride = [stride] * 2
        self.dilation     = dilation
        if isinstance(dilation, int):
            self.dilation = [dilation] * 2
        self.num_groups   = num_groups
        self.untie_bias   = untie_bias
        self.convOP       = convOP
        self.input_shape  = input_shape

        if pad == 'same':
            if any(s % 2 == 0 for s in self.kernel_size):
                raise NotImplementedError('`same` padding requires odd filter size.')
            self.pad = ('same', 'same')
        elif pad == 'valid':
            self.pad = (0, 0)
        elif pad == 'full':
            self.pad = ('full', 'full')
        else:                             # pad must be 2-element tuple
            self.pad = pad

        self.W_shape = [out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]]
        self.W = self.register_param(W, self.W_shape, name='W_Conv2D')
        if b is not None:
            if untie_bias:
                if input_shape[0] is None or input_shape[1] is None:
                    raise ValueError('input_shape must be specified as 2-element tuple of int')
                output_shape = tuple(conv_output_length(input, filter, stride, p)
                                     for input, filter, stride, p in zip(input_shape, self.kernel_size, self.stride, self.pad))
                self.b = self.register_param(b, shape=[out_channels, output_shape[0], output_shape[1]], name='b_Conv2D')
            else:
                self.b = self.register_param(b, shape=[out_channels], name='b_Conv2D')

        self.predict = self.forward                # predict() is the same with forward() for this layer

    def forward(self, input):
        if self.pad[0] == 'same':
            border_mode = 'half'
        elif self.pad[0] == 'valid':
            border_mode = 'valid'
        elif self.pad[0] == 'full':
            border_mode = 'full'
        else:
            border_mode = self.pad
        conved = self.convOP(input,
                             filters=self.W,
                             input_shape=[None, self.in_channels, self.input_shape[0], self.input_shape[1]],
                             filter_shape=self.W_shape,
                             subsample=self.stride,
                             border_mode=border_mode,
                             filter_flip=self.flip_filters,
                             filter_dilation=self.dilation,
                             num_groups=self.num_groups)
        if self.b is None:
            output = conved
        elif self.untie_bias:
            output = conved + tensor.shape_padleft(self.b, 1)
        else:
            output = conved + self.b.dimshuffle(('x', 0) + ('x',) * 2)   # (B, out_channels, out_height, out_width)
        return output   # (B, out_channels, out_height, out_width)

class ConvTransposed2D(Module):
    """
    Transposed convolution 2D. Also known as fractionally-strided convolution or deconvolution (although it is not an actual deconvolution operation)
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), pad='valid', dilation=(1,1), num_groups=1,
                 W=init.GlorotUniform(), b=init.Constant(0.), flip_filters=False,
                 input_shape=(None,None), untie_bias=False,
                 name=None):
        super().__init__(name=name)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * 2
        self.flip_filters = flip_filters
        self.stride       = stride
        if isinstance(stride, int):
            self.stride = [stride] * 2
        self.dilation     = dilation
        if isinstance(dilation, int):
            self.dilation = [dilation] * 2
        self.num_groups   = num_groups
        self.untie_bias   = untie_bias
        self.input_shape  = input_shape   # tuple of 2 int or tensor variable, stands for H and W of image
        self.output_shape = [None, None]

        if pad == 'same':
            if any(s % 2 == 0 for s in self.kernel_size):
                raise NotImplementedError('`same` padding requires odd filter size.')
            self.pad = ('same', 'same')
        elif pad == 'valid':
            self.pad = (0, 0)
        elif pad == 'full':
            self.pad = ('full', 'full')
        else:                             # pad must be 2-element tuple
            self.pad = pad

        self.W_shape = [out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]]
        self.W = self.register_param(W, self.W_shape, name='W_TConv2D')
        if b is not None:
            if untie_bias:
                if input_shape[0] is None or input_shape[1] is None:
                    raise ValueError('input_shape must be specified as 2-element tuple of int')
                self.output_shape = tuple(conv_input_length(input, filter, stride, p)
                                     for input, filter, stride, p in zip(input_shape, self.kernel_size, self.stride, self.pad))
                self.b = self.register_param(b, shape=[out_channels, self.output_shape[0], self.output_shape[1]], name='b_TConv2D')
            else:
                self.b = self.register_param(b, shape=[out_channels], name='b_TConv2D')

        if self.pad[0] == 'same':
            border_mode = 'half'
        elif self.pad[0] == 'valid':
            border_mode = 'valid'
        elif self.pad[0] == 'full':
            border_mode = 'full'
        else:
            border_mode = self.pad
        self.convTOP = tensor.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=[None, self.out_channels, self.output_shape[0], self.output_shape[1]],
                                                                           kshp=self.W_shape, border_mode=border_mode,
                                                                           subsample=self.stride, filter_flip=not self.flip_filters,
                                                                           filter_dilation=self.dilation,
                                                                           num_groups=self.num_groups)
        self.predict = self.forward                # predict() is the same with forward() for this layer

    def forward(self, input):
        if any(s is None for s in self.output_shape):
            B, C, H, W = input.shape
            self.output_shape = tuple(conv_input_length(input, filter, stride, p)
                                      for input, filter, stride, p in zip([H, W], self.kernel_size, self.stride, self.pad))

        conved = self.convTOP(self.W, input, self.output_shape)
        if self.b is None:
            output = conved
        elif self.untie_bias:
            output = conved + tensor.shape_padleft(self.b, 1)
        else:
            output = conved + self.b.dimshuffle(('x', 0) + ('x',) * 2)   # (B, out_channels, out_height, out_width)
        return output   # (B, out_channels, out_height, out_width)

class Dense(Module):
    """
    Apply affine transform `Wx+b` to the last dimension of input.
    The input can have any dimensions.
    Note in Dandelion, no implicit activation applied except for LSTM / GRU
    """
    def __init__(self, input_dims, output_dim, W=init.GlorotUniform(), b=init.Constant(0.),
                 name=None):
        """
        Initialization same with Lasagne
        :param input_dims: integer or list of integers, dimension of the input for different part, allows to have different
                            initializations for different parts of the input.
        :param output_dim:
        :param b: None for no biasing
        :param initializer:
        """
        super().__init__(name=name)
        self.input_dims = input_dims if isinstance(input_dims, (list, tuple)) else [input_dims]
        self.input_dim  = sum(self.input_dims)
        self.output_dim = output_dim

        if isinstance(W, init.Initializer):
            W = create_uneven_weight(self.input_dims, self.output_dim, W)
        self.W = self.register_param(W, name='W_Dense')
        if b is not None:
            self.b = self.register_param(b, shape=[output_dim], name='b_Dense')

        self.predict = self.forward  # predict() is the same with forward() for this layer

    def forward(self, input):
        output = tensor.dot(input, self.W)
        if self.b is not None:
            output += self.b
        return output

class Embedding(Module):
    """
    Word embedding layer, same with Lasagne's implementation.
    """
    def __init__(self, num_embeddings, embedding_dim, W=init.Normal(), name=None):
        """
        :param num_embeddings: number of different embeddings
        :param embedding_dim:  embedding vector dimention
        :param W: initialization value of embedding matrix, shape = (num_embeddings, embedding_dim)
        """
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.W = self.register_param(W, shape=(num_embeddings, embedding_dim), name='W_Embedding') # (num_embeddings, embedding_dim)
        self.predict = self.forward     # predict() is the same with forward() for this layer

    def forward(self, index_input):
        """
        :param index_input: integer tensor
        :return:
        """
        return self.W[index_input]

class BatchNorm(Module):
    """
    Batch normalization using theano's built-in ops
    """
    def __init__(self, input_shape=None, axes='auto', eps=1e-4, alpha=0.1, beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), mode='high_mem',
                 name=None):
        """

        :param input_shape: tuple or list of int or tensor variable. Including batch dimension. Any shape along axis defined in `axes` can be set to None
        :param axes: 'auto' or tuple of int. The axis or axes to normalize over. If ’auto’ (the default), normalize over
                      all axes except for the second: this will normalize over the minibatch dimension for dense layers,
                      and additionally over all spatial dimensions for convolutional layers.
        :param eps: Small constant 𝜖 added to the variance before taking the square root and dividing by it, to avoid numerical problems
        :param alpha:   mean = (1 - alpha) * mean + alpha * batch_mean
        :param beta:
        :param gamma:
        :param mean:
        :param inv_std:
        :param mode:
        :param name:
        """
        super().__init__(name=name)
        self.mode = mode
        if input_shape is None:
            raise ValueError('`input_shape` must be specified for BatchNorm class')
        self.input_shape = input_shape
        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(input_shape)))
        if isinstance(axes, int):
            axes = (axes,)
        self.axes   = axes
        self.eps    = eps
        self.alpha  = alpha

        shape = [size for axis, size in enumerate(input_shape) if axis not in self.axes]  # remove all dimensions in axes
        if any(size is None for size in shape):
            raise ValueError("BatchNorm needs specified input sizes for all axes not normalized over.")

        #--- beta & gamma are trained by BP ---#
        if beta is None:
            self.beta = None
        else:
            self.beta = self.register_param(beta, shape=shape, name='beta_BN')

        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.register_param(gamma, shape=shape, name='gamma_BN')

        #--- mean & inv_std are trained by self-updating ---#
        self.mean = self.register_self_updating_variable(mean, shape=shape, name='mean_BN')
        self.inv_std = self.register_self_updating_variable(inv_std, shape=shape, name='inv_std_BN')

    def forward(self, input, use_input_mean=True):
        """

        :param input:
        :param use_input_mean: default, use mean & std of input batch for normalization; if False, self.mean and
                               self.std will be used for normalization.
                               The reason that input mean is used during training is because at the early training
                               stage, BatchNorm's self.mean is far from the expected mean value and can be detrimental
                               for network convergence.
                               It's recommended to use input mean for early stage training; after that, you can switch to
                               BatchNorm's self.mean for training & inference consistency.
        :return:
        """
        input_mean = input.mean(self.axes)
        input_inv_std = tensor.inv(tensor.sqrt(input.var(self.axes) + self.eps))

        # these `update` will be collected and removed before theano compiling
        self.mean.update = (1 - self.alpha) * self.mean + self.alpha * input_mean
        self.inv_std.update = (1 - self.alpha) * self.inv_std + self.alpha * input_inv_std

        broadcast_shape = [1] * input.ndim
        for i in range(input.ndim):
            if i not in self.axes:
                broadcast_shape[i] = self.input_shape[i]  # broadcast_shape = [1, C, 1, 1]
        if use_input_mean:
            mean = input_mean.reshape(broadcast_shape)
            inv_std = input_inv_std.reshape(broadcast_shape)
        else:
            mean = self.mean.reshape(broadcast_shape)
            inv_std = self.inv_std.reshape(broadcast_shape)
        beta = 0 if self.beta is None else tensor.reshape(self.beta, broadcast_shape)
        gamma = 1 if self.gamma is None else tensor.reshape(self.gamma, broadcast_shape)

        normalized = tensor.nnet.bn.batch_normalization(input, gamma, beta, mean, tensor.inv(inv_std),
                                           mode=self.mode)
        return normalized

    def predict(self, input):
        broadcast_shape = [1] * input.ndim
        for i in range(input.ndim):
            if i not in self.axes:
                broadcast_shape[i] = self.input_shape[i]  # broadcast_shape = [1, C, 1, 1]
        mean    = self.mean.reshape(broadcast_shape)
        inv_std = self.inv_std.reshape(broadcast_shape)
        beta    = 0 if self.beta is None else tensor.reshape(self.beta, broadcast_shape)
        gamma   = 1 if self.gamma is None else tensor.reshape(self.gamma, broadcast_shape)

        normalized = tensor.nnet.bn.batch_normalization(input, gamma, beta, mean, tensor.inv(inv_std),
                                           mode=self.mode)
        return normalized

class Center(Module):
    """
    Compute the class centers during training
    Ref. to "Discriminative feature learning approach for deep face recognition (2016)"
    """
    def __init__(self, feature_dim, center_num, alpha=0.1, center=init.GlorotUniform(), name=None):
        """
        :param alpha: moving averaging coefficient
        :param center: initial value of center
        """
        super().__init__(name=name)
        self.center = self.register_self_updating_variable(center, shape=[center_num, feature_dim], name="center")
        self.alpha = alpha

    def forward(self, features, labels):
        """

        :param features: (B, D)
        :param labels: (B,)
        :return: categorical centers
        """
        center_batch = self.center[labels, :]
        diff = self.alpha * (features - center_batch)
        center_updated = tensor.inc_subtensor(self.center[labels, :], diff)
        self.center.update = center_updated
        return self.center

    def predict(self):
        return self.center

class ChainCRF(Module):
    """
    Linear chain CRF for sequence
    """
    def __init__(self, state_num, transitions=init.GlorotUniform(), p_scale=1.0, l1_regularization=0.001, state_pad=True,
                 transition_matrix_normalization=True, name=None):
        super().__init__(name=name)
        self.state_num = state_num
        self.p_scale = p_scale
        self.l1_regularization = l1_regularization
        self.state_pad = state_pad
        self.transition_matrix_normalization = transition_matrix_normalization

        if state_pad:
            self.transitions = self.register_param(transitions, shape=[state_num+2, state_num+2], name='transition_ChainCRF')
        else:
            self.transitions = self.register_param(transitions, shape=[state_num, state_num], name='transition_ChainCRF')
        self.transitions = tensor.clip(self.transitions, 0.0001, 1.0)
        if state_pad:
            self.transitions = tensor.set_subtensor(self.transitions[:, -2], 0)  # not possible to transit from other states to <sos>
            self.transitions = tensor.set_subtensor(self.transitions[-1, :], 0)  # not possible to transit from <eos> to other states

    def forward(self, x, y):
        """
        compute CRF loss
        :param x  : output from previous LSTM layer, (B, T, N), float32
        :param y  : tag ground truth (B, T), int32
        :return: loss (B,) if self.l1_regularization disabled, else (1,)
        """
        x = x * self.p_scale

        # Score from tags
        real_path_score = tensor.sum(x * one_hot(y, self.state_num), axis=2) # (B, T, N) -> (B, T)
        real_path_score = tensor.sum(real_path_score, axis=1)   # (B, T) -> (B,)

        # Score from transitions
        if self.transition_matrix_normalization:
            self.transitions = self.transitions / self.transitions.sum(axis=1, keepdims=True)
        if self.state_pad:
            self.transitions = tensor.set_subtensor(self.transitions[:, -2], 0)  # not possible to transit from other states to <sos>
            self.transitions = tensor.set_subtensor(self.transitions[-1, :], 0)  # not possible to transit from <eos> to other states
            #--- add dummy <sos> and <eos> data to observation and y
            small = 0.0
            B, T, N = x.shape   # N = state_num
            b_s = np.array([[small] * self.state_num + [0, small]]).astype(floatX)          # begin state, (N+2,)
            b_s = tensor.ones((B, 1)) * b_s                 # (N+2,) -> (B, N+2)
            b_s = b_s.dimshuffle(0, 'x', 1)                 # (B, N+2) -> (B, 1, N+2)
            observations = tensor.concatenate([x, small * tensor.ones((B, T, 2))], axis=2)  # (B, T, N+2)
            x = tensor.concatenate([b_s, observations],  axis=1)            # (B, T+1, N+2)
            y_padded = tensor.concatenate([tensor.ones((B, 1), dtype='int32'), y, tensor.ones((B, 1), dtype='int32')], axis=1) # (B, T+2)
            y_padded = tensor.set_subtensor(y_padded[:, 0], self.state_num)
            y = tensor.set_subtensor(y_padded[:, -1], self.state_num+1)

        N = self.transitions.shape[0]
        y_t          = y[:, :-1]  # (B, T-1)
        y_tp1        = y[:, 1:]   # (B, T-1)
        U_flat       = tensor.reshape(self.transitions, [-1])  # (N, N) -> (N * N,)
        flat_indices = y_t * N + y_tp1  # (B, T-1)
        U_y_t_tp1    = U_flat[flat_indices]          # (B, T-1)
        real_path_score +=  tensor.sum(U_y_t_tp1, axis=1) # (B,)
        alpha = self.CRF_forward(x, self.transitions)  # (B,N)
        all_paths_scores = tensor.sum(alpha, axis=1)
        path_probability_logscale = real_path_score - all_paths_scores   # higher is better
        cost = - path_probability_logscale

        # regularization
        if self.l1_regularization > 0:
            transition_L1 = tensor.sum(abs(self.transitions))
            cost = tensor.sum(cost) + self.l1_regularization * transition_L1

        return cost  # (B,)

    def predict(self, x):
        """
        Viterbi decoding
        :param x  : output from previous LSTM layer, (B, T, N)
        :return: decoding result
        """
        x = x * self.p_scale
        if self.transition_matrix_normalization:
            self.transitions = self.transitions / self.transitions.sum(axis=1, keepdims=True)
        if self.state_pad:
            self.transitions = tensor.set_subtensor(self.transitions[:, -2], 0)  # not possible to transit from other states to <sos>
            self.transitions = tensor.set_subtensor(self.transitions[-1, :], 0)  # not possible to transit from <eos> to other states
            #--- add dummy <sos> and <eos> data to observation
            small = 0.0
            B, T, N = x.shape
            b_s = np.array([[small] * self.state_num + [0, small]]).astype(floatX)          # begin state, (N+2,)
            e_s = np.array([[small] * self.state_num + [small, 0]]).astype(floatX)          # end state, (N+2,)
            b_s, e_s = tensor.ones((B, 1)) * b_s, tensor.ones((B, 1)) * e_s                 # (N+2,) -> (B, N+2)
            b_s, e_s = b_s.dimshuffle(0, 'x', 1), e_s.dimshuffle(0, 'x', 1)                 # (B, N+2) -> (B, 1, N+2)
            observations = tensor.concatenate([x, small * tensor.ones((B, T, 2))], axis=2)  # (B, T, N+2)
            x = tensor.concatenate([b_s, observations],  axis=1)            # (B, T+2, N+2)

        y = self.CRF_decode(x, self.transitions)  # (B, T)
        if self.state_pad:
            return y[:, 1:]
        else:
            return y

    @staticmethod
    def CRF_forward(observations, transitions):
        """

        :param observations: (B, T, N)
        :param transitions:  (N, N)
        :return: alpha: (B, N) at the last time step
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
        return alpha[-1, :, :]  # (B, N)

    @staticmethod
    def CRF_decode(observations, transitions):
        """

        :param observations: (B, T, N)
        :param transitions:  (N, N)
        :return: best sequence (B, T)
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


if __name__ == '__main__':
    INFO = ['Dandelion framework: module pool\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)


