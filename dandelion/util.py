
import numpy as np
import theano
import theano.tensor as tensor
import gzip, pickle
import socket, time

def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.

    Parameters
    ----------
    arr : array_like
        The data to be converted.

    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return np.asarray(arr, dtype=theano.config.floatX)

def shared_empty(dim=2, dtype=None):
    """Creates empty Theano shared variable.

    Shortcut to create an empty Theano shared variable with
    the specified number of dimensions.

    Parameters
    ----------
    dim : int, optional
        The number of dimensions for the empty variable, defaults to 2.
    dtype : a numpy data-type, optional
        The desired dtype for the variable. Defaults to the Theano
        ``floatX`` dtype.

    Returns
    -------
    Theano shared variable
        An empty Theano shared variable of dtype ``dtype`` with
        `dim` dimensions.
    """
    if dtype is None:
        dtype = theano.config.floatX

    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype=dtype))

def as_theano_expression(input):
    """Wrap as Theano expression.

    Wraps the given input as a Theano constant if it is not
    a valid Theano expression already. Useful to transparently
    handle numpy arrays and Python scalars, for example.

    Parameters
    ----------
    input : number, numpy array or Theano expression
        Expression to be converted to a Theano constant.

    Returns
    -------
    Theano symbolic constant
        Theano constant version of `input`.
    """
    if isinstance(input, theano.gof.Variable):
        return input
    else:
        try:
            return theano.tensor.constant(input)
        except Exception as e:
            raise TypeError("Input of type %s is not a Theano expression and "
                            "cannot be wrapped as a Theano constant (original "
                            "exception: %s)" % (type(input), e))

def collect_shared_vars(expressions):
    """Returns all shared variables the given expression(s) depend on.

    Parameters
    ----------
    expressions : Theano expression or iterable of Theano expressions
        The expressions to collect shared variables from.

    Returns
    -------
    list of Theano shared variables
        All shared variables the given expression(s) depend on, in fixed order
        (as found by a left-recursive depth-first search). If some expressions
        are shared variables themselves, they are included in the result.
    """
    # wrap single expression in list
    if isinstance(expressions, theano.Variable):
        expressions = [expressions]
    # return list of all shared variables
    return [v for v in theano.gof.graph.inputs(reversed(expressions))
            if isinstance(v, theano.compile.SharedVariable)]

def one_hot(x, m=None):
    """One-hot representation of integer vector.

    Given a vector of integers from 0 to m-1, returns a matrix
    with a one-hot representation, where each row corresponds
    to an element of x.

    Parameters
    ----------
    x : integer vector
        The integer vector to convert to a one-hot representation.
    m : int, optional
        The number of different columns for the one-hot representation. This
        needs to be strictly greater than the maximum value of `x`.
        Defaults to ``max(x) + 1``.

    Returns
    -------
    Theano tensor variable
        A Theano tensor variable of shape (``n``, `m`), where ``n`` is the
        length of `x`, with the one-hot representation of `x`.

    Notes
    -----
    If your integer vector represents target class memberships, and you wish to
    compute the cross-entropy between predictions and the target class
    memberships, then there is no need to use this function, since the function
    :func:`lasagne.objectives.categorical_crossentropy()` can compute the
    cross-entropy from the integer vector directly.

    """
    if m is None:
        m = tensor.cast(tensor.max(x) + 1, 'int32')

    return tensor.eye(m)[tensor.cast(x, 'int32')]

def batch_gather(x, index):
    """

    :param x: (B, N)
    :param index: (B,)
    :return:
    """
    B, N = x.shape
    flat_index = tensor.arange(0, B) * N + tensor.flatten(index)
    x = tensor.flatten(x)
    return x[flat_index]

def unique(l):
    """Filters duplicates of iterable.

    Create a new list from l with duplicate entries removed,
    while preserving the original order.

    Parameters
    ----------
    l : iterable
        Input iterable to filter of duplicates.

    Returns
    -------
    list
        A list of elements of `l` without duplicates and in the same order.
    """
    new_list = []
    seen = set()
    for el in l:
        if el not in seen:
            new_list.append(el)
            seen.add(el)

    return new_list

def as_tuple(x, N, t=None):
    """
    Coerce a value to a tuple of given length (and possibly given type).

    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type, optional
        required type for all elements

    Returns
    -------
    tuple
        ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.

    Raises
    ------
    TypeError
        if `type` is given and `x` or any of its elements do not match it
    ValueError
        if `x` is iterable, but does not have exactly `N` elements
    """
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X

def inspect_kwargs(func):
    """
    Inspects a callable and returns a list of all optional keyword arguments.

    Parameters
    ----------
    func : callable
        The callable to inspect

    Returns
    -------
    kwargs : list of str
        Names of all arguments of `func` that have a default value, in order
    """
    # We try the Python 3.x way first, then fall back to the Python 2.x way
    try:
        from inspect import signature
    except ImportError:  # pragma: no cover
        from inspect import getargspec
        spec = getargspec(func)
        return spec.args[-len(spec.defaults):] if spec.defaults else []
    else:  # pragma: no cover
        params = signature(func).parameters
        return [p.name for p in params.values() if p.default is not p.empty]

def compute_norms(array, norm_axes=None):
    """ Compute incoming weight vector norms.

    Parameters
    ----------
    array : numpy array or Theano expression
        Weight or bias.
    norm_axes : sequence (list or tuple)
        The axes over which to compute the norm.  This overrides the
        default norm axes defined for the number of dimensions
        in `array`. When this is not specified and `array` is a 2D array,
        this is set to `(0,)`. If `array` is a 3D, 4D or 5D array, it is
        set to a tuple listing all axes but axis 0. The former default is
        useful for working with dense layers, the latter is useful for 1D,
        2D and 3D convolutional layers.
        Finally, in case `array` is a vector, `norm_axes` is set to an empty
        tuple, and this function will simply return the absolute value for
        each element. This is useful when the function is applied to all
        parameters of the network, including the bias, without distinction.
        (Optional)

    Returns
    -------
    norms : 1D array or Theano vector (1D)
        1D array or Theano vector of incoming weight/bias vector norms.

    Examples
    --------
    >>> array = np.random.randn(100, 200)
    >>> norms = compute_norms(array)
    >>> norms.shape
    (200,)

    >>> norms = compute_norms(array, norm_axes=(1,))
    >>> norms.shape
    (100,)
    """

    # Check if supported type
    if not isinstance(array, theano.Variable) and \
       not isinstance(array, np.ndarray):
        raise RuntimeError(
            "Unsupported type {}. "
            "Only theano variables and numpy arrays "
            "are supported".format(type(array))
        )

    # Compute default axes to sum over
    ndim = array.ndim
    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 1:          # For Biases that are in 1d (e.g. b of DenseLayer)
        sum_over = ()
    elif ndim == 2:          # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}. "
            "Must specify `norm_axes`".format(array.ndim)
        )

    # Run numpy or Theano norm computation
    if isinstance(array, theano.Variable):
        # Apply theano version if it is a theano variable
        if len(sum_over) == 0:
            norms = tensor.abs_(array)   # abs if we have nothing to sum over
        else:
            norms = tensor.sqrt(tensor.sum(array**2, axis=sum_over))
    elif isinstance(array, np.ndarray):
        # Apply the numpy version if ndarray
        if len(sum_over) == 0:
            norms = abs(array)     # abs if we have nothing to sum over
        else:
            norms = np.sqrt(np.sum(array**2, axis=sum_over))

    return norms

def create_param(spec, shape=None, name=None, dim_broadcast=None):
    """
    Helper method to create Theano shared variables for layer parameters
    and to initialize them. Modified from Lasagne.utils.py

    Parameters
    ----------
    spec : scalar number, numpy array, Theano expression, or callable
        Either of the following:

        * a scalar or a numpy array with the initial parameter values
        * a Theano expression or shared variable representing the parameters
        * a function or callable that takes the desired shape of
          the parameter array as its single argument and returns
          a numpy array, a Theano expression, or a shared variable
          representing the parameters.

    shape : iterable of int (optional)
        a tuple or other iterable of integers representing the desired
        shape of the parameter array.

    dim_broadcast: tuple or list of boolean, indicating each dimension is broadcastable or not.
                   if None, then no dimension is explicitly set as broadcastable.

    name : string (optional)
        The name to give to the parameter variable. Ignored if `spec`
        is or returns a Theano expression or shared variable that
        already has a name.

    Returns
    -------
    Theano shared variable or Theano expression
        A Theano shared variable or expression representing layer parameters.
        If a scalar or a numpy array was provided, a shared variable is
        initialized to contain this array. If a shared variable or expression
        was provided, it is simply returned. If a callable was provided, it is
        called, and its output is used to initialize a shared variable.
    """
    import numbers  # to check if argument is a number
    if shape is not None:
        shape = tuple(shape)  # convert to tuple if needed
        if any(d <= 0 for d in shape):
            raise ValueError((
                "Cannot create param with a non-positive shape dimension. "
                "Tried to create param with shape=%r, name=%r") % (shape, name))

    err_prefix = "cannot initialize parameter %s: " % name
    if callable(spec):
        spec = spec(shape)
        err_prefix += "the %s returned by the provided callable"
    else:
        err_prefix += "the provided %s"

    if isinstance(spec, numbers.Number) or isinstance(spec, np.generic) \
            and spec.dtype.kind in 'biufc':
        spec = np.asarray(spec)

    if isinstance(spec, np.ndarray):
        if shape is not None and spec.shape != shape:
            raise ValueError("%s has shape %s, should be %s" %
                             (err_prefix % "numpy array", spec.shape, shape))
        if dim_broadcast is None:
            spec = theano.shared(spec)
        else:
            spec = theano.shared(spec, broadcastable=dim_broadcast)
        # if shape is None:
            # spec = theano.shared(spec)
        # else:
            # bcast = tuple(s == 1 for s in shape)
            # spec = theano.shared(spec, broadcastable=bcast)

    # if dim_broadcast is not None:
        # spec = tensor.patternbroadcast(spec, dim_broadcast)

    if isinstance(spec, theano.Variable):
        # We cannot check the shape here, Theano expressions (even shared
        # variables) do not have a fixed compile-time shape. We can check the
        # dimensionality though.
        if shape is not None and spec.ndim != len(shape):
            raise ValueError("%s has %d dimensions, should be %d" %
                             (err_prefix % "Theano variable", spec.ndim,
                              len(shape)))
        # We only assign a name if the user hasn't done so already.
        if not spec.name:
            spec.name = name
        return spec

    else:
        if "callable" in err_prefix:
            raise TypeError("%s is not a numpy array or a Theano expression" %
                            (err_prefix % "value"))
        else:
            raise TypeError("%s is not a numpy array, a Theano expression, "
                            "or a callable" % (err_prefix % "spec"))

def unroll_scan(fn, sequences, outputs_info, non_sequences, n_steps,
                go_backwards=False):
        """
        Helper function to unroll for loops. Can be used to unroll theano.scan.
        The parameter names are identical to theano.scan, please refer to here
        for more information.

        Note that this function does not support the truncate_gradient
        setting from theano.scan.

        Parameters
        ----------

        fn : function
            Function that defines calculations at each step.

        sequences : TensorVariable or list of TensorVariables
            List of TensorVariable with sequence data. The function iterates
            over the first dimension of each TensorVariable.

        outputs_info : list of TensorVariables
            List of tensors specifying the initial values for each recurrent
            value.

        non_sequences: list of TensorVariables
            List of theano.shared variables that are used in the step function.

        n_steps: int
            Number of steps to unroll.

        go_backwards: bool
            If true the recursion starts at sequences[-1] and iterates
            backwards.

        Returns
        -------
        List of TensorVariables. Each element in the list gives the recurrent
        values at each time step.

        """
        if not isinstance(sequences, (list, tuple)):
            sequences = [sequences]

        # When backwards reverse the recursion direction
        counter = range(n_steps)
        if go_backwards:
            counter = counter[::-1]

        output = []
        prev_vals = outputs_info
        for i in counter:
            step_input = [s[i] for s in sequences] + prev_vals + non_sequences
            out_ = fn(*step_input)
            # The returned values from step can be either a TensorVariable,
            # a list, or a tuple.  Below, we force it to always be a list.
            if isinstance(out_, tensor.TensorVariable):
                out_ = [out_]
            if isinstance(out_, tuple):
                out_ = list(out_)
            output.append(out_)

            prev_vals = output[-1]

        # iterate over each scan output and convert it to same format as scan:
        # [[output11, output12,...output1n],
        # [output21, output22,...output2n],...]
        output_scan = []
        for i in range(len(output[0])):
            l = map(lambda x: x[i], output)
            output_scan.append(tensor.stack(*l))

        return output_scan

class chunked_byte_writer(object):
    """
    This class is used for by-passing the bug in gzip/zlib library: when data length exceeds unsigned int limit, gzip/zlib will break
    file: a file object
    """
    def __init__(self, file, chunksize=4294967295):
        self.file = file
        self.chunksize = chunksize
    def write(self, data):
        for i in range(0, len(data), self.chunksize):
            self.file.write(data[i:i+self.chunksize])

class gpickle(object):
    """
    A pickle class with gzip enabled
    """
    @staticmethod
    def dump(data, filename, compresslevel=9):
        with gzip.open(filename, mode='wb', compresslevel=compresslevel) as f:
            pickle.dump(data, chunked_byte_writer(f))
            f.close()
    @staticmethod
    def load(filename):
        """
        The chunked read mechanism here is for by-passing the bug in gzip/zlib library: when
        data length exceeds unsigned int limit, gzip/zlib will break
        :param filename:
        :return:
        """
        buf = b''
        chunk = b'NULL'
        with gzip.open(filename, mode='rb') as f:
            while len(chunk) > 0:
                chunk = f.read(429496729)
                buf += chunk
        data = pickle.loads(buf)
        return data

    @staticmethod
    def loads(buf):
        return pickle.loads(buf)

    @staticmethod
    def dumps(data):
        return pickle.dumps(data)

def get_weight_by_name(module_weights, name):
    """
    Retrieve parameter weight values from result of Module.get_weights()
    :param module_weights: all weights of a module, returned by Module.get_weights()
    :param name:
    :return:
    """
    for w, w_name in module_weights:
        if name == w_name:
            return w
    return None

def theano_safe_run(fn, input_list):
    """
    Help catch theano memory exceptions during running theano functions.
    :param fn:
    :param input_list:
    :return: (status, result), status > 0 means exception catched.
    """
    try:
        result = fn(*input_list)
        status = 0
        return status, result
    except MemoryError:
        print('Memory error catched')
        status = 1
        return status, None
    except RuntimeError as e:
        print('RuntimeError encountered')
        if e.args[0].startswith('CudaNdarray_ZEROS: allocation failed.'):
            print('Memory error catched')
            status = 2
            return status, None
        elif str(e).startswith('gpudata_alloc: cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY: out of memory'):
            print('Memory error catched')
            status = 3
            return status, None
        else:
            raise e
    except Exception as e:
        if e.args[0].startswith("b'cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY: out of memory'"):
            print('New backend memory error catched')
            status = 4
            return status, None
        else:
            raise e

class finite_memory_array(object):

    def __init__(self, array=None, shape=None, dtype=np.double):
        if array is not None:
            self.array = array
        elif shape is not None:
            self.array = np.zeros(shape=shape, dtype=dtype)
        else:
            raise ValueError('bad initialization para for Finite_Memory_Array')
        self.shape = self.array.shape
        self.curpos = 0
        self.first_round = True

    def update(self, value):
        self.array[:, self.curpos] = value
        self.curpos += 1
        if self.curpos >= self.shape[1]:
            self.curpos = 0
            self.first_round = False

    def clear(self):
        self.array[:] = 0
        self.curpos = 0
        self.first_round = True

    def get_current_position(self):
        return self.curpos

    @property
    def content(self):
        if self.first_round is True:
            return self.array[:, :self.curpos]
        else:
            return self.array

Finite_Memory_Array = finite_memory_array

def pad_sequence_into_array(Xs, maxlen=None, truncating='post', padding='post', value=0.):
    """
    Padding sequence (list of numpy arrays) into an numpy array
    :param Xs: list of numpy arrays. The arrays must have the same shape except the first dimension.
    :param maxlen: the allowed maximum of the first dimension of Xs's arrays. Any array longer than maxlen is truncated to maxlen
    :param truncating: = 'pre'/'post', indicating whether the truncation happens at either the beginning or the end of the array (default)
    :param padding: = 'pre'/'post',indicating whether the padding happens at either the beginning or the end of the array (default)
    :param value: scalar, the padding value, default = 0.0
    :return: Xout, the padded sequence (now an augmented array with shape (Narrays, N1stdim, N2nddim, ...)
    :return: mask, the corresponding mask, binary array, with shape (Narray, N1stdim)
    """
    Nsamples = len(Xs)
    if maxlen is None:
        lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
        maxlen = np.max(lengths)

    Xout = np.ones(shape = [Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
    Mask = np.zeros(shape = [Nsamples, maxlen], dtype=Xout.dtype)
    for i in range(Nsamples):
        x = Xs[i]
        if truncating == 'pre':
            trunc = x[-maxlen:]
        elif truncating == 'post':
            trunc = x[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
        if padding == 'post':
            Xout[i, :len(trunc)] = trunc
            Mask[i, :len(trunc)] = 1
        elif padding == 'pre':
            Xout[i, -len(trunc):] = trunc
            Mask[i, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return Xout, Mask

def get_local_ip():
    """
    Get local host IP address
    :return:
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 0))                                # connecting to a UDP address doesn't send packets
    local_ip_address = s.getsockname()[0]
    return local_ip_address

def get_time_stamp():
    """
    Create a formatted string time stamp
    :return:
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

class sys_output_tap:
    """
    Helper class to redirect sys output to file meanwhile keeping the output on screen
    Based on code snippet from https://github.com/smlaine2/tempens/blob/master/train.py#L30
    Example:
        #--- setup ---#
        stdout_tap = sys_output_tap(sys.stdout)
        stderr_tap = sys_output_tap(sys.stderr)
        sys.stdout = stdout_tap
        sys.stderr = stderr_tap
        #--- before training ---#
        stdout_tap.set_file(open(os.path.join(save_folder, 'stdout.txt'), 'wt'))
        stderr_tap.set_file(open(os.path.join(save_folder, 'stderr.txt'), 'wt'))
    """
    def __init__(self, stream, only_output_to_file=False):
        """

        :param stream: usually sys.stdout/sys.stderr
        :param only_output_to_file: flag. Whether disable output to original stream, default False.
        """
        self.stream = stream
        self.buffer = ''
        self.file = None
        self.only_output_to_file = only_output_to_file
        pass

    def write(self, s):
        if not self.only_output_to_file or self.file is None:
            self.stream.write(s)
            self.stream.flush()
        if self.file is not None:
            self.file.write(s)
            self.file.flush()
        else:
            self.buffer = self.buffer + s

    def set_file(self, f):
        assert(self.file is None)
        self.file = f
        self.file.write(self.buffer)
        self.file.flush()
        self.buffer = ''

    def flush(self):
        if not self.only_output_to_file or self.file is None:
            self.stream.flush()
        if self.file is not None:
            self.file.flush()

    def close(self):
        self.stream.close()
        if self.file is not None:
            self.file.close()
            self.file = None



if __name__ == '__main__':
    INFO = ['This is a collection of auxiliary functions for DL.\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)

    # x1 = np.array([1,2,3])
    # x2 = np.array([1,2,3,4])
    # x3 = np.array([5,4,3,2,1])
    x1, x2, x3 = np.random.rand(2, 4), np.random.rand(3, 4), np.random.rand(5,4)

    Xout, mask = pad_sequence_into_array([x1,x2,x3])
    print(Xout)
    print(mask)
    print(Xout.shape)
    print(mask.shape)