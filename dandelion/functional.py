# coding:utf-8
'''
Dandelion functional pool
Created   :   2, 27, 2018
Revised   :   5, 24, 2018  add `align_crop` which is the same with Lasagne's autocrop()
              8,  3, 2018  move `channel_shuffle()` from `model.shufflenet.py` into `functional.py`
              10, 4, 2018  feature improvement in `spatial_pyramid_pooling` (quanqiu.nick@foxmail.com)
All rights reserved
'''
__author__ = 'dawei.leng'
import theano
floatX = theano.config.floatX
from theano import tensor
from theano.tensor.signal import pool
from .util import as_tuple
from theano.tensor.nnet.neighbours import images2neibs

def pool_1d(x, ws=2, ignore_border=True, stride=None, pad=0, mode='max', axis=-1):
    """
    Pooling 1 dimension along the given axis, support for any dimensional input
    :param x: input to be pooled, can have any dimension
    :param ws:
    :param ignore_border:
    :param stride: if None, same with ws
    :param pad:
    :param mode:
    :param axis: default the last dimension
    :return:
    """
    if stride is not None:
        stride = (stride, 1)
    ndim = x.ndim
    if axis < 0:
        axis += ndim
    pattern = []
    pattern_reverse = list(range(ndim))
    for i in range(ndim):
        if i != axis:
            pattern.append(i)
            if i != len(pattern)-1:
                pattern_reverse[i] = len(pattern)-1
    pattern.extend([axis, 'x'])
    pattern_reverse[axis] = ndim-1
    x = x.dimshuffle(pattern)
    x = pool.pool_2d(x, ws=(ws, 1), ignore_border=ignore_border, stride=stride, pad=(pad,0), mode=mode)[..., 0]
    x = x.dimshuffle(pattern_reverse)
    return x

def pool_2d(x, ws=(2,2), ignore_border=True, stride=None, pad=(0, 0), mode='max'):
    """
    Pooling 2 dimension along the last 2 dimensions of input, support for any dimensional input with ndim>=2
    :param x:
    :param ws:
    :param ignore_border:
    :param stride:
    :param pad:
    :param mode:
    :return:
    """
    return pool.pool_2d(x, ws=ws, ignore_border=ignore_border, stride=stride, pad=pad, mode=mode)

def pool_3d(x, ws=(2,2,2), ignore_border=True, stride=None, pad=(0, 0, 0), mode='max'):
    """
    Pooling 3 dimension along the last 3 dimensions of input, support for any dimensional input with ndim>=3.
    :param x:
    :param ws:
    :param ignore_border:
    :param stride:
    :param pad:
    :param mode:
    :return:
    """
    return pool.pool_3d(x, ws=ws, ignore_border=ignore_border, stride=stride, pad=pad, mode=mode)

def align_crop(tensor_list, cropping):
    """
    align and crop the input tensors by specified rules. Basically the same with Lasagne's autocrop().

    Cropping takes a sequence of tensor_list and crops them per-axis in order to
    ensure that their sizes are consistent so that they can be combined
    in an element-wise fashion. If cropping is enabled for a specific axis,
    the minimum size in that axis of all tensor_list is computed, and all
    tensor_list are cropped to that size.

    The per-axis cropping modes are:

    `None`: this axis is not cropped, tensor_list are unchanged in this axis

    `'lower'`: tensor_list are cropped choosing the lower portion in this axis
    (`a[:crop_size, ...]`)

    `'upper'`: tensor_list are cropped choosing the upper portion in this axis
    (`a[-crop_size:, ...]`)

    `'center'`: tensor_list are cropped choosing the central portion in this axis
    (``a[offset:offset+crop_size, ...]`` where
    ``offset = (a.shape[0]-crop_size)//2)``

    Parameters
    ----------
    tensor_list : list of Theano expressions
        The input arrays in the form of a list of Theano expressions

    cropping : list of cropping modes
        Cropping modes, one for each axis. If length of `cropping` is less
        than the number of axes in the tensor_list, it is padded with `None`.
        If `cropping` is None, `input` is returned as is.

    Returns
    -------
    list of Theano expressions
        each expression is the cropped version of the corresponding input

    Example
    -------
    For example, given three tensor_list:

    >>> import numpy
    >>> import theano

    >>> a = numpy.random.random((1, 2, 3, 4))
    >>> b = numpy.random.random((5, 4, 4, 2))
    >>> c = numpy.random.random((7, 1, 8, 9))

    Cropping mode for each axis:

    >>> cropping = [None, 'lower', 'center', 'upper']

    Crop (note that the input arrays are converted to Theano vars first,
    and that the results are converted back from Theano expressions to
    numpy arrays by calling `eval()`)
    >>> xa, xb, xc = autocrop([theano.shared(a), \
                               theano.shared(b), \
                               theano.shared(c)], cropping)
    >>> xa, xb, xc = xa.eval(), xb.eval(), xc.eval()

    They will be left as is in axis 0 and cropped in the other three,
    choosing the lower, center and upper portions:

    Axis 0: choose all, axis 1: lower 1 element,
    axis 2: central 3 (all) and axis 3: upper 2
    >>> (xa == a[:, :1, :3, -2:]).all()
    True

    Axis 0: choose all, axis 1: lower 1 element,
    axis 2: central 3 starting at 0 and axis 3: upper 2 (all)
    >>> (xb == b[:, :1, :3, -2:]).all()
    True

    Axis 0: all, axis 1: lower 1 element (all),
    axis 2: central 3 starting at 2 and axis 3: upper 2
    >>> (xc == c[:, :1, 2:5:, -2:]).all()
    True
    """
    if cropping is None:
        # No cropping in any dimension
        return tensor_list
    else:
        # Get the number of dimensions
        ndim = tensor_list[0].ndim
        # Check for consistent number of dimensions
        if not all(input.ndim == ndim for input in tensor_list):
            raise ValueError("Not all tensor_list are of the same "
                             "dimensionality. Got {0} tensor_list of "
                             "dimensionalities {1}.".format(
                                len(tensor_list),
                                [input.ndim for input in tensor_list]))
        # Get the shape of each input, where each shape will be a Theano
        # expression
        shapes = [input.shape for input in tensor_list]
        # Convert the shapes to a matrix expression
        shapes_tensor = tensor.as_tensor_variable(shapes)
        # Min along axis 0 to get the minimum size in each dimension
        min_shape = tensor.min(shapes_tensor, axis=0)

        # Nested list of slices; each list in `slices` corresponds to
        # an input and contains a slice for each dimension
        slices_by_input = [[] for i in range(len(tensor_list))]

        # If there are more dimensions than cropping entries, pad the cropping rules with None
        cropping = list(cropping)
        if ndim > len(cropping):
            cropping = list(cropping) + [None] * (ndim - len(cropping))

        # For each dimension
        for dim, cr in enumerate(cropping):
            if cr is None:
                # Don't crop this dimension
                slice_all = slice(None)
                for slices in slices_by_input:
                    slices.append(slice_all)
            else:
                # We crop all tensor_list in the dimension `dim` so that they
                # are the minimum found in this dimension from all tensor_list
                sz = min_shape[dim]
                if cr == 'lower':
                    # Choose the first `sz` elements
                    slc_lower = slice(None, sz)
                    for slices in slices_by_input:
                        slices.append(slc_lower)
                elif cr == 'upper':
                    # Choose the last `sz` elements
                    slc_upper = slice(-sz, None)
                    for slices in slices_by_input:
                        slices.append(slc_upper)
                elif cr == 'center':
                    # Choose `sz` elements from the center
                    for sh, slices in zip(shapes, slices_by_input):
                        offset = (sh[dim] - sz) // 2
                        slices.append(slice(offset, offset+sz))
                else:
                    raise ValueError(
                        'Unknown crop mode \'{0}\''.format(cr))

        return [input[slices] for input, slices in
                zip(tensor_list, slices_by_input)]


def spp_pad_zero(section, pyramid_dim, axis=2, dtype='float32'):
    """
    For spatial_pyramid_pooling method.
    """
    delta = pyramid_dim - section.shape[axis]
    if 2 == axis:
        p = tensor.zeros(shape=(section.shape[0], section.shape[1], delta, section.shape[3]), dtype=dtype)
    elif 3 == axis:
        p = tensor.zeros(shape=(section.shape[0], section.shape[1], section.shape[2], delta), dtype=dtype)
    return tensor.concatenate([section, p], axis=axis)


def spatial_pyramid_pooling(x, pyramid_dims=(6, 4, 2, 1), mode='max', implementation='fast'):
    """
    Spatial pyramid pooling. Refer to: He, Kaiming et al (2015), Spatial Pyramid Pooling in Deep Convolutional Networks
    for Visual Recognition. http://arxiv.org/pdf/1406.4729.pdf and Lasagne's SpatialPyramidPoolingLayer implementation.
    This function will generate spatially fix-sized output no matter the spatial size of input, useful when CNN+FC used for
    image classification or detection.
    :param x: 4D tensor, (B, C, H, W)
    :param pyramid_dims: list of pyramid dims,
    :param mode:
    :param implementation : string
        'fast' | 'fast_ls' | 'stretch'.
        The 'fast' implementation is fast and pad zero when input size is too small.
        The 'fast_ls' implementation is same as Lasagne fast implementation. The size of the input map MUST be larger
            than the output map size.
        The 'stretch' implementation is slower. The implementation will get same feature at some position just like
            nearest neighbor interpolation when the input size is less than the output size.
    :return:
    """
    from theano.ifelse import ifelse
    input_size = x.shape[2:]
    section_list = []

    if 'fast' == implementation:
        for pyramid_dim in pyramid_dims:
            win_size = tuple((i + pyramid_dim - 1) // pyramid_dim for i in input_size)
            str_size = tuple((i + pyramid_dim - 1) // pyramid_dim for i in input_size)
            section = pool.pool_2d(x,
                           ws=win_size,
                           stride=str_size,
                           mode=mode,
                           pad=None,
                           ignore_border=False)

            section = ifelse(tensor.lt(section.shape[2], pyramid_dim),
                             spp_pad_zero(section, pyramid_dim, axis=2, dtype=x.dtype), section)
            section = ifelse(tensor.lt(section.shape[3], pyramid_dim),
                             spp_pad_zero(section, pyramid_dim, axis=3, dtype=x.dtype), section)

            section = section.flatten(3)
            section_list.append(section)

    elif 'fast_ls' == implementation:
        for pyramid_dim in pyramid_dims:
            win_size = tuple((i + pyramid_dim - 1) // pyramid_dim for i in input_size)
            str_size = tuple(i // pyramid_dim for i in input_size)
            section = pool.pool_2d(x,
                           ws=win_size,
                           stride=str_size,
                           mode=mode,
                           pad=None,
                           ignore_border=True)
            section = section.flatten(3)
            section_list.append(section)

    elif 'stretch' == implementation:
        for pyramid_dim in pyramid_dims:
            h, w = input_size
            h = h.astype('float32')
            w = w.astype('float32')
            n = float(pyramid_dim)
            for row in range(pyramid_dim):
                for col in range(pyramid_dim):
                    row = float(row)
                    col = float(col)
                    start_h = tensor.floor(row / n * h).astype('int32')
                    end_h = tensor.ceil((row + 1.0) / n * h).astype('int32')
                    start_w = tensor.floor(col / n * w).astype('int32')
                    end_w = tensor.ceil((col + 1.0) / n * w).astype('int32')

                    pooling_region = x[:, :, start_h:end_h, start_w:end_w]

                    win_size = (end_h-start_h, end_w-start_w)
                    str_size = (end_h-start_h, end_w-start_w)
                    section = pool.pool_2d(pooling_region,
                                           ws=win_size,
                                           stride=str_size,
                                           mode=mode,
                                           pad=None,
                                           ignore_border=False)
                    section = section.flatten(3)
                    section_list.append(section)

    return tensor.concatenate(section_list, axis=2)


def upsample_2d(x, ratio, mode='repeat'):
    """
    Upsample 2 dimension along the last 2 dimensions of input, support for any dimensional input with ndim>=2.
    :param x:
    :param ratio: must be integer or tuple of integers >=1
    :param mode: 'repeat' | 'dilate'. Repeat element values or upsample leaving zeroes between upsampled elements
    :return:
    """
    a, b = as_tuple(ratio, 2)
    upscaled = x
    if mode == 'repeat':
        if b > 1:
            upscaled = tensor.extra_ops.repeat(upscaled, b, axis=-1)
        if a > 1:
            upscaled = tensor.extra_ops.repeat(upscaled, a, axis=-2)
    elif mode == 'dilate':
        if b > 1 or a > 1:
            output_shape = list(x.shape)  # copy / convert to mutable list
            if output_shape[-2] is not None:
                output_shape[-2] *= a
            if output_shape[-1] is not None:
                output_shape[-1] *= b
            upscaled = tensor.zeros(shape=tuple(output_shape), dtype=x.dtype)
            upscaled = tensor.set_subtensor(upscaled[:, :, ::a, ::b], x)
    else:
        raise ValueError('mode = %s not supported' % mode)
    return upscaled

def upsample_2d_bilinear(x, ratio=None, frac_ratio=None, use_1D_kernel=True):
    """
    Upsample 2D with bilinear interpolation. Support for fractional ratio, and only apply for 4D tensor.
    :param x: 4D tensor.
    :param ratio: int or tuple of int >=1. You can only specify either `ratio` or `frac_ratio`, not both.
    :param frac_ratio: None, tuple of int or tuple of tuples of int. A fractional upsampling scale is described by
                       (numerator, denominator).
    :param use_1D_kernel: only for speed matter
    :return:
    """
    return tensor.nnet.abstract_conv.bilinear_upsampling(x, ratio=ratio, frac_ratio=frac_ratio, use_1D_kernel=use_1D_kernel)

def channel_shuffle(x, group_num):
    """
    Pseudo shuffle channel by dimshuffle & reshape, first introduced in [ShuffleNet](https://arxiv.org/abs/1610.02357)
    :param x: (B, C, H, W)
    :param group_num: int scalar, C must be divisible by group_num
    :return:
    """
    if group_num == 1:
        return x
    B, C, H, W = x.shape
    x = tensor.reshape(x, (B, group_num, C//group_num, H, W))
    x = x.dimshuffle(0, 2, 1, 3, 4)
    x = tensor.reshape(x, (B, -1, H, W))
    return x

#todo: due to theano's images2neibs(), this tensor function does not support gradient computation [7-30-2018]
#todo: this API won't be exposed until we find another way to implement `im2col` with gradient support
#todo: if this API is deleted, test/test_im2col.py should be deleted as well
def _im2col(x, nb_size=(3,3), border_mode='half', merge_channel=False):
    """
    Function im2col allows to apply a sliding window operation to a 4D tensor, based on theano's images2neibs() and corresponding to Caffe's im2col()
    :todo: support any nb_step
    :param x: must be 4D tensor with shape (B, C, H, W)
    :param nb_size: neighbour size (h, w), i.e., the sliding window size, 2D vector. Only odd shapes are supported.
    :param border_mode: 'half' | 'wrap_centered' : on image border, whether pad by 0 or wrap image
    :param merge_channel: if False, return with shape (B, C, H, W, h*w), otherwise return with shape (B, H, W, C*h*w), corresponding to Caffe's im2col()
    :return:
    """
    nb_step = (1, 1)
    B, C, H, W = x.shape
    x = images2neibs(x, neib_shape=nb_size, neib_step=nb_step, mode=border_mode) # (B*C*H*W, h*w)
    x = tensor.reshape(x, (B, C, H, W, -1))  # (B, C, H, W, h*w)
    if merge_channel:
        x = x.dimshuffle(0, 2, 3, 1, 4)  # (B, H, W, C, h*w)
        x = tensor.reshape(x, (B, H, W, -1))  # (B, H, W, C*h*w)
    return x



if __name__ == '__main__':
    INFO = ['Dandelion framework: module pool\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)



