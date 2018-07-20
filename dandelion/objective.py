# coding:utf-8
'''
  Dandelion objectives pool
  Created   :   9, 20, 2017
  Revised   :   9, 20, 2017
'''

import theano
import theano.tensor as tensor
from .ctc_theano import CTC_Timescale, CTC_Logscale
from .util import as_theano_expression, one_hot

# __all__ = [
#     "categorical_crossentropy",
#     "ctc_cost_logscale",
#     "ctc_cost_timescale",
#     "ctc_best_path_decode",
#     "ctc_CER"
# ]

def align_targets(predictions, targets):
    """Helper function turning a target 1D vector into a column if needed.
    This way, combining a network of a single output unit with a target vector
    works as expected by most users, not broadcasting outputs against targets.

    Parameters
    ----------
    predictions : Theano tensor
        Expression for the predictions of a neural network.
    targets : Theano tensor
        Expression or variable for corresponding targets.

    Returns
    -------
    predictions : Theano tensor
        The predictions unchanged.
    targets : Theano tensor
        If `predictions` is a column vector and `targets` is a 1D vector,
        returns `targets` turned into a column vector. Otherwise, returns
        `targets` unchanged.
    """
    if (getattr(predictions, 'broadcastable', None) == (False, True) and
            getattr(targets, 'ndim', None) == 1):
        targets = as_theano_expression(targets).dimshuffle(0, 'x')
    return predictions, targets


def binary_crossentropy(predictions, targets):
    """Computes the binary cross-entropy between predictions and targets.

    .. math:: L = -t \\log(p) - (1 - t) \\log(1 - p)

    Parameters
    ----------
    predictions : Theano tensor
        Predictions in (0, 1), such as sigmoidal output of a neural network.
    targets : Theano tensor
        Targets in [0, 1], such as ground truth labels.

    Returns
    -------
    Theano tensor
        An expression for the element-wise binary cross-entropy.

    Notes
    -----
    This is the loss function of choice for binary classification problems
    and sigmoid output units.
    """
    predictions, targets = align_targets(predictions, targets)
    return theano.tensor.nnet.binary_crossentropy(predictions, targets)


def categorical_crossentropy(predictions, targets, eps=1e-7, m=None, class_weight=None):
    """
    Computes the categorical cross-entropy.
    :param predictions: usually returned by softmax(), 2D tensor with shape (B, N)
    :param targets: Theano 2D tensor or 1D tensor, with shape (B, N) or (B,)
    :param m: possible max value of `targets`'s element, required when `targets` is 1D tensor and `class_weight` is not None.
    :param class_weight: tensor vector with shape (Nclass,), for class weighting, optional.
    :return:
    """
    if eps > 0:
        predictions = theano.tensor.clip(predictions, eps, 1.0 - eps)

    if targets.ndim == predictions.ndim:
        if class_weight is None:
            return -tensor.sum(targets * tensor.log(predictions), axis=predictions.ndim - 1)
        else:
            return -tensor.sum(targets * tensor.log(predictions) * class_weight, axis=predictions.ndim - 1)
    elif targets.ndim == predictions.ndim - 1:
        if class_weight is None:
            return theano.tensor.nnet.crossentropy_categorical_1hot(predictions, targets)
        else:
            return -tensor.sum(one_hot(targets, m=m) * tensor.log(predictions) * class_weight, axis=predictions.ndim - 1)
    else:
        raise TypeError('shape mismatch between `predictions` and `targets`')

def categorical_crossentropy_log(log_predictions, targets, m=None, class_weight=None):
    """
    Computes the categorical cross-entropy in log-domain.
    :param log_predictions: usually returned by log_softmax()
    :param targets: Theano 2D tensor or 1D tensor
    :param m: possible max value of `targets`'s element, required when `targets` is 1D tensor.
    :param class_weight: tensor vector with shape (Nclass,), for class weighting, optional.
    :return:
    """

    if targets.ndim == log_predictions.ndim:
        if class_weight is None:
            return -tensor.sum(targets * log_predictions, axis=log_predictions.ndim-1)
        else:
            return -tensor.sum(targets * log_predictions * class_weight, axis=log_predictions.ndim-1)
    elif targets.ndim == log_predictions.ndim - 1:
        if class_weight is None:
            return -tensor.sum(one_hot(targets, m=m) * log_predictions, axis=log_predictions.ndim-1)
        else:
            return -tensor.sum(one_hot(targets, m=m) * log_predictions * class_weight, axis=log_predictions.ndim - 1)
    else:
        raise TypeError('shape mismatch between `log_predictions` and `targets`')

def squared_error(a, b):
    """Computes the element-wise squared difference between two tensors.

    .. math:: L = (p - t)^2

    Parameters
    ----------
    a, b : Theano tensor
        The tensors to compute the squared difference between.

    Returns
    -------
    Theano tensor
        An expression for the element-wise squared difference.

    Notes
    -----
    This is the loss function of choice for many regression problems
    or auto-encoders with linear output units.
    """
    a, b = align_targets(a, b)
    return theano.tensor.square(a - b)


def aggregate(loss, weights=None, mode='mean'):
    """Aggregates an element- or item-wise loss to a scalar loss.

    Parameters
    ----------
    loss : Theano tensor
        The loss expression to aggregate.
    weights : Theano tensor, optional
        The weights for each element or item, must be broadcastable to
        the same shape as `loss` if given. If omitted, all elements will
        be weighted the same.
    mode : {'mean', 'sum', 'normalized_sum'}
        Whether to aggregate by averaging, by summing or by summing and
        dividing by the total weights (which requires `weights` to be given).

    Returns
    -------
    Theano scalar
        A scalar loss expression suitable for differentiation.

    Notes
    -----
    By supplying binary weights (i.e., only using values 0 and 1), this
    function can also be used for masking out particular entries in the
    loss expression. Note that masked entries still need to be valid
    values, not-a-numbers (NaNs) will propagate through.

    When applied to batch-wise loss expressions, setting `mode` to
    ``'normalized_sum'`` ensures that the loss per batch is of a similar
    magnitude, independent of associated weights. However, it means that
    a given data point contributes more to the loss when it shares a batch
    with low-weighted or masked data points than with high-weighted ones.
    """
    if weights is not None:
        loss = loss * weights
    if mode == 'mean':
        return loss.mean()
    elif mode == 'sum':
        return loss.sum()
    elif mode == 'normalized_sum':
        if weights is None:
            raise ValueError("require weights for mode='normalized_sum'")
        return loss.sum() / weights.sum()
    else:
        raise ValueError("mode must be 'mean', 'sum' or 'normalized_sum', "
                         "got %r" % mode)


def binary_hinge_loss(predictions, targets, delta=1, log_odds=None,
                      binary=True):
    """Computes the binary hinge loss between predictions and targets.

    .. math:: L_i = \\max(0, \\delta - t_i p_i)

    Parameters
    ----------
    predictions : Theano tensor
        Predictions in (0, 1), such as sigmoidal output of a neural network
        (or log-odds of predictions depending on `log_odds`).
    targets : Theano tensor
        Targets in {0, 1} (or in {-1, 1} depending on `binary`), such as
        ground truth labels.
    delta : scalar, default 1
        The hinge loss margin
    log_odds : bool, default None
        ``False`` if predictions are sigmoid outputs in (0, 1), ``True`` if
        predictions are sigmoid inputs, or log-odds. If ``None``, will assume
        ``True``, but warn that the default will change to ``False``.
    binary : bool, default True
        ``True`` if targets are in {0, 1}, ``False`` if they are in {-1, 1}

    Returns
    -------
    Theano tensor
        An expression for the element-wise binary hinge loss

    Notes
    -----
    This is an alternative to the binary cross-entropy loss for binary
    classification problems.

    Note that it is a drop-in replacement only when giving ``log_odds=False``.
    Otherwise, it requires log-odds rather than sigmoid outputs. Be aware that
    depending on the Theano version, ``log_odds=False`` with a sigmoid
    output layer may be less stable than ``log_odds=True`` with a linear layer.
    """
    if log_odds is None:  # pragma: no cover
        log_odds = True
        raise FutureWarning(
                "The `log_odds` argument to `binary_hinge_loss` will change "
                "its default to `False` in a future version. Explicitly give "
                "`log_odds=True` to retain current behavior in your code, "
                "but also check the documentation if this is what you want.")
    if not log_odds:
        predictions = theano.tensor.log(predictions / (1 - predictions))
    if binary:
        targets = 2 * targets - 1
    predictions, targets = align_targets(predictions, targets)
    return theano.tensor.nnet.relu(delta - predictions * targets)


def multiclass_hinge_loss(predictions, targets, delta=1):
    """Computes the multi-class hinge loss between predictions and targets.

    .. math:: L_i = \\max_{j \\not = t_i} (0, p_j - p_{t_i} + \\delta)

    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either a vector of int giving the correct class index per data point
        or a 2D tensor of one-hot encoding of the correct class in the same
        layout as predictions (non-binary targets in [0, 1] do not work!)
    delta : scalar, default 1
        The hinge loss margin

    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise multi-class hinge loss

    Notes
    -----
    This is an alternative to the categorical cross-entropy loss for
    multi-class classification problems
    """
    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between targets and predictions')
    corrects = predictions[targets.nonzero()]
    rest = theano.tensor.reshape(predictions[(1-targets).nonzero()],
                                 (-1, num_cls-1))
    rest = theano.tensor.max(rest, axis=1)
    return theano.tensor.nnet.relu(rest - corrects + delta)


def binary_accuracy(predictions, targets, threshold=0.5):
    """Computes the binary accuracy between predictions and targets.

    .. math:: L_i = \\mathbb{I}(t_i = \mathbb{I}(p_i \\ge \\alpha))

    Parameters
    ----------
    predictions : Theano tensor
        Predictions in [0, 1], such as a sigmoidal output of a neural network,
        giving the probability of the positive class
    targets : Theano tensor
        Targets in {0, 1}, such as ground truth labels.
    threshold : scalar, default: 0.5
        Specifies at what threshold to consider the predictions being of the
        positive class

    Returns
    -------
    Theano tensor
        An expression for the element-wise binary accuracy in {0, 1}

    Notes
    -----
    This objective function should not be used with a gradient calculation;
    its gradient is zero everywhere. It is intended as a convenience for
    validation and testing, not training.

    To obtain the average accuracy, call :func:`theano.tensor.mean()` on the
    result, passing ``dtype=theano.config.floatX`` to compute the mean on GPU.
    """
    predictions, targets = align_targets(predictions, targets)
    predictions = theano.tensor.ge(predictions, threshold)
    return theano.tensor.eq(predictions, targets)


def categorical_accuracy(predictions, targets, top_k=1):
    """Computes the categorical accuracy between predictions and targets.

    .. math:: L_i = \\mathbb{I}(t_i = \\operatorname{argmax}_c p_{i,c})

    Can be relaxed to allow matches among the top :math:`k` predictions:

    .. math::
        L_i = \\mathbb{I}(t_i \\in \\operatorname{argsort}_c (-p_{i,c})_{:k})

    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either a vector of int giving the correct class index per data point
        or a 2D tensor of 1 hot encoding of the correct class in the same
        layout as predictions
    top_k : int
        Regard a prediction to be correct if the target class is among the
        `top_k` largest class probabilities. For the default value of 1, a
        prediction is correct only if the target class is the most probable.

    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise categorical accuracy in {0, 1}

    Notes
    -----
    This is a strictly non differential function as it includes an argmax.
    This objective function should never be used with a gradient calculation.
    It is intended as a convenience for validation and testing not training.

    To obtain the average accuracy, call :func:`theano.tensor.mean()` on the
    result, passing ``dtype=theano.config.floatX`` to compute the mean on GPU.
    """
    if targets.ndim == predictions.ndim:
        targets = theano.tensor.argmax(targets, axis=-1)
    elif targets.ndim != predictions.ndim - 1:
        raise TypeError('rank mismatch between targets and predictions')

    if top_k == 1:
        # standard categorical accuracy
        top = theano.tensor.argmax(predictions, axis=-1)
        return theano.tensor.eq(top, targets)
    else:
        # top-k accuracy
        top = theano.tensor.argsort(predictions, axis=-1)
        # (Theano cannot index with [..., -top_k:], we need to simulate that)
        top = top[[slice(None) for _ in range(top.ndim - 1)] +
                  [slice(-top_k, None)]]
        targets = theano.tensor.shape_padaxis(targets, axis=-1)
        return theano.tensor.any(theano.tensor.eq(top, targets), axis=-1)


def ctc_cost_timescale(seq, sm, seq_mask=None, sm_mask=None, blank_symbol=None):
    """
    seq (B, L), sm (B, T, C+1), seq_mask (B, L), sm_mask (B, T)
    Compute CTC cost, using only the forward pass
    :param seq: query sequence, (L, B), float32
    :param sm: score matrix, (T, C+1, B), float32
    :param seq_mask: mask for query sequence, (L, B), float32
    :param sm_mask: mask for score matrix, (T, B), float32
    :param blank_symbol: scalar
    :return: negative log likelihood averaged over a batch
    """
    queryseq = seq.T
    scorematrix = sm.dimshuffle(1, 2, 0)
    if seq_mask is None:
        queryseq_mask = None
    else:
        queryseq_mask = seq_mask.T
    if sm_mask is None:
        scorematrix_mask = None
    else:
        scorematrix_mask = sm_mask.T

    return CTC_Timescale.cost(queryseq, scorematrix, queryseq_mask, scorematrix_mask, blank_symbol)

def ctc_cost_logscale(seq, sm, seq_mask=None, sm_mask=None, blank_symbol=None, align='pre'):
    """
    Compute CTC cost, using only the forward pass
    :param seq: query sequence, (L, B), float32
    :param sm: score matrix, (T, C+1, B), float32
    :param seq_mask: mask for query sequence, (L, B), float32
    :param sm_mask: mask for score matrix, (T, B), float32
    :param blank_symbol: scalar
    :return: negative log likelihood averaged over a batch
    """
    queryseq = tensor.addbroadcast(seq.T)
    scorematrix = sm.dimshuffle(1, 2, 0)
    if seq_mask is None:
        queryseq_mask = None
    else:
        queryseq_mask = seq_mask.T
    if sm_mask is None:
        scorematrix_mask = None
    else:
        scorematrix_mask = sm_mask.T

    return CTC_Logscale.cost(queryseq, scorematrix, queryseq_mask, scorematrix_mask, blank_symbol, align)

def ctc_best_path_decode(Y, Y_mask=None, blank_symbol=None):
    """
    Decode the network output scorematrix by best-path-decoding scheme
    :param Y: output of a network, with shape (batch, timesteps, Nclass+1)
    :param Y_mask: mask of Y, with shape (batch, timesteps)
    :return:
    """
    scorematrix = Y.dimshuffle(1, 2, 0)
    if Y_mask is None:
        scorematrix_mask = None
    else:
        scorematrix_mask = Y_mask.dimshuffle(1, 0)
    if blank_symbol is None:
        blank_symbol = Y.shape[2] - 1
    resultseq, resultseq_mask = CTC_Logscale.best_path_decode(scorematrix, scorematrix_mask, blank_symbol)
    return resultseq, resultseq_mask

def ctc_CER(resultseq, targetseq, resultseq_mask=None, targetseq_mask=None):
    return CTC_Logscale.calc_CER(resultseq, targetseq, resultseq_mask, targetseq_mask)
