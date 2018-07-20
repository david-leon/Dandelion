## ctc_cost_logscale
CTC cost calculated in `log` scale. This CTC objective is written purely in Theano, so it runs on both Windows and Linux. Theano itself also has a [wrapper](http://deeplearning.net/software/theano/library/tensor/nnet/ctc.html) for Baidu's `warp-ctc` library, which requires separate install and only runs on Linux.
```python
ctc_cost_logscale(seq, sm, seq_mask=None, sm_mask=None, blank_symbol=None, align='pre')
```
* **seq**: query sequence, shape of `(L, B)`, `float32`-typed
* **sm**: score matrix, shape of `(T, C+1, B)`, `float32`-typed
* **seq_mask**: mask for query sequence, shape of `(L, B)`, `float32`-typed
* **sm_mask**: mask for score matrix, shape of `(T, B)`, `float32`-typed
* **blank_symbol**: scalar, = `C` by default
* **align**: string, {'pre'/'post'}, indicating how input samples are aligned in one batch
* **return**: negative log likelihood averaged over a batch

_______________________________________________________________________
## ctc_best_path_decode
Decode the network output scorematrix by best-path-decoding scheme.
```python
ctc_best_path_decode(Y, Y_mask=None, blank_symbol=None)
```
* **Y**: output of a network, with shape `(B, T, C+1)`
* **Y_mask**: mask of Y, with shape `(B, T)`
* **return**: result sequence of shape `(T, B`), and result sequence mask of shape `(T, B)`

_______________________________________________________________________
## ctc_CER
Calculate the character error rate (CER) given ground truth `targetseq` and CTC decoding output `resultseq`
```python
ctc_CER(resultseq, targetseq, resultseq_mask=None, targetseq_mask=None)
```
* **resultseq**: CTC decoding output, with shape `(T1, B)`
* **targetseq**: sequence ground truth, with shape `(T2, B)`
* **return**: tuple of `(CER, TE, TG)`, in which `TE` is the batch-wise total edit distance, `TG` is the batch-wise total ground truth sequence length, and `CER` equals to `TE/TG`

_______________________________________________________________________
## categorical_crossentropy
Computes the categorical cross-entropy between predictions and targets
```python
categorical_crossentropy(predictions, targets, eps=1e-7, m=None, class_weight=None)
```
* **predictions**: Theano 2D tensor, predictions in (0, 1), such as softmax output of a neural network, with data points in rows and class probabilities in columns.
* **targets**: Theano 2D tensor or 1D tensor, either targets in [0, 1] (float32 type) matching the layout of `predictions`, or a vector of int giving the correct class index per data point. In the case of an integer vector argument, each element represents the position of the '1' in a one-hot encoding.
* **eps**: epsilon added to `predictions` to prevent numerical unstability when using with softmax activation
* **m**: possible max value of `targets`'s element, required when `targets` is 1D tensor and `class_weight` is not None.
* **class_weight**: tensor vector with shape (Nclass,), used for class weighting, optional.
* **return**: Theano 1D tensor, an expression for the item-wise categorical cross-entropy.

_______________________________________________________________________
## categorical_crossentropy_log
Computes the categorical cross-entropy between predictions and targets, in log-domain.
```python
categorical_crossentropy_log(log_predictions, targets, m=None, class_weight=None)
```
* **log_predictions**: Theano 2D tensor, predictions in log of (0, 1), such as log_softmax output of a neural network, with data points in rows and class probabilities in columns.
* **targets**: Theano 2D tensor or 1D tensor, either targets in [0, 1] (float32 type) matching the layout of `predictions`, or a vector of int giving the correct class index per data point. In the case of an integer vector argument, each element represents the position of the '1' in a one-hot encoding.
* **m**: possible max value of `targets`'s element, only used when `targets` is 1D vector. When `targets` is integer vector, the implementation of `categorical_crossentropy_log` is different from `categorical_crossentropy`: the latter relies on `theano.tensor.nnet.categorical_crossentropy` whereas the former uses a simpler way, we transform the integer vector `targets` into one-hot encoded matrix. That's why we need the `m` argument here. The possible limitation is that our implementation does not allow `m` changing on-the-fly.
* **class_weight**: tensor vector with shape (Nclass,), used for class weighting, optional.
* **return**: Theano 1D tensor, an expression for the item-wise categorical cross-entropy in log-domain

_______________________________________________________________________

You're recommended to refer to [`Lasagne.objectives` document](http://lasagne.readthedocs.io/en/latest/modules/objectives.html) for the following objectives:

* binary_crossentropy
* squared_error
* binary_hinge_loss
* multiclass_hinge_loss
* binary_accuracy
* categorical_accuracy