Dandelion's `activation` module is mostly inherited from [Lasagne](https://github.com/Lasagne/Lasagne) except for the `softmax()` activation. 

You're recommended to refer to [`Lasagne.nonlinearities` document](http://lasagne.readthedocs.io/en/latest/modules/nonlinearities.html) for the following activations:

* sigmoid
* tanh
* relu
* softplus
* ultra_fast_sigmoid
* ScaledTanH
* leaky_rectify
* very_leaky_rectify
* elu
* SELU
* linear
* identity

_______________________________________________________________________
## softmax
Apply softmax to the last dimension of input `x`
```python
softmax(x)
```
* **x**: theano tensor of any shape