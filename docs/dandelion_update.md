Dandelion's `update` module is mostly inherited from [Lasagne](https://github.com/Lasagne/Lasagne), you're recommended to refer to [`Lasagne.updates` document](http://lasagne.readthedocs.io/en/latest/modules/updates.html) for the following optimizers & helper functions:  

* apply_momentum
* momentum
* apply_nesterov_momentum
* nesterov_momentum
* adagrad
* rmsprop
* adamax
* norm_constraint
* total_norm_constrain

_______________________________________________________________________
## sgd
Stochastic gradient descent optimizer.

```python
sgd(loss_or_grads, params, learning_rate=1e-4, clear_nan=False)
```

* **loss_or_grads**: a scalar loss expression, or a list of gradient expressions
* **params**:  list of shared variables to generate update expressions for
* **learning_rate**: float or symbolic scalar, learning rate controlling the size of update steps
* **clear_nan**: boolean flag, if `True`, `nan` in gradients will be replaced with 0

_______________________________________________________________________
## adam
Adam optimizer implemented as described in *"Kingma, Diederik, and Jimmy Ba (2014): Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980."*

```python
adam(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
     beta2=0.999, epsilon=1e-8, clear_nan=False)
```

* **loss_or_grads**: a scalar loss expression, or a list of gradient expressions
* **params**:  list of shared variables to generate update expressions for
* **learning_rate**: float or symbolic scalar, learning rate controlling the size of update steps
* **clear_nan**: boolean flag, if `True`, `nan` in gradients will be replaced with 0
* **beta1**: float or symbolic scalar, exponential decay rate for the first moment estimates
* **beta2**: float or symbolic scalar, exponential decay rate for the second moment estimates
* **epsilon**: float or symbolic scalar, constant for numerical stability

_______________________________________________________________________
## adadelta
Adadelta optimizer implemented as described in *"Zeiler, M. D. (2012): ADADELTA: An Adaptive Learning Rate Method. arXiv Preprint arXiv:1212.5701."*

```python
adadelta(loss_or_grads, params, learning_rate=1.0, 
         rho=0.95, epsilon=1e-6, clear_nan=False)
```

* **loss_or_grads**: a scalar loss expression, or a list of gradient expressions
* **params**:  list of shared variables to generate update expressions for
* **learning_rate**: float or symbolic scalar, learning rate controlling the size of update steps
* **clear_nan**: boolean flag, if `True`, `nan` in gradients will be replaced with 0
* **rho**: float or symbolic scalar, squared gradient moving average decay factor
* **epsilon**: float or symbolic scalar, constant for numerical stability

`rho` should be between 0 and 1. A value of `rho` close to 1 will decay the moving average slowly and a value close to 0 will decay the moving average fast.  
`rho` = 0.95 and `epsilon`=1e-6 are suggested in the paper and reported to work for multiple datasets (MNIST, speech).  
In the paper, no learning rate is considered (so `learning_rate`=1.0). Probably best to keep it at this value. `epsilon` is important for the very first update (so the numerator does not become 0).




