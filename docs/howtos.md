# Tutorial III: Howtos

### 1) How to freeze a module during training?
To *freeze* a module during training, use the `include` and `exclude` arguments of module's `.collect_params()` and `.collect_self_updates()` functions.
#### Example
```python
class FOO(Module):
    def __init__(self):
        self.cnn0 = Conv2D(...)
        self.cnn1 = Conv2D(...)
        self.cnn2 = Conv2D(...)
        ....
        
# Now we will freeze cnn0 and cnn1 submodules during training
model    = Foo()
loss     = ...
params   = model.collect_params(exclude=['cnn0', 'cnn1'])
updates  = optimizer(loss, params)
updates.update(model.colect_self_updates(exclude=['cnn0', 'cnn1']))
train_fn = theano.function([...], [...], updates=updates, no_default_updates=False)
```

### 2) How to initialize a partially modified model with previouslly trained weights?
A frequently encountered scenario in research is that we want to re-use trained weights from a previous model to initialize a new model, usually partially modified. The most convenient way is to use `Module`'s `set_weights_by_name()` method with the `unmatched` argument set to `warn` or `ignore`. To use this method, it's assumed that you didn't change the variable's name to be initialized; otherwise, you can use the `name_map` argument to input the corresponding *weight->variable* mapping, or the most primitive way, use tensor's `get_value()` and `set_value()` methods explicitly.
#### Example
```python
from dandelion.util import gpickle
old_model_file = ...
old_module_weights, old_userdata = gpickle.load(old_model_file)
new_model = ...
new_model.set_weights_by_name(old_module_weights, unmatched='warn')
```

### 3) How to add random noise to a tensor?
Just use Theano's `MRG_RandomStreams` module.
#### Example
```python
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams(np.random.randint(1, 2147462579))
....
y = x + srng.normal(x.shape, avg=0.0, std=0.1)   # add Gaussian noise to x
```
What you'd keep in mind is that if you used Theano's `MRG_RandomStreams` module, remember to set `no_default_updates=False` when compiling functions.

### 4) How to do model-parallel training?
According to [issue 6655](https://github.com/Theano/Theano/issues/6655), model-parallel multi-GPU support of Theano will never be finished, so it won't be possible to do model-parallel training with Theano, and of course, Dandelion.  
For data-parallel training, refer to [platoon](https://github.com/mila-udem/platoon) for possible solution. We may implement our multi-GPU data-parallel training scheme later, stay tuned.