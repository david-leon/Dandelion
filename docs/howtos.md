# Tutorial III: Howtos

### 1) How to freeze a module during training like in Keras/Lasagne?
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
