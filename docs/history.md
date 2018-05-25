# History

## version 0.15.1 [5-25-2018]
* **NEW**: add `model` module into master branch of Dandelion
* **NEW**: add U-net FCN implementation into `model` module
* **NEW**: add `align_crop()` into `functional` module

## version 0.14.4 [4-17-2018]
Rename `updates.py` with `update.py`

## version 0.14.0 [4-10-2018]
In this version the `Module`'s parameter interfaces are mostly redesigned, so it's **incompatible** with previous version.
Now `self.params` and `self.self_updating_variables` do not include sub-modules' parameters any more, to get all the parameters to be
trained by optimizer, including sub-modules' during training, you'll need to call the new interface function  `.collect_params()`. 
To collect self-defined updates for training, still call `.collect_self_updates()`.

* **MODIFIED**: `.get_weights()` and `.set_weights()` traverse the parameters in the same order of sub-modules, so they're **incompatible** with previous version.
* **MODIFIED**: Rewind all `trainable` flags, you're now expected to use the `include` and `exclude` arguments in `.collect_params()` and 
`.collect_self_updates()` to enable/disable training for certain module's parameters.
* **MODIFIED**: to define self-update expression for `self_updating_variable`, use `.update` attribute instead of previous `.default_update`
* **NEW**: add auto-naming feature to root class `Module`: if a sub-module is unnamed yet, it'll be auto-named by its instance name, 
from now on you don't need to name a sub-module manually any more.
* **NEW**: add `.set_weights_by_name()` to `Module` class, you can use this function to set module weights saved by previous version of Dandelion
