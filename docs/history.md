# History

## version 0.16.10 [7-10-2018]
* **MODIFIED**: disable all install requirements to prevent possible conflict of pip and conda channel.

## version 0.16.9 [7-10-2018]
* **MODIFIED**: import all model reference implementations into `dandelion.model`'s namespace
* **FIXED**: `ConvTransposed2D`'s `W_shape` should use `in_channels` as first dimension; incorrect `W_shape` when `num_groups` > 1.

## version 0.16.8 [7-9-2018]
* **NEW**: add `model::shufflenet::DSConv2D` for Depthwise Separable Convolution reference implementation.
* **NEW**: add `model::shufflenet::ShuffleUnit` for ShuffleNet reference implementation
* **FIXED**: `W_shape` of `module::Conv2D` should count for `num_groups`

## version 0.16.7 [7-6-2018]
* **NEW**: add `model::vgg::model_VGG16` for VGG-16 reference implementation.
* **NEW**: add `model::resnet::ResNet_bottleneck` for ResNet reference implementation
* **NEW**: add `model::feature_pyramid_net::model_FPN` for Feature Pyramid Network reference implementation

## version 0.16.6 [7-5-2018]
* **NEW**: add `functional::upsample_2d()` for 2D upsampling
* **NEW**: add `functional::upsample_2d_bilinear()`for bilinear 2D upsampling

## version 0.16.5 [7-5-2018]
* **NEW**: add `functional::spatial_pyramid_pooling()` for SPP-net implementation.

## version 0.16.4 [7-4-2018]
* **FIXED**: wrong indexing when `targets` is int vector for `objective::categorical_crossentropy_log()`.


## version 0.16.3 [7-3-2018]
* **NEW**: add `activation::log_softmax()` for more numerically stable softmax.
* **NEW**: add `objective::categorical_crossentropy_log()` for more numerically stable categorical cross-entropy
* **MODIFIED**: add `eps` argument to `objective::categorical_crossentropy()` for numerical stability purpose. Note 1e-7 is set as default value of `eps`. You can set it to 0 to get the old `categorical_crossentropy()` back.

## version 0.16.0 [6-13-2018]
* **NEW**: add `ext` module into master branch of Dandelion. All the miscellaneous extensions will be organized in here.
* **NEW**: add `ext.CV` sub-module, containing image I/O functions and basic image processing functions commonly used in model training.

## version 0.15.2 [5-28-2018]
* **FIXED**: `convTOP` should be constructed each time the `forward()` function of `ConvTransposed2D` is called.

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
