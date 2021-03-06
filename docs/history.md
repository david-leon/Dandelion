# History

## version 0.17.25 [7-19-2019]
* **NEW**: Add `only_output_to_file` arg to `util.sys_output_tap` class. You can disable screen output by set this flag to `True`, this will make the script run *silently*.

## version 0.17.24 [4-3-2019]
* **NEW**: Add [`Anti-996 License`](https://github.com/996icu/996.ICU/blob/master/LICENSE) as auxiliary license

## version 0.17.23 [3-5-2019]
* **NEW**: now `BatchNorm`'s `mean` can be set to `None` to disable mean substraction

## version 0.17.22 [2-28-2019]
* **FIXED**: `self.b` should be set to `None` if not specified for `Conv2D` module

## version 0.17.21 [2-13-2019]
* **NEW**: now `BatchNorm`'s `inv_std` can be set to `None` to disable variance scaling

## version 0.17.20 [2-11-2019]
* **FIXED**: `self.b` should be set to `None` if not specified for `Dense` module

## version 0.17.19 [1-23-2019]
* **NEW**: add `clear_nan` argument for `sgd`, `adam` and `adadelta` optimizers.
* **MODIFIED**: add default value 1e-4 for `sgd`'s `learning_rate` arg.

## version 0.17.17 [11-22-2018]
* **NEW**: add `GroupNorm` module for group normalization implementation;
* **MODIFIED**: expose `dim_broadcast` arg for `Module.register_param()` method;
* **MODIFIED**: replace `spec = tensor.patternbroadcast(spec, dim_broadcast)` with `spec = theano.shared(spec, broadcastable=dim_broadcast)` for `util.create_param()`, due to the former would change tensor's type.

## version 0.17.16 [11-19-2018]
* **FIXED**: wrong scale of `model_size` for `ext.visual.get_model_summary()`;
* **MODIFIED**: add `stride` arg for `ShuffleUnit_Stack` and `ShuffleUnit_v2_Stack`; add `fusion_mode` arg for `ShuffleUnit_Stack`; improve their documentation.

## version 0.17.15 [11-16-2018]
* **NEW**: add `ext.visual` sub-module, containing model summarization & visualization toolkits.

## version 0.17.14 [11-15-2018]
* **FIXED**: remove redundant `bn5` layer of `ShuffleUnit_v2` when `stride` = 1 and `batchnorm_mode` = 2.

## version 0.17.13 [11-13-2018]
* **NEW**: add `model.Alternate_2D_LSTM` module for 2D LSTM implementaton by alternating 1D LSTM along different input dimensions.

## version 0.17.12 [11-6-2018]
* **NEW**: add `LSTM2D` module for 2D LSTM implementation
* **NEW**: add `.todevice()` interface to `Module` class for possible support of model-parallel multi-GPU training. However due to [Theano issue 6655](https://github.com/Theano/Theano/issues/6655), this feature won't be finished, so use it at your own risk.
* **MODIFIED**: `activation` param of `Sequential` class now supports list input.
* **MODIFIED**:  merge pull request [#1](https://github.com/david-leon/Dandelion/pull/1), [#2](https://github.com/david-leon/Dandelion/pull/2), now `functional.spatial_pyramid_pooling()` supports 3 different implementations.

## version 0.17.11 [9-3-2018]
* **MODIFIED**: returned `bbox`'s shape is changed to (B, H, W, k, n) for `model_CTPN`

## version 0.17.10 [8-16-2018]
* **MODIFIED**: add `border_mode` arg to `dandelion.ext.CV.imrotate()`, the `interpolation` arg type is changed to string.
* **MODIFIED**: returned `bbox`'s shape is changed to (B, H, W, n*k) for `model_CTPN`

## version 0.17.9 [8-7-2018]
* **NEW**: add `theano_safe_run, Finite_Memory_Array, pad_sequence_into_array, get_local_ip, get_time_stamp` into `dandelion.util`
* **NEW**: add documentation of `dandelion.util`, unfinished.

## version 0.17.8 [8-3-2018]
* **MODIFIED**: disable the auto-broadcasting in `create_param()`. This auto-broadcasting would result in theano exception for parameter with shape = [1].
* **NEW**: add `model.shufflenet.ShuffleUnit_v2_Stack` and `model.shufflenet.ShuffleNet_v2` for ShuffleNet_v2 reference implementation.
* **NEW**: move `channel_shuffle()` from `model.shufflenet.py` into `functional.py`

## version 0.17.7 [8-2-2018]
From this version the documentaiton supports latex math officially.  
* **MODIFIED**: move arg `alpha` of `Module.Center` from class delcaration to its `.forward()` interface.

## version 0.17.6 [7-25-2018]
* **MODIFIED**: change default value of `Module.set_weights_by_name()`'s arg `unmatched` from `ignore` to `raise`
* **MODIFIED**: change default value of `model.vgg.model_VGG16()`'s arg `flip_filters` from `True` to `False`

## version 0.17.5 [7-20-2018]
* **FIXED**: fixed typo in `objective.categorical_crossentropy()`

## version 0.17.4 [7-20-2018]
* **NEW**: add class weighting support for `objective.categorical_crossentropy()` and `objective.categorical_crossentropy_log()`
* **NEW**: add `util.theano_safe_run()` to help catch memory exceptions when running theano functions.

## version 0.17.3 [7-18-2018]
* **FIXED**: pooling mode in `model.shufflenet.ShuffleUnit` changed to `average_inc_pad` for correct gradient.

## version 0.17.2 [7-17-2018]
* **NEW**: add `model.shufflenet.model_ShuffleSeg` for Shuffle-Seg model reference implementation.

## version 0.17.1 [7-12-2018]
* **MODIFIED**: modify all `Test/Test_*.py` to be compatible with pytest. 
* **NEW**: add Travis CI for automatic unit test.

## version 0.17.0 [7-12-2018]
In this version the `Module`'s parameter and sub-module naming conventions are changed to make sure unique name for each variable/module in a complex network.  
It's **incompatible** with previous version if your work accessed their names, otherwise there is no impact.  
Note: to set weights saved by previous dandelion(>=version 0.14.0), use `.set_weights()` instead of `.set_weights_by_name()`. For weights saved by dandelion of version < 0.14.0, the quick way is to set the model's submodule weight explicitly as `model_new_dandelion.conv1.W.set_value(model_old_dandelion.conv1.W.get_value())`.    
From this version, it's recommonded to let the framework auto-name the module parameters when you define your own module with `register_param()` and `register_self_updating_variable()`.

* **MODIFIED**: module's variable name convention changed to `variable_name@parent_module_name` to make sure unique name for each variable in a complex network
* **MODIFIED**: module's name convention changed to `class_name|instance_name@parent_module_name` to make sure unique name for each module in a complex network
* **MODIFIED**: remove all specified names for `register_param()` and `register_self_updating_variable()`. Leave the variables to be named automatically by their parent module.
* **MODIFIED**: improve `model.shufflenet.ShuffleUnit`.
* **NEW**: add `Sequential` container in `dandelion.module` for usage convenience.
* **NEW**: add `model.shufflenet.ShuffleUnit_Stack` and `model.shufflenet.ShuffleNet` for ShuffleNet reference implementation.

## version 0.16.10 [7-10-2018]
* **MODIFIED**: disable all install requirements to prevent possible conflict of pip and conda channel.

## version 0.16.9 [7-10-2018]
* **MODIFIED**: import all model reference implementations into `dandelion.model`'s namespace
* **FIXED**: `ConvTransposed2D`'s `W_shape` should use `in_channels` as first dimension; incorrect `W_shape` when `num_groups` > 1.

## version 0.16.8 [7-9-2018]
* **NEW**: add `model.shufflenet.DSConv2D` for Depthwise Separable Convolution reference implementation.
* **NEW**: add `model.shufflenet.ShuffleUnit` for ShuffleNet reference implementation
* **FIXED**: `W_shape` of `module.Conv2D` should count for `num_groups`

## version 0.16.7 [7-6-2018]
* **NEW**: add `model.vgg.model_VGG16` for VGG-16 reference implementation.
* **NEW**: add `model.resnet.ResNet_bottleneck` for ResNet reference implementation
* **NEW**: add `model.feature_pyramid_net.model_FPN` for Feature Pyramid Network reference implementation

## version 0.16.6 [7-5-2018]
* **NEW**: add `functional.upsample_2d()` for 2D upsampling
* **NEW**: add `functional.upsample_2d_bilinear()`for bilinear 2D upsampling

## version 0.16.5 [7-5-2018]
* **NEW**: add `functional.spatial_pyramid_pooling()` for SPP-net implementation.

## version 0.16.4 [7-4-2018]
* **FIXED**: wrong indexing when `targets` is int vector for `objective.categorical_crossentropy_log()`.


## version 0.16.3 [7-3-2018]
* **NEW**: add `activation.log_softmax()` for more numerically stable softmax.
* **NEW**: add `objective.categorical_crossentropy_log()` for more numerically stable categorical cross-entropy
* **MODIFIED**: add `eps` argument to `objective.categorical_crossentropy()` for numerical stability purpose. Note 1e-7 is set as default value of `eps`. You can set it to 0 to get the old `categorical_crossentropy()` back.

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
