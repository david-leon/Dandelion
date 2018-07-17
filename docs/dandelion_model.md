## U-net FCN
Reference implementation of [U-net](https://arxiv.org/abs/1505.04597) FCN

```python
class model_Unet(channel=1, im_height=128, im_width=128, Nclass=2, kernel_size=3, 
                 border_mode='same', base_n_filters=64, output_activation=softmax)
```
* **channel**: input channel number
* **Nclass**: output channel number

The model accepts input of shape in the order of (B, C, H, W), and outputs with shape in the order of (B, H, W, C).


_______________________________________________________________________
## VGG-16 network
Reference implementation of [VGG-16](https://arxiv.org/abs/1409.1556) network

```python
class model_VGG16(channel=3, im_height=224, im_width=224, Nclass=1000, 
                  kernel_size=3, border_mode=(1, 1))
```
* **channel**: input channel number
* **Nclass**: output class number

The model accepts input of shape in the order of (B, C, H, W), and outputs with shape (B, N).


_______________________________________________________________________
## ResNet bottleneck
Reference implementation of bottleneck building block of [ResNet](https://arxiv.org/abs/1512.03385) network

```python
class ResNet_bottleneck(outer_channel=256, inner_channel=64, border_mode='same',
                        batchnorm_mode=1, activation=relu)
```
* **outer_channel**: channel number of block input
* **inner_channel**: channel number inside the block
* **batchnorm_mode**: {0 | 1 | 2}. 0 means no batch normalization applied; 1 means batch normalization applied to each cnn; 2 means batch normalization only applied to the last cnn
* **activation**: default = relu. **Note no activation applied to the last element-wise sum output.**

The model accepts input of shape in the order of (B, C, H, W), and outputs with the same shape.


_______________________________________________________________________
## Feature Pyramid Network
Reference implementation of [feature pyramid network](https://arxiv.org/abs/1612.03144)

```python
class model_FPN(input_channel=3, base_n_filters=64, batchnorm_mode=1)
```
* **batchnorm_mode**: same with `ResNet_bottleneck`
* **return** 4-element tuple `(p2, p3, p4, p5)`,  CNN pyramid features at different scales, each with #channel = 4 * `base_n_filters`


_______________________________________________________________________
## Depthwise Separable Convolution
Reference implementation of [Depthwise Separable Convolution](https://arxiv.org/abs/1610.02357)

```python
class DSConv2D(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), 
               dilation=(1,1), pad='valid')
```
* **input_channels**: int. Input shape is (B, input_channels, H_in, W_in)
* **out_channels**: int. Output shape is (B output_channels, H_out, W_out)
* **kernel_size**: int scalar or tuple of int. Convolution kernel size
* **stride**: Factor by which to subsample the output
* **pad**: `same`/`valid`/`full` or 2-element tuple of int. Control image border padding.
* **dilation**: factor by which to subsample (stride) the input.

The model do the depthwise 2D convolution per-channel of input, then map the output to #out_channels number of channel by pointwise 1*1 convolution. No activation applied inside.


_______________________________________________________________________
## ShuffleUnit
Reference implementation of [shuffle-net](https://arxiv.org/abs/1707.01083) unit

```python
class ShuffleUnit(in_channels=256, inner_channels=None, out_channels=None, group_num=4, border_mode='same', 
                  batchnorm_mode=1, activation=relu, stride=(1,1), dilation=(1,1), fusion_mode='add')
```
* **in_channels**: channel number of unit input
* **inner_channel**: optional, channel number inside the unit, default = `in_channels//4`
* **out_channels**: channel number of unit output, only used when `fusion_mode` = 'concat', and must > `in_channels`
* **group_num**: number of convolution groups
* **border_mode**: only `same` allowed
* **batchnorm_mode**: {0 | 1 | 2}. 0 means no batch normalization applied; 1 means batch normalization applied to each cnn; 2 means batch normalization only applied to the last cnn
* **activation**: default = relu. **Note no activation applied to the last output.**
* **stride, dilation**: only used for depthwise separable convolution module inside
* **fusion_mode**: {'add' | 'concat'}.
* **return**: convolution result with #channel = `in_channels` when `fusion_mode`='add', #channel = `out_channels` when `fusion_mode`='concat'

## ShuffleUnit_Stack
Reference implementation of shuffle-net unit stack

```python
class ShuffleUnit_Stack(in_channels, inner_channels=None, out_channels=None, group_num=4, batchnorm_mode=1, 
                        activation=relu, stack_size=3)
```
* **in_channels**: channel number of input
* **inner_channel**: optional, channel number inside the shuffle-unit, default = `in_channels//4`
* **out_channels**: channel number of stack output, must > `in_channels`
* **group_num**: number of convolution groups
* **batchnorm_mode**: {0 | 1 | 2}. 0 means no batch normalization applied; 1 means batch normalization applied to each cnn; 2 means batch normalization only applied to the last cnn
* **activation**: default = relu. **Note no activation applied to the last output.**
* **stack_size**: number of shuffle-unit in the stack

## ShuffleNet
Reference implementation of [shuffle-net](https://arxiv.org/abs/1707.01083), without the final Dense layer.

```python
class model_ShuffleNet(in_channels, group_num=4, stage_channels=(24, 272, 544, 1088), stack_size=(3, 7, 3), 
                       batchnorm_mode=1, activation=relu)
```
* **in_channels**: channel number of input
* **group_num**: number of convolution groups
* **stage_channels**: channel number of each stage output.
* **stack_size**: size of each stack.
* **batchnorm_mode**: {0 | 1 | 2}. 0 means no batch normalization applied; 1 means batch normalization applied to each cnn; 2 means batch normalization only applied to the last cnn
* **activation**: default = relu. **Note no activation applied to the last output.**

_______________________________________________________________________
## Shuffle-Seg network
Model reference implementation of [ShuffleSeg](https://arxiv.org/abs/1803.03816)

```python
class model_ShuffleSeg(in_channels=1, Nclass=6, SF_group_num=4, SF_stage_channels=(24, 272, 544, 1088), 
                       SF_stack_size=(3, 7, 3), SF_batchnorm_mode=1, SF_activation=relu)
```
* **in_channels**: channel number of input
* **Nclass**: output class number
* **SF_group_num**: number of convolution groups for inside ShuffleNet encoder.
* **SF_stage_channels**: channel number of each stage output for inside ShuffleNet encoder.
* **SF_stack_size**: size of each stack for inside ShuffleNet encoder.
* **SF_batchnorm_mode**: {0 | 1 | 2}. 0 means no batch normalization applied; 1 means batch normalization applied to each cnn; 2 means batch normalization only applied to the last cnn. For inside ShuffleNet encoder
* **SF_activation**: default = relu. For inside ShuffleNet encoder.