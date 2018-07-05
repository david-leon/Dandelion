## pool_1d
Pooling 1 dimension along the given axis, support for any dimensional input.
```python
pool_1d(x, ws=2, ignore_border=True, stride=None, pad=0, mode='max', axis=-1)
```
* **ws**: scalar int. Factor by which to downsample the input
* **ignore_border**: bool. When `True`, dimension size=5 with `ws`=2 will generate a dimension size=2 output. 3 otherwise.
* **stride**: scalar int. The number of shifts over rows/cols to get the next pool region. If stride is None, it is considered equal to ws (no overlap on pooling regions), eg: `stride`=1 will shifts over one row for every iteration.
* **pad**: pad zeros to extend beyond border of the input
* **mode**: {`max`, `sum`, `average_inc_pad`, `average_exc_pad`}. Operation executed on each window. `max` and `sum` always exclude the padding in the computation. `average` gives you the choice to include or exclude it.
* **axis**: scalar int. Specify along which axis the pooling will be done

_______________________________________________________________________
## pool_2d
Pooling 2 dimension along the last 2 dimensions of input, support for any dimensional input with `ndim`>=2.
```python
pool_2d(x, ws=(2,2), ignore_border=True, stride=None, pad=(0,0), mode='max')
```
* **ws**: scalar tuple. Factor by which to downsample the input
* **ignore_border**: bool. When `True`, (5,5) input with `ws`=(2,2) will generate a (2,2) output. (3,3) otherwise.
* **stride**: scalar tuple. The number of shifts over rows/cols to get the next pool region. If stride is None, it is considered equal to ws (no overlap on pooling regions), eg: `stride`=(1,1) will shifts over one row and one column for every iteration.
* **pad**: pad zeros to extend beyond border of the input
* **mode**: {`max`, `sum`, `average_inc_pad`, `average_exc_pad`}. Operation executed on each window. `max` and `sum` always exclude the padding in the computation. `average` gives you the choice to include or exclude it.

_______________________________________________________________________
## pool_3d
Pooling 3 dimension along the last 3 dimensions of input, support for any dimensional input with `ndim`>=3.
```python
pool_3d(x, ws=(2,2,2), ignore_border=True, stride=None, pad=(0,0,0), mode='max')
```
* **ws**: scalar tuple. Factor by which to downsample the input
* **ignore_border**: bool. When `True`, (5,5,5) input with `ws`=(2,2,2) will generate a (2,2,2) output. (3,3,3) otherwise.
* **stride**: scalar tuple. The number of shifts over rows/cols to get the next pool region. If stride is None, it is considered equal to ws (no overlap on pooling regions).
* **pad**: pad zeros to extend beyond border of the input
* **mode**: {`max`, `sum`, `average_inc_pad`, `average_exc_pad`}. Operation executed on each window. `max` and `sum` always exclude the padding in the computation. `average` gives you the choice to include or exclude it.

_______________________________________________________________________
## align_crop
Align a list of tensors at each axis by specified rules and crop them to make axis concatenation possible.
```python
align_crop(tensor_list, cropping)
```
* **tensor_list**: list of tensors to be processed, they much have the same `ndim`s.
* **cropping**: list of cropping rules for each dimension. Acceptable rules include {`None`|`lower`|`upper`|`center`}. 
  * `None`: this axis is not cropped, tensors are unchanged in this axis
  * `lower`: tensors are cropped choosing the lower portion in this axis as `a[:crop_size, ...]`
  * `upper`: tensors are cropped choosing the upper portion in this axis as `a[-crop_size:, ...]`
  * `center`: tensors are cropped choosing the central portion in this axis as ``a[offset:offset+crop_size, ...]`` where ``offset = (a.shape[0]-crop_size)//2)``

_______________________________________________________________________
## spatial_pyramid_pooling
Spatial pyramid pooling. This function will use different scale pooling pyramid to generate spatially fix-sized output no matter the spatial size of input, useful when CNN+FC used for image classification or detection with variable-sized samples.
```python
spatial_pyramid_pooling(x, pyramid_dims=(6, 4, 2, 1), mode='max')
```
* **x**: 4D tensor with shape (B, C, H, W)
* **pyramid_dims**: list or tuple of integers. Refer to Ref[1] for details.
* **mode**: {`max`, `sum`, `average_inc_pad`, `average_exc_pad`}. Operation executed on each window. `max` and `sum` always exclude the padding in the computation. `average` gives you the choice to include or exclude it.

<sub>Ref[1]: He, Kaiming et al (2015), Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition. http://arxiv.org/pdf/1406.4729.pdf</sub>
