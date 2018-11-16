**Image Processing and Computer Vision Toolkits**
_______________________________________________________________________

## imread
Read image file and return as numpy `ndarray`, using PILLOW as backend. Support for EXIF rotation specification.
```python
imread(f, flatten=False, dtype='float32')
```
* **f**: str or file object. The file name or file object to be read from.
* **flatten**: bool. If `True`, flattens the color channels into a single gray-scale channel.
* **dtype**: returned data type

_______________________________________________________________________
## imsave
Save an image `ndarray` into file, using PILLOW as backend 
```python
imsave(f, I, **params)
```
* **f**: str or file object. The file name or file object to be written into.
* **I**: Image `ndarray`. Note for `jpeg` format, `I` should be of `uint8` type.
* **params**: other parameters passed directly to PILLOW's `image.save()`

_______________________________________________________________________
## imresize
Resize image, using scipy as backend 
```python
imresize(I, size, interp='bilinear', mode=None)
```
* **I**: Image `ndarray`
* **size**: target size
* **interp**: Interpolation to use for resizing, {'nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic'}.
* **mode**: . The PIL image mode ('P', 'L', etc.) to convert `I` before resizing, optional.

_______________________________________________________________________
## imrotate
Rotate image, using opencv as backend 
```python
imrotate(I, angle, padvalue=0.0, interpolation='linear', target_size=None, border_mode='reflect_101')
```
* **I**: Image `ndarray`
* **angle**: in degree, positive for counter-clockwise
* **interpolation**: image interpolation method, {'linear'|'nearest'|'cubic'|'LANCZOS4'|'area'}, refer to opencv:INTER_* constants for details
* **border_mode**: image boundary handling method, {'reflect_101'|'reflect'|'wrap'|'constant'|'replicate'}, refer to opencv:BORDER_* constants for details
* **padvalue**: used when `border_mode` = 'constant'
* **target_size**: target size of output image, optional.