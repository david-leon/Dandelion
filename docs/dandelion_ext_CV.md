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

