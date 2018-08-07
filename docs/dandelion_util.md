## gpickle
Pickle with gzip enabled.
```python
.dump(data, filename, compresslevel=9)
```
* **data**: data to be dumped to file
* **filename**: file path
* **compresslevel**: gzip compression level, default = 9.

```python
.load(filename)
```
* **filename**: file to be loaded

_______________________________________________________________________
## theano_safe_run
Help catch theano memory exceptions during running theano function
```python
theano_safe_run(fn, input_list)
```
* **fn**: theano function to run
* **input_list**: list of input arguments
* **return**: errcode and funtion excution result

`theano_safe_run()` catches the following 4 memory exceptions (range from theano 0.x to 1.x):
* MemoryError. errcode=1
* CudaNdarray_ZEROS: allocation failed. errcode=2
* gpudata_alloc: cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY: out of memory. errcode=3
* cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY: out of memory. errcode=4

