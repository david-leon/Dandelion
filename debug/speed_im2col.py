import numpy as np
import theano as th
import theano.tensor as T
import theano.tensor.nnet.neighbours as N

import time


def im_to_col(im, psize, n_channels=3):
  """Similar to MATLAB's im2col function.

  Args:
    im - a Theano tensor3, of the form <n_channels, height, width>.
    psize - an int specifying the (square) block size to use
    n_channels - the number of channels in im

  Returns: a 5-tensor of the form <patch_id_i, patch_id_j, n_channels, psize,
           psize>.
  """
  assert im.ndim == 3, "im must have dimension 3."
  im = im[:, ::-1, ::-1]
  res = T.zeros((n_channels, psize * psize, im.shape[1] - psize + 1,
                 im.shape[2] - psize + 1))
  filts = T.reshape(T.eye(psize * psize, psize * psize),
                    (psize * psize, psize, psize))
  filts = T.shape_padleft(filts).dimshuffle((1, 0, 2, 3))

  for i in range(n_channels):
    cur_slice = T.shape_padleft(im[i], n_ones=2)
    res = T.set_subtensor(res[i], T.nnet.conv.conv2d(cur_slice, filts)[0])

  return res.dimshuffle((0, 2, 3, 1)).reshape(
      (n_channels, im.shape[1] - psize + 1, im.shape[2] - psize + 1,
       psize, psize)).dimshuffle((1, 2, 0, 3, 4))


def main():
  # Turn these knobs if you wish to work with larger/smaller data
  img_dims = (500, 500)
  fsize = 2
  n_channels = 3

  # Create a random image
  img = np.asarray(np.random.rand(*((n_channels,) + img_dims)),
                   dtype=th.config.floatX)
  img = np.arange(n_channels * img_dims[0] * img_dims[1],
                  dtype=th.config.floatX).reshape(n_channels, *img_dims)

  # Adapt the code to use the CPU/GPU. In the GPU case, do NOT transfer the
  # results back to memory.
  wrap = ((lambda x: x) if th.config.device == "cpu" else
          (lambda x: th.Out(th.sandbox.cuda.basic_ops.gpu_from_host(x),
                            borrow=True)))

  # Convolution method
  x = th.shared(img)
  f = th.function(
      inputs=[],
      outputs=wrap(im_to_col(x, fsize, n_channels=n_channels)),
      name='im_to_col')

  # Time the convolution method
  tic = time.time()
  out_conv = f()
  conv_time = time.time() - tic
  print ("Convolution-based method: {0}".format(conv_time))

  # Time the neighbors method
  neighs = N.NeighbourhoodsFromImages(1, (fsize, fsize), strides=(1, 1),
                                          ignore_border=True)(x)
  f = th.function([], outputs=wrap(neighs),
                  name='old neighs')
  tic = time.time()
  out_old = f()
  neigh_time = time.time() - tic
  print ("Neighbors-based method: {0}".format(neigh_time))

  # Time the new neighbours method ignore border
  neighs = N.images2neibs(x.dimshuffle('x', 0, 1, 2),
                                              (fsize, fsize), (1, 1),
                                              mode='ignore_borders')
  f = th.function([], outputs=wrap(neighs),
                  name='new neighs ignore border')
  tic = time.time()
  out_new = f()
  neigh_time = time.time() - tic
  print ("New Neighbors-based ignore border method: {0}".format(neigh_time))

  # Time the new neighbours method
  neighs = N.images2neibs(x.dimshuffle('x', 0, 1, 2),
                                              (fsize, fsize), (1, 1),
                                              mode='valid')
  f = th.function([], outputs=wrap(neighs),
                  name='new neighs valid')
  tic = time.time()
  out_new = f()
  neigh_time = time.time() - tic
  print ("New Neighbors-based valid method: {0}".format(neigh_time))

  # Print speedup results
  if conv_time < neigh_time:
    print ("Conv faster than neigh. Speedup: {0}x".format(neigh_time / conv_time))
  else:
    print ("Neigh faster than conv. Speedup: {0}x".format(conv_time / neigh_time))
if __name__ == "__main__":
  main()
