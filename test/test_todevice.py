# coding:utf-8
'''
Test Module's `.todevice` interface
According to [Issue](https://github.com/Theano/Theano/issues/6655), this feature of Theano is never finished.
Created   :  11,  2, 2018
Revised   :  11,  2, 2018
All rights reserved
'''
# ------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import os
os.environ['THEANO_FLAGS'] = "floatX=float32,mode=FAST_RUN, warn_float64='raise',contexts=dev0->cuda3;dev1->cuda6"
import theano
theano_path = os.path.split(theano.__file__)[0]
print('theano path = %s\n' % theano_path)
import theano.tensor as tensor
import dandelion
dandelion_path = os.path.split(dandelion.__file__)[0]
print('dandelion path = %s\n' % dandelion_path)
from dandelion.module import *


class build_model_on_single_device(Module):
    def __init__(self, in_dim=1024, out_dim=512, device_context=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dense1 = Dense(input_dims=self.in_dim, output_dim=self.out_dim)
        self.dense2 = Dense(input_dims=self.out_dim, output_dim=self.out_dim)
        if device_context is not None:
            self.dense1.todevice(device_context)
            self.dense2.todevice(device_context)

    def forward(self, x):
        x = self.dense1.forward(x)
        x = relu(x)
        x = self.dense2.forward(x)
        x = relu(x)
        return x

    def predict(self, x):
        x = self.dense1.predict(x)
        x = relu(x)
        x = self.dense2.predict(x)
        x = relu(x)
        return x

class build_model_on_multiple_device(Module):
    def __init__(self, in_dim=1024, out_dim=512, device_context=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dense1 = Dense(input_dims=self.in_dim, output_dim=self.out_dim)
        self.dense2 = Dense(input_dims=self.out_dim, output_dim=self.out_dim)
        if device_context is not None:
            self.dense1.todevice(device_context[0])
            self.dense2.todevice(device_context[1])
        self.device_context = device_context

    def forward(self, x):
        x = self.dense1.forward(x)
        x = relu(x)
        if self.device_context is not None:
            x = x.transfer(self.device_context[1])
        x = self.dense2.forward(x)
        x = relu(x)
        return x

    def predict(self, x):
        x = self.dense1.predict(x)
        x = relu(x)
        if self.device_context is not None:
            x = x.transfer(self.device_context[1])
        x = self.dense2.predict(x)
        x = relu(x)
        return x

def test_case_0(batch=256, in_dim=1024, out_dim=512):
    import numpy as np
    import time
    try:
        model_on_single_device   = build_model_on_single_device(in_dim=in_dim, out_dim=out_dim, device_context='dev0')
        model_on_multiple_device = build_model_on_multiple_device(in_dim=in_dim, out_dim=out_dim, device_context=['dev0', 'dev1'])
    except ValueError as e:
        if str(e).startswith("Can't transfer to target"):
            print('GPU not present, test skipped')
            return

    model_on_multiple_device.dense1.W.set_value(model_on_single_device.dense1.W.get_value())
    model_on_multiple_device.dense1.b.set_value(model_on_single_device.dense1.b.get_value())
    model_on_multiple_device.dense2.W.set_value(model_on_single_device.dense2.W.get_value())
    model_on_multiple_device.dense2.b.set_value(model_on_single_device.dense2.b.get_value())

    x = tensor.fmatrix()
    x1 = x.transfer('dev0')
    y0 = model_on_single_device.forward(x1)
    y1 = model_on_multiple_device.forward(x1)

    f0 = theano.function([x], y0, no_default_updates=True)
    f1 = theano.function([x], y1, no_default_updates=True)

    for i in range(20):
        x = np.random.rand(batch, in_dim).astype(np.float32)
        time00 = time.time()
        y0 = f0(x)
        time01 = time.time()
        y1 = f1(x)
        time02 = time.time()
        time0 = time01 - time00
        time1 = time02 - time01
        diff = np.sum(np.abs(y0 - y1))
        print('i=%d, diff=%0.6f, time0=%0.6fs, time1=%0.6fs, time0/time1=%0.4f' % (i, diff, time0, time1, time0/time1))
        if diff>1e-4:
            raise ValueError('diff is too big')

if __name__ == '__main__':
    try:
        test_case_0(batch=512, in_dim=1024, out_dim=512)
    except ValueError as e:
        if str(e).startswith("Can't transfer to target"):
            print('GPU not present, test skipped')
    print('Test passed')