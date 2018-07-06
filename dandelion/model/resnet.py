# coding:utf-8
'''
Model definition of ResNet
Created   :   7,  6, 2018
Revised   :   7,  6, 2018
All rights reserved
'''
# ------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import theano.tensor as tensor
from ..module import *
from ..functional import *
from ..activation import *


class ResNet_bottleneck(Module):
    """
    [ResNet bottleneck block] (https://arxiv.org/abs/1512.03385).
    """

    def __init__(self, outer_channel=256, inner_channel=64, border_mode='same', batchnorm_mode=1, activation=relu):
        """

        :param outer_channel: channel number of block input
        :param inner_channel: channel number inside the block
        :param border_mode:
        :param batchnorm_mode: {0 | 1 | 2}. 0 means no batch normalization applied; 1 means batch normalization applied to each cnn;
                               2 means batch normalization only applied to the last cnn
        :param activation: default = relu. Note no activation applied to the last element-wise sum output.
        """
        super().__init__()
        self.activation = activation
        self.batchnorm_mode = batchnorm_mode
        self.conv1 = Conv2D(in_channels=outer_channel,  out_channels=inner_channel,  kernel_size=1, pad=border_mode)
        self.conv2 = Conv2D(in_channels=inner_channel,  out_channels=inner_channel,  kernel_size=3, pad=border_mode)
        self.conv3 = Conv2D(in_channels=inner_channel,  out_channels=outer_channel,  kernel_size=1, pad=border_mode)
        if batchnorm_mode == 0:   # no batch normalization
            pass
        elif batchnorm_mode == 1: # batch normalization per convolution
            self.bn1 = BatchNorm(input_shape=(None, inner_channel, None, None))
            self.bn2 = BatchNorm(input_shape=(None, inner_channel, None, None))
            self.bn3 = BatchNorm(input_shape=(None, outer_channel, None, None))
        elif batchnorm_mode == 2: # only one batch normalization at the end
            self.bn3  = BatchNorm(input_shape=(None, outer_channel, None, None))
        else:
            raise ValueError('batchnorm_mode should be {0 | 1 | 2}')

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return:
        """
        self.work_mode = 'train'

        x0 = x
        x  = self.conv1.forward(x)
        if self.batchnorm_mode == 1:
            x = self.bn1.forward(x)
        x = self.activation(x)
        x  = self.conv2.forward(x)
        if self.batchnorm_mode == 1:
            x = self.bn2.forward(x)
        x = self.activation(x)
        x  = self.conv3.forward(x)
        if self.batchnorm_mode in {1, 2}:
            x = self.bn3.forward(x)
        x = self.activation(x)
        x = x + x0
        return x

    def predict(self, x):
        self.work_mode = 'inference'

        x0 = x
        x  = self.conv1.predict(x)
        if self.batchnorm_mode == 1:
            x = self.bn1.predict(x)
        x = self.activation(x)
        x  = self.conv2.predict(x)
        if self.batchnorm_mode == 1:
            x = self.bn2.predict(x)
        x = self.activation(x)
        x  = self.conv3.predict(x)
        if self.batchnorm_mode in {1, 2}:
            x = self.bn3.predict(x)
        x = self.activation(x)
        x = x + x0
        return x
