# coding:utf-8
'''
Reference implementation of [Shuffle-net](https://arxiv.org/abs/1707.01083)
Created   :   7,  9, 2018
Revised   :   7,  9, 2018
All rights reserved
'''
# ------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import theano.tensor as tensor
from ..module import *
from ..functional import *
from ..activation import *


class DSConv2D(Module):
    """
    Depthwise Separable Convolution (https://arxiv.org/abs/1610.02357)
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), dilation=(1,1), pad='valid'):
        """
        depthwise separable convolution = separate 3*3 conv on each input channel, then use 1*1 conv to map to #out_channels output channels
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param dilation:
        :param pad:
        """
        super().__init__()
        self.conv_depthwise = Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, pad=pad, dilation=dilation, num_groups=in_channels)
        self.conv_pointwise = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), pad='same')

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return:
        """
        self.work_mode = 'train'

        x = self.conv_depthwise.forward(x)   # W = (g, 1, h, w), g = group number = in_channels, (h, w) = kernel_size
        x = self.conv_pointwise.forward(x)   # W = (out_channels, n_channels, 1, 1)
        return x

    def predict(self, x):
        self.work_mode = 'inference'

        x = self.conv_depthwise.predict(x)   # W = (g, h, w), g = group number = in_channels, (h, w) = kernel_size
        x = self.conv_pointwise.predict(x)   # W = (in_channels, out_channels, 1, 1)
        return x


class ShuffleUnit(Module):
    """
    Shuffle unit reference implementation (https://arxiv.org/abs/1610.02357)
    """
    def __init__(self, outer_channel=256, inner_channel=64, group_num=4, border_mode='same', batchnorm_mode=1, activation=relu):
        """

        :param outer_channel: channel number of block input
        :param inner_channel: channel number inside the block
        :param group_num: number of convolution groups
        :param border_mode:
        :param batchnorm_mode: {0 | 1 | 2}. 0 means no batch normalization applied; 1 means batch normalization applied to each cnn;
                               2 means batch normalization only applied to the last cnn
        :param activation: default = relu. Note no activation applied to the last element-wise sum output.
        """
        super().__init__()
        self.activation = activation
        self.batchnorm_mode = batchnorm_mode
        self.group_num = group_num
        self.conv1 = Conv2D(in_channels=outer_channel, out_channels=inner_channel, kernel_size=1, pad=border_mode, num_groups=group_num)
        self.conv2 = DSConv2D(in_channels=inner_channel, out_channels=inner_channel, kernel_size=3, pad=border_mode)
        self.conv3 = Conv2D(in_channels=inner_channel, out_channels=outer_channel, kernel_size=1, pad=border_mode)
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

    @staticmethod
    def channel_shuffle(x, group_num):
        """
        Pseudo shuffle channel by dimshuffle & reshape
        :param x: (B, C, H, W)
        :param group_num: int
        :return: 
        """
        x = x.dimshuffle((0, 2, 3, 1)) # (B, H, W, c*g)
        B, H, W, C = x.shape
        x = tensor.reshape(x, (B, H, W, C//group_num, group_num))
        x = x.dimshuffle((0, 1, 2, 4, 3))
        x = x.flatten(ndim=4)
        x = x.dimshuffle(0, 3, 1, 2)
        return x

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
        x = self.activation(x)          # (B, c*g, H, W)
        x = self.channel_shuffle(x, self.group_num)
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
        x = self.activation(x)          # (B, c*g, H, W)
        x = self.channel_shuffle(x, self.group_num)
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
