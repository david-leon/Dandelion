# coding:utf-8
'''
Model definition of [feature pyramid network](https://arxiv.org/abs/1612.03144)
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
from .resnet import ResNet_bottleneck


class model_FPN(Module):
    """
    Reference implementation of [feature pyramid network](https://arxiv.org/abs/1612.03144)
    return tuple of (p2, p3, p4, p5), cnn pyramid features at different scales
    """

    def __init__(self, input_channel=3, base_n_filters=64, batchnorm_mode=1):
        """
        8 effective conv in depth
        :param input_channel:
        :param base_n_filters:
        :param batchnorm_mode: passed to `ResNet_bottleneck()`
        """
        super().__init__()
        self.conv1 = Conv2D(in_channels=input_channel, out_channels=base_n_filters, kernel_size=3, pad='same', stride=(2, 2))
        self.bn1   = BatchNorm(input_shape=(None, base_n_filters, None, None))
        self.conv2 = Conv2D(in_channels=base_n_filters, out_channels=base_n_filters * 2, kernel_size=3, pad='same', stride=(2, 2))
        self.bn2   = BatchNorm(input_shape=(None, base_n_filters * 2, None, None))
        self.conv3 = Conv2D(in_channels=base_n_filters * 2, out_channels=base_n_filters * 4, kernel_size=3, pad='same', stride=(2, 2))
        self.bn3   = BatchNorm(input_shape=(None, base_n_filters * 4, None, None))
        self.conv4 = Conv2D(in_channels=base_n_filters * 4, out_channels=base_n_filters * 8, kernel_size=3, pad='same', stride=(2, 2))
        self.bn4   = BatchNorm(input_shape=(None, base_n_filters * 8, None, None))

        self.res_block1 = ResNet_bottleneck(outer_channel=base_n_filters, inner_channel=base_n_filters // 2, batchnorm_mode=batchnorm_mode)
        self.res_block2 = ResNet_bottleneck(outer_channel=base_n_filters * 2, inner_channel=base_n_filters, batchnorm_mode=batchnorm_mode)
        self.res_block3 = ResNet_bottleneck(outer_channel=base_n_filters * 4, inner_channel=base_n_filters * 2, batchnorm_mode=batchnorm_mode)
        self.res_block4 = ResNet_bottleneck(outer_channel=base_n_filters * 8, inner_channel=base_n_filters * 4, batchnorm_mode=batchnorm_mode)

        self.conv5 = Conv2D(in_channels=base_n_filters * 8, out_channels=base_n_filters * 4, kernel_size=1, pad='same')
        self.conv6 = Conv2D(in_channels=base_n_filters * 4, out_channels=base_n_filters * 4, kernel_size=1, pad='same')
        self.conv7 = Conv2D(in_channels=base_n_filters * 2, out_channels=base_n_filters * 4, kernel_size=1, pad='same')
        self.conv8 = Conv2D(in_channels=base_n_filters,     out_channels=base_n_filters * 4, kernel_size=1, pad='same')

        self.conv9  = Conv2D(in_channels=base_n_filters * 4, out_channels=base_n_filters * 4, kernel_size=1, pad='same')
        self.conv10 = Conv2D(in_channels=base_n_filters * 4, out_channels=base_n_filters * 4, kernel_size=1, pad='same')
        self.conv11 = Conv2D(in_channels=base_n_filters * 4, out_channels=base_n_filters * 4, kernel_size=1, pad='same')

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return:
        """
        self.work_mode = 'train'

        c1 = relu(self.bn1.forward(self.conv1.forward(x)))  # stride = 2
        c2 = relu(self.res_block1.forward(c1))

        c3 = relu(self.bn2.forward(self.conv2.forward(c2)))  # stride = 4
        c3 = relu(self.res_block2.forward(c3))

        c4 = relu(self.bn3.forward(self.conv3.forward(c3)))  # stride = 8
        c4 = relu(self.res_block3.forward(c4))

        c5 = relu(self.bn4.forward(self.conv4.forward(c4)))  # stride = 16
        c5 = relu(self.res_block4.forward(c5))

        p5 = relu(self.conv5.forward(c5))
        p4 = relu(self.conv6.forward(c4)) + upsample_2d(p5, ratio=2, mode='repeat')
        p3 = relu(self.conv7.forward(c3)) + upsample_2d(p4, ratio=2, mode='repeat')
        p2 = relu(self.conv8.forward(c2)) + upsample_2d(p3, ratio=2, mode='repeat')

        p4 = relu(self.conv9.forward(p4))
        p3 = relu(self.conv10.forward(p3))
        p2 = relu(self.conv11.forward(p2))

        return p2, p3, p4, p5

    def predict(self, x):
        self.work_mode = 'inference'

        c1 = relu(self.bn1.predict(self.conv1.predict(x)))  # stride = 2
        c2 = relu(self.res_block1.predict(c1))

        c3 = relu(self.bn2.predict(self.conv2.predict(c2)))  # stride = 4
        c3 = relu(self.res_block2.predict(c3))

        c4 = relu(self.bn3.predict(self.conv3.predict(c3)))  # stride = 8
        c4 = relu(self.res_block3.predict(c4))

        c5 = relu(self.bn4.predict(self.conv4.predict(c4)))  # stride = 16
        c5 = relu(self.res_block4.predict(c5))

        p5 = relu(self.conv5.predict(c5))
        p4 = relu(self.conv6.predict(c4)) + upsample_2d(p5, ratio=2, mode='repeat')
        p3 = relu(self.conv7.predict(c3)) + upsample_2d(p4, ratio=2, mode='repeat')
        p2 = relu(self.conv8.predict(c2)) + upsample_2d(p3, ratio=2, mode='repeat')

        p4 = relu(self.conv9.predict(p4))
        p3 = relu(self.conv10.predict(p3))
        p2 = relu(self.conv11.predict(p2))

        return p2, p3, p4, p5
