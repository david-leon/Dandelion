# coding:utf-8
'''
Model definition of ancient VGG CNN nets
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


class model_VGG16(Module):
    """
    VGG16-net reference implementation: 13 cnn + 3 dense + 2 dropout.
    """

    def __init__(self, channel=3, im_height=224, im_width=224, Nclass=1000, kernel_size=3, border_mode=(1, 1), flip_filters=False):
        super().__init__()
        self.conv1_1 = Conv2D(in_channels=channel, out_channels=64, kernel_size=kernel_size, pad=border_mode, input_shape=(im_height, im_width), flip_filters=flip_filters)
        self.conv1_2 = Conv2D(in_channels=64,  out_channels=64,  kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.conv2_1 = Conv2D(in_channels=64,  out_channels=128, kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.conv2_2 = Conv2D(in_channels=128, out_channels=128, kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.conv3_1 = Conv2D(in_channels=128, out_channels=256, kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.conv3_2 = Conv2D(in_channels=256, out_channels=256, kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.conv3_3 = Conv2D(in_channels=256, out_channels=256, kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.conv4_1 = Conv2D(in_channels=256, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.conv4_2 = Conv2D(in_channels=512, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.conv4_3 = Conv2D(in_channels=512, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.conv5_1 = Conv2D(in_channels=512, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.conv5_2 = Conv2D(in_channels=512, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.conv5_3 = Conv2D(in_channels=512, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=flip_filters)
        self.fc6     = Dense(input_dims=512 * im_height//32 * im_width//32, output_dim=4096)
        self.fc7     = Dense(input_dims=4096, output_dim=4096)
        self.fc8     = Dense(input_dims=4096, output_dim=Nclass)
        self.fc6_dropout = Dropout()
        self.fc7_dropout = Dropout()


    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return:
        """
        self.work_mode = 'train'

        x = relu(self.conv1_1.forward(x))  # (B, 64, 224, 224)
        x = relu(self.conv1_2.forward(x))
        x = pool_2d(x, ws=(2, 2))           # (B, 64, 112, 112)
        x = relu(self.conv2_1.forward(x))
        x = relu(self.conv2_2.forward(x))
        x = pool_2d(x, ws=(2, 2))          # (B, 128, 56, 56)
        x = relu(self.conv3_1.forward(x))
        x = relu(self.conv3_2.forward(x))
        x = relu(self.conv3_3.forward(x))
        x = pool_2d(x, ws=(2, 2))          # (B, 256, 28, 28)
        x = relu(self.conv4_1.forward(x))
        x = relu(self.conv4_2.forward(x))
        x = relu(self.conv4_3.forward(x))
        x = pool_2d(x, ws=(2, 2))          # (B, 512, 14, 14)
        x = relu(self.conv5_1.forward(x))
        x = relu(self.conv5_2.forward(x))
        x = relu(self.conv5_3.forward(x))
        x = pool_2d(x, ws=(2, 2))          # (B, 512, 7, 7)
        x = tensor.flatten(x, ndim=2)  # (B, 512 * 7 * 7)
        x = relu(self.fc6.forward(x))
        x = self.fc6_dropout.forward(x, p=0.5)
        x = relu(self.fc7.forward(x))
        x = self.fc7_dropout.forward(x, p=0.5)
        x = softmax(self.fc8.forward(x))

        return x

    def predict(self, x):
        self.work_mode = 'inference'

        x = relu(self.conv1_1.predict(x))  # (B, 64, 224, 224)
        x = relu(self.conv1_2.predict(x))
        x = pool_2d(x, ws=(2, 2))           # (B, 64, 112, 112)
        x = relu(self.conv2_1.predict(x))
        x = relu(self.conv2_2.predict(x))
        x = pool_2d(x, ws=(2, 2))          # (B, 128, 56, 56)
        x = relu(self.conv3_1.predict(x))
        x = relu(self.conv3_2.predict(x))
        x = relu(self.conv3_3.predict(x))
        x = pool_2d(x, ws=(2, 2))          # (B, 256, 28, 28)
        x = relu(self.conv4_1.predict(x))
        x = relu(self.conv4_2.predict(x))
        x = relu(self.conv4_3.predict(x))
        x = pool_2d(x, ws=(2, 2))          # (B, 512, 14, 14)
        x = relu(self.conv5_1.predict(x))
        x = relu(self.conv5_2.predict(x))
        x = relu(self.conv5_3.predict(x))
        x = pool_2d(x, ws=(2, 2))          # (B, 512, 7, 7)
        x = tensor.flatten(x, ndim=2)      # (B, 512 * 7 * 7)
        x = relu(self.fc6.predict(x))
        x = relu(self.fc7.predict(x))
        x = softmax(self.fc8.predict(x))

        return x
