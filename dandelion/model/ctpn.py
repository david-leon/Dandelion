# coding:utf-8
'''
Reference implementation of [CTPN](https://arxiv.org/abs/1609.03605) for text line detection
Created   :   7, 26, 2018
Revised   :   8,  2, 2018  add alternative implementation to bypass the absense of `im2col` in Theano.
              9,  3, 2018  modified: change output bbox's shape to (B, H, W, k, n) (from (B, H, W, k*n)).
All rights reserved
'''
# ------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import theano.tensor as tensor
from ..module import *
from ..functional import *
from ..activation import *


class model_CTPN(Module):
    """
    Reference implementation of [CTPN](https://arxiv.org/abs/1609.03605) for text line detection
    """

    def __init__(self, k=10, do_side_refinement_regress=False,
                 batchnorm_mode=1, channel=3, im_height=None, im_width=None,
                 kernel_size=3, border_mode=(1, 1), VGG_flip_filters=False,
                 im2col=None):
        """

        :param k: anchor box number
        :param do_side_refinement_regress: whether implement side refinement regression
        :param batchnorm_mode: 1|0, whether insert batch normalization into the end of each convolution stage of VGG-16 net
        :param channel: input channel number
        :param im_height: input image height, optional
        :param im_width:  input image width, optional
        :param kernel_size: convolution kernel size of VGG-16 net
        :param border_mode: border mode of VGG-16 net
        :param VGG_flip_filters: whether flip convolution kernels for VGG-16 net
        :param im2col: function corresponding to Caffe's `im2col()`. If None, the CTPN implementation will not strictly follow the original paper.
        """
        super().__init__()
        self.batchnorm_mode = batchnorm_mode
        self.k = k
        self.im2col = im2col
        #--- encoding part, VGG16, 13 conv layers ---#
        self.conv1_1 = Conv2D(in_channels=channel, out_channels=64, kernel_size=kernel_size, pad=border_mode, input_shape=(im_height, im_width), flip_filters=VGG_flip_filters)
        self.conv1_2 = Conv2D(in_channels=64,  out_channels=64,  kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)
        self.conv2_1 = Conv2D(in_channels=64,  out_channels=128, kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)
        self.conv2_2 = Conv2D(in_channels=128, out_channels=128, kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)
        self.conv3_1 = Conv2D(in_channels=128, out_channels=256, kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)
        self.conv3_2 = Conv2D(in_channels=256, out_channels=256, kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)
        self.conv3_3 = Conv2D(in_channels=256, out_channels=256, kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)
        self.conv4_1 = Conv2D(in_channels=256, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)
        self.conv4_2 = Conv2D(in_channels=512, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)
        self.conv4_3 = Conv2D(in_channels=512, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)
        self.conv5_1 = Conv2D(in_channels=512, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)
        self.conv5_2 = Conv2D(in_channels=512, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)
        self.conv5_3 = Conv2D(in_channels=512, out_channels=512, kernel_size=kernel_size, pad=border_mode, flip_filters=VGG_flip_filters)  # (B, C, H, W)
        if batchnorm_mode == 0:
            pass
        elif batchnorm_mode == 1:
            self.bn1 = BatchNorm(input_shape=(None,  64, None, None))
            self.bn2 = BatchNorm(input_shape=(None, 128, None, None))
            self.bn3 = BatchNorm(input_shape=(None, 256, None, None))
            self.bn4 = BatchNorm(input_shape=(None, 512, None, None))
            self.bn5 = BatchNorm(input_shape=(None, 512, None, None))
        else:
            raise ValueError('batchnorm_mode should = 0 or 1')
        #--- detection part ---#
        if im2col is not None:  # implementation strictly follow the reference paper
            self.lstm_f    = LSTM(input_dims=512*9, hidden_dim=128, grad_clipping=100) # (W, B*H, C)
            self.lstm_b    = LSTM(input_dims=512*9, hidden_dim=128, grad_clipping=100) # (W, B*H, C)
            self.conv_rpn  = Conv2D(in_channels=256, out_channels=512, kernel_size=1, pad='same')  # (B, C, H, W)
            self.conv_cls_score = Conv2D(in_channels=512, out_channels=2 * k, kernel_size=1, pad='same')  # (B, 2*k, H, W)
            if do_side_refinement_regress:
                self.conv_bbox_pred = Conv2D(in_channels=512, out_channels=3 * k, kernel_size=1, pad='same')  # (B, 3*k, H, W), include side-refinement
            else:
                self.conv_bbox_pred = Conv2D(in_channels=512, out_channels=2 * k, kernel_size=1, pad='same')  # (B, 2*k, H, W), no side-refinement

        else: # implementation putting convolution before RNN, doesn't follow the reference paper
            self.conv_rpn = Conv2D(in_channels=512, out_channels=512, kernel_size=(3,3), pad=(1,1))  # (B, C, H, W)
            self.lstm_f   = LSTM(input_dims=512, hidden_dim=128, grad_clipping=100)  # (W, B*H, C)
            self.lstm_b   = LSTM(input_dims=512, hidden_dim=128, grad_clipping=100)  # (W, B*H, C)
            self.conv_cls_score = Conv2D(in_channels=256, out_channels=2*k, kernel_size=1, pad='same')  # (B, 2*k, H, W)
            if do_side_refinement_regress:
                self.conv_bbox_pred = Conv2D(in_channels=256, out_channels=3*k, kernel_size=1, pad='same')  # (B, 3*k, H, W), include side-refinement
            else:
                self.conv_bbox_pred = Conv2D(in_channels=256, out_channels=2*k, kernel_size=1, pad='same')  # (B, 2*k, H, W), no side-refinement

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return: cls_score, bbox  # (B, H, W, k, 2), (B, H, W, n*k) n = 2 or 3
        """
        self.work_mode = 'train'

        #--- encoding part, VGG16, 13 conv layers ---#
        x = relu(self.conv1_1.forward(x))   # (B, 64, 224, 224)
        x = relu(self.conv1_2.forward(x))
        if self.batchnorm_mode == 1:
            x = self.bn1.forward(x)
        x = pool_2d(x, ws=(2, 2))           # (B, 64, 112, 112)
        x = relu(self.conv2_1.forward(x))
        x = relu(self.conv2_2.forward(x))
        if self.batchnorm_mode == 1:
            x = self.bn2.forward(x)
        x = pool_2d(x, ws=(2, 2))           # (B, 128, 56, 56)
        x = relu(self.conv3_1.forward(x))
        x = relu(self.conv3_2.forward(x))
        x = relu(self.conv3_3.forward(x))
        if self.batchnorm_mode == 1:
            x = self.bn3.forward(x)
        x = pool_2d(x, ws=(2, 2))           # (B, 256, 28, 28)
        x = relu(self.conv4_1.forward(x))
        x = relu(self.conv4_2.forward(x))
        x = relu(self.conv4_3.forward(x))
        if self.batchnorm_mode == 1:
            x = self.bn4.forward(x)
        x = pool_2d(x, ws=(2, 2))           # (B, 512, 14, 14)
        x = relu(self.conv5_1.forward(x))
        x = relu(self.conv5_2.forward(x))
        x = relu(self.conv5_3.forward(x))
        if self.batchnorm_mode == 1:
            x = self.bn5.forward(x)
        #--- detection part ---#
        if self.im2col is not None:
            x = self.im2col(x, nb_size=(3,3), merge_channel=True)   # (B, H, W, C*h*w)
            B, H, W, Chw = x.shape
            x = x.dimshuffle((2, 0, 1, 3))
            x = x.reshape((W, -1, Chw))   # (W, B*H, C*h*w)
        else:
            x = relu(self.conv_rpn.forward(x))
            B, C, H, W = x.shape
            x = x.dimshuffle((3, 0, 2, 1)) # (W, B, H, C)
            x = x.reshape((W, -1, C))      # (W, B*H, C)

        x_f = self.lstm_f.forward(x)
        x_b = self.lstm_b.forward(x, backward=True)
        x = tensor.concatenate([x_f, x_b], axis=2)  # (W, B*H, 256)
        x = x.reshape((W, B, H, -1))
        x = x.dimshuffle((1, 3, 2, 0)) # (B, 256, H, W)
        if self.im2col is not None:
            x = relu(self.conv_rpn.forward(x))   # (B, 512, H, W)

        cls_score = self.conv_cls_score.forward(x)  # (B, 2*k, H, W)
        cls_score = cls_score.reshape((B, 2, self.k, H, W)) # (B, 2, k, H, W)
        cls_score = softmax(cls_score.dimshuffle((0, 3, 4, 2, 1)))  # (B, H, W, k, 2)
        bbox = self.conv_bbox_pred.forward(x) # (B, 3*k, H, W), no activation applied
        bbox = bbox.dimshuffle((0, 2, 3, 1, 'x'))  # (B, H, W, 3*k, 1)
        bbox = bbox.reshape((B, H, W, self.k, -1)) # (B, H, W, k, n)

        return cls_score, bbox  # (B, H, W, k, 2), (B, H, W, k, n) n = 2 or 3

    def predict(self, x):
        self.work_mode = 'inference'

        #--- encoding part, VGG16, 13 conv layers ---#
        x = relu(self.conv1_1.predict(x))   # (B, 64, 224, 224)
        x = relu(self.conv1_2.predict(x))
        if self.batchnorm_mode == 1:
            x = self.bn1.predict(x)
        x = pool_2d(x, ws=(2, 2))           # (B, 64, 112, 112)
        x = relu(self.conv2_1.predict(x))
        x = relu(self.conv2_2.predict(x))
        if self.batchnorm_mode == 1:
            x = self.bn2.predict(x)
        x = pool_2d(x, ws=(2, 2))           # (B, 128, 56, 56)
        x = relu(self.conv3_1.predict(x))
        x = relu(self.conv3_2.predict(x))
        x = relu(self.conv3_3.predict(x))
        if self.batchnorm_mode == 1:
            x = self.bn3.predict(x)
        x = pool_2d(x, ws=(2, 2))           # (B, 256, 28, 28)
        x = relu(self.conv4_1.predict(x))
        x = relu(self.conv4_2.predict(x))
        x = relu(self.conv4_3.predict(x))
        if self.batchnorm_mode == 1:
            x = self.bn4.predict(x)
        x = pool_2d(x, ws=(2, 2))           # (B, 512, 14, 14)
        x = relu(self.conv5_1.predict(x))
        x = relu(self.conv5_2.predict(x))
        x = relu(self.conv5_3.predict(x))
        if self.batchnorm_mode == 1:
            x = self.bn5.predict(x)
        #--- detection part ---#
        if self.im2col is not None:
            x = self.im2col(x, nb_size=(3,3), merge_channel=True)   # (B, H, W, C*h*w)
            B, H, W, Chw = x.shape
            x = x.dimshuffle((2, 0, 1, 3))
            x = x.reshape((W, -1, Chw))   # (W, B*H, C*h*w)
        else:
            x = relu(self.conv_rpn.predict(x))
            B, C, H, W = x.shape
            x = x.dimshuffle((3, 0, 2, 1)) # (W, B, H, C)
            x = x.reshape((W, -1, C))      # (W, B*H, C)

        x_f = self.lstm_f.predict(x)
        x_b = self.lstm_b.predict(x, backward=True)
        x = tensor.concatenate([x_f, x_b], axis=2)  # (W, B*H, 256)
        x = x.reshape((W, B, H, -1))
        x = x.dimshuffle((1, 3, 2, 0)) # (B, 256, H, W)
        if self.im2col is not None:
            x = relu(self.conv_rpn.predict(x))   # (B, 512, H, W)

        cls_score = self.conv_cls_score.predict(x)  # (B, 2*k, H, W)
        cls_score = cls_score.reshape((B, 2, self.k, H, W)) # (B, 2, k, H, W)
        cls_score = softmax(cls_score.dimshuffle((0, 3, 4, 2, 1)))  # (B, H, W, k, 2)
        bbox = self.conv_bbox_pred.predict(x) # (B, 3*k, H, W), no activation applied
        bbox = bbox.dimshuffle((0, 2, 3, 1, 'x'))  # (B, H, W, 3*k, 1)
        bbox = bbox.reshape((B, H, W, self.k, -1)) # (B, H, W, k, n)

        return cls_score, bbox  # (B, H, W, k, 2), (B, H, W, k, n) n = 2 or 3