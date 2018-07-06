# coding:utf-8
'''
Model definition of U-net FCN
Created   :   5, 24, 2018
Revised   :   5, 24, 2018
All rights reserved
'''
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import theano.tensor as tensor
from ..module import *
from ..initialization import HeNormal 
from ..functional import *
from ..activation import *

class model_Unet(Module):
    """
    Unet reference implementation, same structure with Lasagne's [implementation](https://github.com/Lasagne/Recipes/blob/master/modelzoo/Unet.py)
    Note that Dandelion's `BatchNorm` implementation is different from Lasagne's counter-part.
    `contr` := contract
    """
    def __init__(self, channel=1, im_height=128, im_width=128, Nclass=2, kernel_size=3, border_mode='same', base_n_filters=64, output_activation=softmax):
        super().__init__()
        self.Nclass = Nclass
        self.output_activation = output_activation

        self.contr1_1  = Conv2D(in_channels=channel, out_channels=base_n_filters, kernel_size=kernel_size, pad=border_mode, input_shape=(im_height, im_width), W=HeNormal(gain="relu"))
        self.bn1_1     = BatchNorm(input_shape=(None, base_n_filters, None, None))
        self.contr1_2  = Conv2D(in_channels=base_n_filters, out_channels=base_n_filters, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn1_2     = BatchNorm(input_shape=(None, base_n_filters, None, None))

        self.contr2_1  = Conv2D(in_channels=base_n_filters, out_channels=base_n_filters*2, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn2_1     = BatchNorm(input_shape=(None, base_n_filters*2, None, None))
        self.contr2_2  = Conv2D(in_channels=base_n_filters*2, out_channels=base_n_filters*2, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn2_2     = BatchNorm(input_shape=(None, base_n_filters*2, None, None))

        self.contr3_1  = Conv2D(in_channels=base_n_filters*2, out_channels=base_n_filters*4, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn3_1     = BatchNorm(input_shape=(None, base_n_filters*4, None, None))
        self.contr3_2  = Conv2D(in_channels=base_n_filters*4, out_channels=base_n_filters*4, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn3_2     = BatchNorm(input_shape=(None, base_n_filters*4, None, None))

        self.contr4_1  = Conv2D(in_channels=base_n_filters*4, out_channels=base_n_filters*8, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn4_1     = BatchNorm(input_shape=(None, base_n_filters*8, None, None))
        self.contr4_2  = Conv2D(in_channels=base_n_filters*8, out_channels=base_n_filters*8, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn4_2     = BatchNorm(input_shape=(None, base_n_filters*8, None, None))

        self.encode_1  = Conv2D(in_channels=base_n_filters*8, out_channels=base_n_filters*16, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn_enc_1  = BatchNorm(input_shape=(None, base_n_filters*16, None, None))
        self.encode_2  = Conv2D(in_channels=base_n_filters*16, out_channels=base_n_filters*16, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn_enc_2  = BatchNorm(input_shape=(None, base_n_filters*16, None, None))

        self.upscale_1 = ConvTransposed2D(in_channels=base_n_filters*16, out_channels=base_n_filters*16, kernel_size=2, stride=2, pad='valid', W=HeNormal(gain="relu"))
        self.bn_ups_1  = BatchNorm(input_shape=(None, base_n_filters * 16, None, None))

        self.expand1_1 = Conv2D(in_channels=base_n_filters*24, out_channels=base_n_filters*8, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn_epd1_1 = BatchNorm(input_shape=(None, base_n_filters*8, None, None))
        self.expand1_2 = Conv2D(in_channels=base_n_filters*8, out_channels=base_n_filters*8, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn_epd1_2 = BatchNorm(input_shape=(None, base_n_filters*8, None, None))

        self.upscale_2 = ConvTransposed2D(in_channels=base_n_filters*8, out_channels=base_n_filters*8, kernel_size=2, stride=2, pad='valid', W=HeNormal(gain="relu"))
        self.bn_ups_2  = BatchNorm(input_shape=(None, base_n_filters * 8, None, None))

        self.expand2_1 = Conv2D(in_channels=base_n_filters*12, out_channels=base_n_filters*4, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn_epd2_1 = BatchNorm(input_shape=(None, base_n_filters*4, None, None))
        self.expand2_2 = Conv2D(in_channels=base_n_filters*4, out_channels=base_n_filters*4, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn_epd2_2 = BatchNorm(input_shape=(None, base_n_filters*4, None, None))

        self.upscale_3 = ConvTransposed2D(in_channels=base_n_filters*4, out_channels=base_n_filters*4, kernel_size=2, stride=2, pad='valid', W=HeNormal(gain="relu"))
        self.bn_ups_3  = BatchNorm(input_shape=(None, base_n_filters * 4, None, None))

        self.expand3_1 = Conv2D(in_channels=base_n_filters*6, out_channels=base_n_filters*2, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn_epd3_1 = BatchNorm(input_shape=(None, base_n_filters*2, None, None))
        self.expand3_2 = Conv2D(in_channels=base_n_filters*2, out_channels=base_n_filters*2, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn_epd3_2 = BatchNorm(input_shape=(None, base_n_filters*2, None, None))

        self.upscale_4 = ConvTransposed2D(in_channels=base_n_filters*2, out_channels=base_n_filters*2, kernel_size=2, stride=2, pad='valid', W=HeNormal(gain="relu"))
        self.bn_ups_4  = BatchNorm(input_shape=(None, base_n_filters * 2, None, None))

        self.expand4_1 = Conv2D(in_channels=base_n_filters*3, out_channels=base_n_filters, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn_epd4_1 = BatchNorm(input_shape=(None, base_n_filters, None, None))
        self.expand4_2 = Conv2D(in_channels=base_n_filters, out_channels=base_n_filters, kernel_size=kernel_size, pad=border_mode, W=HeNormal(gain="relu"))
        self.bn_epd4_2 = BatchNorm(input_shape=(None, base_n_filters, None, None))

        self.output    = Conv2D(in_channels=base_n_filters, out_channels=Nclass, kernel_size=1, pad='valid')  # (B, C, H, W)

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return: 
        """
        self.work_mode = 'train'

        x = elu(self.bn1_1.forward(self.contr1_1.forward(x)))
        x_bn1_2 = elu(self.bn1_2.forward(self.contr1_2.forward(x)))
        x = pool_2d(x_bn1_2, ws=(2,2))

        x = elu(self.bn2_1.forward(self.contr2_1.forward(x)))
        x_bn2_2 = elu(self.bn2_2.forward(self.contr2_2.forward(x)))
        x = pool_2d(x_bn2_2, ws=(2,2))

        x = elu(self.bn3_1.forward(self.contr3_1.forward(x)))
        x_bn3_2 = elu(self.bn3_2.forward(self.contr3_2.forward(x)))
        x = pool_2d(x_bn3_2, ws=(2,2))

        x = elu(self.bn4_1.forward(self.contr4_1.forward(x)))
        x_bn4_2 = elu(self.bn4_2.forward(self.contr4_2.forward(x)))
        x = pool_2d(x_bn4_2, ws=(2,2))

        x = elu(self.bn_enc_1.forward(self.encode_1.forward(x)))
        x = elu(self.bn_enc_2.forward(self.encode_2.forward(x)))

        x = elu(self.bn_ups_1.forward(self.upscale_1.forward(x)))
        x1, x2 = align_crop([x, x_bn4_2], cropping=(None, None, 'center', 'center'))
        x = tensor.concatenate([x1, x2], axis=1)

        x = elu(self.bn_epd1_1.forward(self.expand1_1.forward(x)))
        x = elu(self.bn_epd1_2.forward(self.expand1_2.forward(x)))

        x = elu(self.bn_ups_2.forward(self.upscale_2.forward(x)))
        x1, x2 = align_crop([x, x_bn3_2], cropping=(None, None, 'center', 'center'))
        x = tensor.concatenate([x1, x2], axis=1)

        x = elu(self.bn_epd2_1.forward(self.expand2_1.forward(x)))
        x = elu(self.bn_epd2_2.forward(self.expand2_2.forward(x)))

        x = elu(self.bn_ups_3.forward(self.upscale_3.forward(x)))
        x1, x2 = align_crop([x, x_bn2_2], cropping=(None, None, 'center', 'center'))
        x = tensor.concatenate([x1, x2], axis=1)

        x = elu(self.bn_epd3_1.forward(self.expand3_1.forward(x)))
        x = elu(self.bn_epd3_2.forward(self.expand3_2.forward(x)))

        x = elu(self.bn_ups_4.forward(self.upscale_4.forward(x)))
        x1, x2 = align_crop([x, x_bn1_2], cropping=(None, None, 'center', 'center'))
        x = tensor.concatenate([x1, x2], axis=1)

        x = elu(self.bn_epd4_1.forward(self.expand4_1.forward(x)))
        x = elu(self.bn_epd4_2.forward(self.expand4_2.forward(x)))

        x = self.output.forward(x)  # (B, C, H, W)
        x = x.dimshuffle((0, 2, 3, 1))  # (B, H, W, C)
        x = self.output_activation(x)
        return x
 
    def predict(self, x):
        self.work_mode = 'inference'

        x = elu(self.bn1_1.predict(self.contr1_1.predict(x)))
        x_bn1_2 = elu(self.bn1_2.predict(self.contr1_2.predict(x)))
        x = pool_2d(x_bn1_2, ws=(2,2))

        x = elu(self.bn2_1.predict(self.contr2_1.predict(x)))
        x_bn2_2 = elu(self.bn2_2.predict(self.contr2_2.predict(x)))
        x = pool_2d(x_bn2_2, ws=(2,2))

        x = elu(self.bn3_1.predict(self.contr3_1.predict(x)))
        x_bn3_2 = elu(self.bn3_2.predict(self.contr3_2.predict(x)))
        x = pool_2d(x_bn3_2, ws=(2,2))

        x = elu(self.bn4_1.predict(self.contr4_1.predict(x)))
        x_bn4_2 = elu(self.bn4_2.predict(self.contr4_2.predict(x)))
        x = pool_2d(x_bn4_2, ws=(2,2))

        x = elu(self.bn_enc_1.predict(self.encode_1.predict(x)))
        x = elu(self.bn_enc_2.predict(self.encode_2.predict(x)))

        x = elu(self.bn_ups_1.predict(self.upscale_1.predict(x)))
        x1, x2 = align_crop([x, x_bn4_2], cropping=(None, None, 'center', 'center'))
        x = tensor.concatenate([x1, x2], axis=1)

        x = elu(self.bn_epd1_1.predict(self.expand1_1.predict(x)))
        x = elu(self.bn_epd1_2.predict(self.expand1_2.predict(x)))

        x = elu(self.bn_ups_2.predict(self.upscale_2.predict(x)))
        x1, x2 = align_crop([x, x_bn3_2], cropping=(None, None, 'center', 'center'))
        x = tensor.concatenate([x1, x2], axis=1)

        x = elu(self.bn_epd2_1.predict(self.expand2_1.predict(x)))
        x = elu(self.bn_epd2_2.predict(self.expand2_2.predict(x)))

        x = elu(self.bn_ups_3.predict(self.upscale_3.predict(x)))
        x1, x2 = align_crop([x, x_bn2_2], cropping=(None, None, 'center', 'center'))
        x = tensor.concatenate([x1, x2], axis=1)

        x = elu(self.bn_epd3_1.predict(self.expand3_1.predict(x)))
        x = elu(self.bn_epd3_2.predict(self.expand3_2.predict(x)))

        x = elu(self.bn_ups_4.predict(self.upscale_4.predict(x)))
        x1, x2 = align_crop([x, x_bn1_2], cropping=(None, None, 'center', 'center'))
        x = tensor.concatenate([x1, x2], axis=1)

        x = elu(self.bn_epd4_1.predict(self.expand4_1.predict(x)))
        x = elu(self.bn_epd4_2.predict(self.expand4_2.predict(x)))

        x = self.output.predict(x)  # (B, C, H, W)
        x = x.dimshuffle((0, 2, 3, 1))  # (B, H, W, C)
        x = self.output_activation(x)
        return x

