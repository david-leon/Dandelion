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
    def __init__(self, in_channels=256, inner_channels=None, out_channels=None, group_num=4, border_mode='same', batchnorm_mode=1, activation=relu,
                 stride=(1,1), dilation=(1,1), fusion_mode='add'):
        """

        :param in_channels: channel number of block input
        :param inner_channels: channel number inside the block
        :param out_channels: channel number of block output, only used when `fusion_mode` = 'concat', and must > `in_channels`
        :param group_num: number of convolution groups
        :param border_mode: only `same` allowed
        :param batchnorm_mode: {0 | 1 | 2}. 0 means no batch normalization applied; 1 means batch normalization applied to each cnn;
                               2 means batch normalization only applied to the last cnn
        :param activation: default = relu. Note no activation applied to the last element-wise sum output.
        :param stride, dilation: only used for depthwise separable convolution inside
        :param fusion_mode: {'add' | 'concat'}.
        :return y with #channel = `in_channels` when `fusion_mode`='add', #channel = `out_channels` when `fusion_mode`='concat'
        """
        super().__init__()
        self.inner_channels = in_channels//4 if inner_channels is None else inner_channels
        if fusion_mode == 'add':
            self.out_channels = in_channels
        else:
            self.out_channels = out_channels
            if self.out_channels is None or self.out_channels <=in_channels:
                raise ValueError('out_channels must > in_channels')
            self.out_channels -= in_channels
        self.activation     = activation
        self.batchnorm_mode = batchnorm_mode
        self.group_num      = group_num
        self.stride         = as_tuple(stride, 2)
        self.dilation       = as_tuple(dilation, 2)
        self.border_mode    = border_mode
        if border_mode not in {'same'}:
            raise ValueError('Only "same" border mode is supported')
        self.fusion_mode    = fusion_mode
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=self.inner_channels, kernel_size=1, pad=border_mode, num_groups=group_num)
        self.conv2 = DSConv2D(in_channels=self.inner_channels, out_channels=self.inner_channels, kernel_size=3, pad=border_mode, stride=self.stride, dilation=self.dilation)
        self.conv3 = Conv2D(in_channels=self.inner_channels, out_channels=self.out_channels, kernel_size=1, pad=border_mode, num_groups=group_num)
        if batchnorm_mode == 0:   # no batch normalization
            pass
        elif batchnorm_mode == 1: # batch normalization per convolution
            self.bn1 = BatchNorm(input_shape=(None, self.inner_channels, None, None))
            self.bn2 = BatchNorm(input_shape=(None, self.inner_channels, None, None))
            self.bn3 = BatchNorm(input_shape=(None, self.out_channels, None, None))
        elif batchnorm_mode == 2: # only one batch normalization at the end
            self.bn3  = BatchNorm(input_shape=(None, self.out_channels, None, None))
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
        if group_num == 1:
            return x
        x = x.dimshuffle((0, 2, 3, 1)) # (B, H, W, c*g)
        B, H, W, C = x.shape
        x = tensor.reshape(x, (B, H, W, C//group_num, group_num))
        # x = tensor.reshape(x, (B, H, W, group_num, C//group_num,))
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

        if np.max(self.stride) > 1:
            x0 = pool_2d(x0, ws=(3,3), stride=self.stride, pad=[1,1], ignore_border=True, mode='average_exc_pad')
        if self.fusion_mode == 'add':
            x = x0 + x
        elif self.fusion_mode == 'concat':
            x = tensor.concatenate([x0, x], axis=1)
        else:
            raise ValueError('Only "add" or "concat" fusion mode supported')

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

        if np.max(self.stride) > 1:
            x0 = pool_2d(x0, ws=(3,3), stride=self.stride, pad=[1,1], ignore_border=True, mode='average_exc_pad')
        if self.fusion_mode == 'add':
            x = x0 + x
        elif self.fusion_mode == 'concat':
            x = tensor.concatenate([x0, x], axis=1)
        else:
            raise ValueError('Only "add" or "concat" fusion mode supported')

        return x

class ShuffleUnit_Stack(Module):
    """
    Stack of shuffle-unit
    """

    def __init__(self, in_channels, inner_channels=None, out_channels=None, group_num=4, batchnorm_mode=1, activation=relu, stack_size=3):
        """
        :param in_channels: channel number of stack input
        :param inner_channels: channel number inside the shuffle-units
        :param out_channels: channel number of stack output, must > `in_channels`
        :param group_num:
        :param batchnorm_mode:
        :param activation:
        :param stack_size:
        """
        super().__init__()
        self.shuffleunit0 = ShuffleUnit(in_channels=in_channels, inner_channels=inner_channels, out_channels=out_channels, group_num=group_num,
                                        batchnorm_mode=batchnorm_mode, activation=activation, stride=(2,2), fusion_mode='concat')
        self.shuffleunit_stack = [self.shuffleunit0]
        for i in range(stack_size):
            shuffleunit = ShuffleUnit(in_channels=out_channels, inner_channels=inner_channels, group_num=group_num,
                                           batchnorm_mode=batchnorm_mode, activation=activation, fusion_mode='add')
            setattr(self, 'shuffleunit_%d'%(i+1), shuffleunit)

            self.shuffleunit_stack.append(shuffleunit)

    def forward(self, x):
        self.work_mode = 'train'
        for module in self.shuffleunit_stack:
            x = relu(module.forward(x))
        return x

    def predict(self, x):
        self.work_mode = 'inference'
        for module in self.shuffleunit_stack:
            x = relu(module.predict(x))
        return x

class model_ShuffleNet(Module):
    """
    Model reference implementation of [ShuffleNet](https://arxiv.org/abs/1610.02357) without the final Dense layer.
    """
    def __init__(self, in_channels, group_num=4, stage_channels=(24, 272, 544, 1088), stack_size=(3, 7, 3), batchnorm_mode=1, activation=relu):
        super().__init__()
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=stage_channels[0], kernel_size=(3,3), stride=(2,2), pad='same')
        self.bn1   = BatchNorm(input_shape=(None, stage_channels[0], None, None))
        self.stage2 = ShuffleUnit_Stack(in_channels=stage_channels[0], out_channels=stage_channels[1], group_num=1, batchnorm_mode=batchnorm_mode,
                                        activation=activation, stack_size=stack_size[0])
        self.stage3 = ShuffleUnit_Stack(in_channels=stage_channels[1], out_channels=stage_channels[2], group_num=group_num, batchnorm_mode=batchnorm_mode,
                                        activation=activation, stack_size=stack_size[1])
        self.stage4 = ShuffleUnit_Stack(in_channels=stage_channels[2], out_channels=stage_channels[3], group_num=group_num, batchnorm_mode=batchnorm_mode,
                                        activation=activation, stack_size=stack_size[2])

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return:
        """
        self.work_mode = 'train'

        x = self.conv1.forward(x)
        x = relu(self.bn1.forward(x))
        x = pool_2d(x, ws=(3,3), stride=(2,2), mode='max')
        x = relu(self.stage2.forward(x))
        x = relu(self.stage3.forward(x))
        x = relu(self.stage4.forward(x))
        return x


    def predict(self, x):
        """
        :param x: (B, C, H, W)
        :return:
        """
        self.work_mode = 'inference'
    
        x = self.conv1.predict(x)
        x = relu(self.bn1.predict(x))
        x = pool_2d(x, ws=(3, 3), stride=(2, 2), mode='max')
        x = relu(self.stage2.predict(x))
        x = relu(self.stage3.predict(x))
        x = relu(self.stage4.predict(x))
        return x



class model_ShuffleSeg(Module):
    """
    Model reference implementation of [ShuffleSeg](https://arxiv.org/abs/1803.03816)
    todo: unfinished
    """
    def __init__(self, channel=1, im_height=64, im_width=None, Nclass=6, kernel_size=3, border_mode='same',
                 base_n_filters=64, output_activation=log_softmax,
                 noise=(0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)):
        super().__init__()

