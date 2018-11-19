# coding:utf-8
'''
Reference implementation of [Shuffle-net](https://arxiv.org/abs/1707.01083) and other related models in paper.
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
    Due to a bug in Theano's convolution, you'll need newer version of Theano
    after this [commit](https://github.com/Theano/Theano/commit/727477b5c3cfb4352f18b07ea4e4cb6df95cc254)
    to run this model.
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
        x = channel_shuffle(x, self.group_num)
        x  = self.conv2.forward(x)
        if self.batchnorm_mode == 1:
            x = self.bn2.forward(x)
        x = self.activation(x)
        x  = self.conv3.forward(x)
        if self.batchnorm_mode in {1, 2}:
            x = self.bn3.forward(x)
        x = self.activation(x)

        if np.max(self.stride) > 1:
            x0 = pool_2d(x0, ws=(3,3), stride=self.stride, pad=[1,1], ignore_border=True, mode='average_inc_pad')
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
        x = channel_shuffle(x, self.group_num)
        x  = self.conv2.predict(x)
        if self.batchnorm_mode == 1:
            x = self.bn2.predict(x)
        x = self.activation(x)
        x  = self.conv3.predict(x)
        if self.batchnorm_mode in {1, 2}:
            x = self.bn3.predict(x)
        x = self.activation(x)

        if np.max(self.stride) > 1:
            x0 = pool_2d(x0, ws=(3, 3), stride=self.stride, pad=[1, 1], ignore_border=True, mode='average_inc_pad')
        if self.fusion_mode == 'add':
            x = x0 + x
        elif self.fusion_mode == 'concat':
            x = tensor.concatenate([x0, x], axis=1)
        else:
            raise ValueError('Only "add" or "concat" fusion mode supported')

        return x

class ShuffleUnit_v2(Module):
    """
    Shuffle unit v2 reference implementation (https://arxiv.org/abs/1807.11164)
    """
    def __init__(self, in_channels=256, out_channels=None, border_mode='same', batchnorm_mode=1, activation=relu,
                 stride=1, dilation=1):
        """
        :param in_channels: channel number of block input
        :param out_channels: channel number of block output, only used when `stride` > 1.
        :param border_mode: only `same` allowed
        :param batchnorm_mode: {0 | 1 | 2}. 0 means no batch normalization applied; 1 means batch normalization applied to each cnn;
                               2 means batch normalization only applied to the last cnn
        :param activation: default = relu. Note no activation applied to the last element-wise sum output.
        :param stride, dilation: only used for depthwise separable convolution inside, must be integer scalars
        :return y with #channel = `in_channels` when `fusion_mode`='add', #channel = `out_channels` when `fusion_mode`='concat'
        """
        super().__init__()
        self.activation     = activation
        self.batchnorm_mode = batchnorm_mode
        self.stride         = as_tuple(stride, 2)
        self.dilation       = as_tuple(dilation, 2)
        self.border_mode    = border_mode
        channels = out_channels//2
        assert 2*channels == out_channels, "out_channels must be even"
        if out_channels is None:
            out_channels = in_channels
        if stride == 1:
            assert in_channels == out_channels, 'when stride=1, out_channels must be equal to in_channels'
            self.conv1 = Conv2D(in_channels=channels, out_channels=channels, kernel_size=1, pad=border_mode)
            self.conv2 = DSConv2D(in_channels=channels, out_channels=channels, kernel_size=3, pad=border_mode, dilation=self.dilation)
            self.conv3 = Conv2D(in_channels=channels, out_channels=channels, kernel_size=1, pad=border_mode)
        else:
            self.conv1 = Conv2D(in_channels=in_channels, out_channels=channels, kernel_size=1, pad=border_mode)
            self.conv2 = DSConv2D(in_channels=channels, out_channels=channels, kernel_size=3, pad=border_mode, stride=self.stride, dilation=self.dilation)
            self.conv3 = Conv2D(in_channels=channels, out_channels=channels, kernel_size=1, pad=border_mode)
            self.conv4 = DSConv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=3, pad=border_mode, stride=self.stride, dilation=self.dilation)
            self.conv5 = Conv2D(in_channels=in_channels, out_channels=channels, kernel_size=1, pad=border_mode)

        if batchnorm_mode == 0:  # no batch normalization
            pass
        elif batchnorm_mode == 1:  # batch normalization per convolution
            self.bn1 = BatchNorm(input_shape=(None, channels, None, None))
            self.bn2 = BatchNorm(input_shape=(None, channels, None, None))
            self.bn3 = BatchNorm(input_shape=(None, channels, None, None))
            if stride > 1:
                self.bn4 = BatchNorm(input_shape=(None, in_channels, None, None))
                self.bn5 = BatchNorm(input_shape=(None, channels, None, None))
        elif batchnorm_mode == 2:  # only one batch normalization at the end
            self.bn3 = BatchNorm(input_shape=(None, channels, None, None))
            if stride > 1:
                self.bn5 = BatchNorm(input_shape=(None, channels, None, None))
        else:
            raise ValueError('batchnorm_mode should be {0 | 1 | 2}')

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return:
        """
        self.work_mode = 'train'

        if self.stride[0] == 1:
            x1 = x[:, x.shape[1]//2:, :, :]   # the first half channels
            x2 = x[:, :x.shape[1]//2, :, :]   # the last half channels
            x1 = self.conv1.forward(x1)
            if self.batchnorm_mode == 1:
                x1 = self.bn1.forward(x1)
            x1 = self.activation(x1)
            x1 = self.conv2.forward(x1)
            if self.batchnorm_mode == 1:
                x1 = self.bn2.forward(x1)
            x1 = self.conv3.forward(x1)
            if self.batchnorm_mode in {1, 2}:
                x1 = self.bn3.forward(x1)
            x1 = self.activation(x1)
        else:
            x1 = self.conv1.forward(x)
            if self.batchnorm_mode == 1:
                x1 = self.bn1.forward(x1)
            x1 = self.activation(x1)
            x1 = self.conv2.forward(x1)
            if self.batchnorm_mode == 1:
                x1 = self.bn2.forward(x1)
            x1 = self.conv3.forward(x1)
            if self.batchnorm_mode in {1, 2}:
                x1 = self.bn3.forward(x1)
            x1 = self.activation(x1)

            x2 = self.conv4.forward(x)
            if self.batchnorm_mode == 1:
                x2 = self.bn4.forward(x2)
            x2 = self.conv5.forward(x2)
            if self.batchnorm_mode in {1, 2}:
                x2 = self.bn5.forward(x2)
            x2 = self.activation(x2)

        x = tensor.concatenate([x1, x2], axis=1)
        x = channel_shuffle(x, 2)
        return x

    def predict(self, x):
        self.work_mode = 'inference'

        if self.stride[0] == 1:
            x1 = x[:, x.shape[1]//2:, :, :]   # the first half channels
            x2 = x[:, :x.shape[1]//2, :, :]   # the last half channels
            x1 = self.conv1.predict(x1)
            if self.batchnorm_mode == 1:
                x1 = self.bn1.predict(x1)
            x1 = self.activation(x1)
            x1 = self.conv2.predict(x1)
            if self.batchnorm_mode == 1:
                x1 = self.bn2.predict(x1)
            x1 = self.conv3.predict(x1)
            if self.batchnorm_mode in {1, 2}:
                x1 = self.bn3.predict(x1)
            x1 = self.activation(x1)
        else:
            x1 = self.conv1.predict(x)
            if self.batchnorm_mode == 1:
                x1 = self.bn1.predict(x1)
            x1 = self.activation(x1)
            x1 = self.conv2.predict(x1)
            if self.batchnorm_mode == 1:
                x1 = self.bn2.predict(x1)
            x1 = self.conv3.predict(x1)
            if self.batchnorm_mode in {1, 2}:
                x1 = self.bn3.predict(x1)
            x1 = self.activation(x1)

            x2 = self.conv4.predict(x)
            if self.batchnorm_mode == 1:
                x2 = self.bn4.predict(x2)
            x2 = self.conv5.predict(x2)
            if self.batchnorm_mode in {1, 2}:
                x2 = self.bn5.predict(x2)
            x2 = self.activation(x2)

        x = tensor.concatenate([x1, x2], axis=1)
        x = channel_shuffle(x, 2)
        return x

class ShuffleUnit_Stack(Module):
    """
    Stack of shuffle-unit
    downscale factor = (2,2)
    """

    def __init__(self, in_channels, out_channels, inner_channels=None, group_num=4, batchnorm_mode=1, activation=relu, stack_size=3, stride=2, fusion_mode='concat'):
        """
        :param in_channels: channel number of stack input
        :param inner_channels: channel number inside the shuffle-units
        :param out_channels: channel number of stack output, must > `in_channels`
        :param group_num:
        :param batchnorm_mode:
        :param activation:
        :param stack_size:
        :param stride: int or tuple of int, convolution stride for the first unit, default=2
        :param fusion_mode: fusion_mode for the first unit.
        """
        super().__init__()
        self.shuffleunit0 = ShuffleUnit(in_channels=in_channels, inner_channels=inner_channels, out_channels=out_channels, group_num=group_num,
                                        batchnorm_mode=batchnorm_mode, activation=activation, stride=as_tuple(stride, 2), fusion_mode=fusion_mode)
        self.shuffleunit_stack = [self.shuffleunit0]
        for i in range(stack_size):
            shuffleunit = ShuffleUnit(in_channels=out_channels, inner_channels=inner_channels, group_num=group_num,
                                           batchnorm_mode=batchnorm_mode, activation=activation, fusion_mode='add')
            setattr(self, 'shuffleunit%d'%(i+1), shuffleunit)

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

class ShuffleUnit_v2_Stack(Module):
    """
    Stack of shuffle_v2 unit
    downscale factor = (2,2)
    """
    def __init__(self, in_channels, out_channels, batchnorm_mode=1, activation=relu, stack_size=3, stride=2):
        """
        :param in_channels: channel number of stack input
        :param out_channels: channel number of stack output
        :param batchnorm_mode:
        :param activation:
        :param stack_size:
        :param stride: int or tuple of int, convolution stride for the first stack, default=2
        """
        super().__init__()
        self.shuffleunit0 = ShuffleUnit_v2(in_channels=in_channels, out_channels=out_channels, batchnorm_mode=batchnorm_mode,
                                           activation=activation, stride=stride)
        self.shuffleunit_stack = [self.shuffleunit0]
        for i in range(stack_size):
            shuffleunit = ShuffleUnit_v2(in_channels=out_channels, out_channels=out_channels, batchnorm_mode=batchnorm_mode,
                                         activation=activation, stride=1)
            setattr(self, 'shuffleunit%d'%(i+1), shuffleunit)

            self.shuffleunit_stack.append(shuffleunit)

    def forward(self, x):
        self.work_mode = 'train'
        for module in self.shuffleunit_stack:
            x = module.forward(x)
        return x

    def predict(self, x):
        self.work_mode = 'inference'
        for module in self.shuffleunit_stack:
            x = module.predict(x)
        return x

class model_ShuffleNet(Module):
    """
    Model reference implementation of [ShuffleNet](https://arxiv.org/abs/1610.02357) without the final pooling & Dense layer.
    Note no activation applied to the last output
    downscale factor = (32, 32)
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
        x = self.stage4.forward(x)
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
        x = self.stage4.predict(x)
        return x

class model_ShuffleNet_v2(Module):
    """
    Model reference implementation of [ShuffleNet v2](https://arxiv.org/abs/1807.11164) without the final pooling & Dense layer.
    Note no activation applied to the last output
    """

    def __init__(self, in_channels, stage_channels=(24, 116, 232, 464, 1024), stack_size=(3, 7, 3), batchnorm_mode=1, activation=relu):
        super().__init__()
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=stage_channels[0], kernel_size=(3, 3), stride=(2, 2), pad='same')
        self.bn1   = BatchNorm(input_shape=(None, stage_channels[0], None, None))
        self.stage2 = ShuffleUnit_v2_Stack(in_channels=stage_channels[0], out_channels=stage_channels[1], stack_size=stack_size[0], activation=activation, batchnorm_mode=batchnorm_mode)
        self.stage3 = ShuffleUnit_v2_Stack(in_channels=stage_channels[1], out_channels=stage_channels[2], stack_size=stack_size[1], activation=activation, batchnorm_mode=batchnorm_mode)
        self.stage4 = ShuffleUnit_v2_Stack(in_channels=stage_channels[2], out_channels=stage_channels[3], stack_size=stack_size[2], activation=activation, batchnorm_mode=batchnorm_mode)
        self.conv5  = Conv2D(in_channels=stage_channels[3], out_channels=stage_channels[4], kernel_size=(1,1), pad='same')
        self.bn5    = BatchNorm(input_shape=(None, stage_channels[4], None, None))
    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return:
        """
        self.work_mode = 'train'

        x = self.conv1.forward(x)
        x = relu(self.bn1.forward(x))
        x = pool_2d(x, ws=(3, 3), stride=(2, 2), mode='max')
        x = self.stage2.forward(x)
        x = self.stage3.forward(x)
        x = self.stage4.forward(x)
        x = self.conv5.forward(x)
        x = self.bn5.forward(x)
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
        x = self.stage2.predict(x)
        x = self.stage3.predict(x)
        x = self.stage4.predict(x)
        x = self.conv5.predict(x)
        x = self.bn5.predict(x)
        return x

class model_ShuffleSeg(Module):
    """
    Model reference implementation of [ShuffleSeg](https://arxiv.org/abs/1803.03816)
    """
    class ShuffleNet(model_ShuffleNet):
        """
        Same with model_ShuffleNet. Output two more heads: feed1 & feed2
        """
        def __init__(self, in_channels, out_channels, group_num=4, stage_channels=(24, 272, 544, 1088), stack_size=(3, 7, 3), batchnorm_mode=1, activation=relu):
            super().__init__(in_channels=in_channels, group_num=group_num, stage_channels=stage_channels,
                             stack_size=stack_size, batchnorm_mode=batchnorm_mode, activation=activation)
            self.conv2 = Conv2D(in_channels=stage_channels[3], out_channels=out_channels, kernel_size=(1,1), pad='same')
       
        def forward(self, x):
            """
             :param x: (B, C, H, W)
             :return:
             """
            self.work_mode = 'train'

            x = self.conv1.forward(x)
            x = relu(self.bn1.forward(x))
            x = pool_2d(x, ws=(3, 3), stride=(2, 2), mode='max')
            x = relu(self.stage2.forward(x))
            feed2 = x
            x = relu(self.stage3.forward(x))
            feed1 = x
            x = relu(self.stage4.forward(x))
            x = self.conv2.forward(x)
            return x, feed1, feed2  # (H, W) = [(7,7), (14, 14), (28, 28)] for (224, 224) input
        
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
            feed2 = x
            x = relu(self.stage3.predict(x))
            feed1 = x
            x = relu(self.stage4.predict(x))
            x = self.conv2.predict(x)
            return x, feed1, feed2

    def __init__(self, in_channels=1, Nclass=6, SF_group_num=4, SF_stage_channels=(24, 272, 544, 1088), SF_stack_size=(3, 7, 3), SF_batchnorm_mode=1, SF_activation=relu):
        super().__init__()
        self.encoder = self.ShuffleNet(in_channels=in_channels, out_channels=Nclass, group_num=SF_group_num, stage_channels=SF_stage_channels,
                                       stack_size=SF_stack_size, batchnorm_mode=SF_batchnorm_mode, activation=SF_activation)
        #--- decoder part ---#
        self.convT1  = ConvTransposed2D(in_channels=Nclass, out_channels=Nclass, kernel_size=(4,4), stride=(2,2), pad=(1,1))
        self.bn1     = BatchNorm(input_shape=(None, Nclass, None, None))
        self.conv1   = Conv2D(in_channels=SF_stage_channels[2], out_channels=Nclass, kernel_size=(1,1), pad='same')
        self.bn2     = BatchNorm(input_shape=(None, Nclass, None, None))
        self.convT2  = ConvTransposed2D(in_channels=Nclass, out_channels=Nclass, kernel_size=(4,4), stride=(2,2), pad=(1,1))
        self.bn3     = BatchNorm(input_shape=(None, Nclass, None, None))
        self.conv2   = Conv2D(in_channels=SF_stage_channels[1], out_channels=Nclass, kernel_size=(1,1), pad='same')
        self.bn4     = BatchNorm(input_shape=(None, Nclass, None, None))
        self.convT3  = ConvTransposed2D(in_channels=Nclass, out_channels=Nclass, kernel_size=(16,16), stride=(8,8), pad=(4,4))

    def forward(self, x):
        """
        No activation applied in decoder part, same with the reference paper.
        :param x:
        :return:
        """
        self.work_mode = 'train'

        encoder_out, feed1, feed2 = self.encoder.forward(x)
        upscore2    = self.bn1.forward(self.convT1.forward(encoder_out))
        score_feed1 = self.bn2.forward(self.conv1.forward(feed1))
        fuse_feed1  = score_feed1 + upscore2
        upscore4    = self.bn3.forward(self.convT2.forward(fuse_feed1))
        score_feed2 = self.bn4.forward(self.conv2.forward(feed2))
        fuse_feed2  = score_feed2 + upscore4
        x           = self.convT3.forward(fuse_feed2)
        return x

    def predict(self, x):
        self.work_mode = 'inference'

        encoder_out, feed1, feed2 = self.encoder.predict(x)
        upscore2    = self.bn1.predict(self.convT1.predict(encoder_out))
        score_feed1 = self.bn2.predict(self.conv1.predict(feed1))
        fuse_feed1  = score_feed1 + upscore2
        upscore4    = self.bn3.predict(self.convT2.predict(fuse_feed1))
        score_feed2 = self.bn4.predict(self.conv2.predict(feed2))
        fuse_feed2  = score_feed2 + upscore4
        x           = self.convT3.predict(fuse_feed2)
        return x







