import torch
from torch import nn
import torch.nn.functional as F


def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(num_features=out_channels)
    )
    return net


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )
    return net


class ResNetBottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBottleNeckBlock, self).__init__()
        self.constriction = 4

        self.constriction_channels = in_channels // self.constriction

        self.blocks = nn.Sequential(
            conv_bn_relu(in_channels, self.constriction_channels, kernel_size=1),
            conv_bn_relu(self.constriction_channels, self.constriction_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding),
            conv_bn(self.constriction_channels, out_channels, kernel_size=1),
        )

        self.shortcut = conv_bn(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        residual = self.shortcut(x)

        x = self.blocks(x)
        x += residual

        x = F.relu(x)
        return x


class DepthwiseConv2d(torch.nn.Conv2d):
    def __init__(self,
                 in_channels,
                 depth_multiplier=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros'
                 ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )


class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
        super(MobileNetBlock, self).__init__()

        self.net = nn.Sequential(
            DepthwiseConv2d(in_channels, depth_multiplier=out_channels//in_channels,
                            kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class Stacked2Block(nn.Module):
    def __init__(self, in_channels, out_channels, block='standart'):
        super(Stacked2Block, self).__init__()

        if block == 'resnet':
            self.blocks = nn.Sequential(
                conv_bn_relu(in_channels, out_channels, padding=1),
                ResNetBottleNeckBlock(out_channels, out_channels)
            )
        elif block == 'mobilenet':
            self.blocks = nn.Sequential(
                conv_bn_relu(in_channels, out_channels, padding=1),
                MobileNetBlock(out_channels, out_channels)
            )
        elif block == 'standart':
            self.blocks = nn.Sequential(
                conv_bn_relu(in_channels, out_channels, padding=1),
                conv_bn_relu(out_channels, out_channels, padding=1)
            )

    def forward(self, net):
        net = self.blocks(net)
        return net


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block='standart'):
        super(UpSamplingBlock, self).__init__()

        # Понижаем число каналов
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2)

        # Стакаем с симметричным слоем из левой половины "U".
        # Число каналов входной карты при этом удваивается.
        self.convolve = Stacked2Block(2 * in_channels, out_channels, block=block)

    def forward(self, left_net, right_net):
        right_net = self.upsample(right_net)
        net = torch.cat([left_net, right_net], dim=1)
        net = self.convolve(net)
        return net


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block='standart'):
        super(DownSamplingBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Stacked2Block(in_channels, out_channels, block=block)
        )

    def forward(self, net):
        return self.blocks(net)
