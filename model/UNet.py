import torch
from torch import nn

from model.blocks import Stacked2Block, DownSamplingBlock, UpSamplingBlock


class Unet(nn.Module):
    def __init__(self, num_classes=91 + 91 + 1, block='standart'):
        super(Unet, self).__init__()
        self.init_conv = Stacked2Block(3, 32)

        self.downsample_1 = DownSamplingBlock(32, 64, block)
        self.downsample_2 = DownSamplingBlock(64, 128, block)
        self.downsample_3 = DownSamplingBlock(128, 256, block)
        self.downsample_4 = DownSamplingBlock(256, 512, block)

        # В середине есть блок без пары с 512 каналами
        # с ним конкатенировать некого, потому просто свернём его
        self.upconv = Stacked2Block(512, 256, block)

        # Подъём. Аналогично.

        self.upsample_1 = UpSamplingBlock(256, 128, block)
        self.upsample_2 = UpSamplingBlock(128, 64, block)
        self.upsample_3 = UpSamplingBlock(64, 32, block)
        # Чтобы учесть входной слой после самой первой свёртки
        # и согласовать размерности
        self.upsample_4 = UpSamplingBlock(32, 32, block)

        # Предсказание
        self.agg_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        net0 = self.init_conv(x)  # 3 --> 32

        net1 = self.downsample_1(net0)  # 32 --> 64
        net2 = self.downsample_2(net1)  # 64 --> 128
        net3 = self.downsample_3(net2)  # 128 --> 256
        net = self.downsample_4(net3)  # 256 --> 512

        net = self.upconv(net)  # 512 --> 256

        net = self.upsample_1(net3, net)  # 256 --> 128
        net = self.upsample_2(net2, net)  # 128 --> 64
        net = self.upsample_3(net1, net)  # 64 --> 32
        net = self.upsample_4(net0, net)  # 32 --> 32

        net = self.agg_conv(net)  # 32 --> 183

        return net
