# @Time    : 2019-07-16 10:54
# @Author  : Wangzhen
# @Email   : wangzhen@edu.xpu.edu.cn
# @File    : model.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import time
from collections import OrderedDict
from tensorboardX import SummaryWriter

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)


def conv1x1(in_channels, out_channels, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, groups=groups)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def channel_split(x, splits=[24, 24]):
    return torch.split(x, splits, dim=1)


class ParimaryModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=24):
        super(ParimaryModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ParimaryModule = nn.Sequential(
            OrderedDict(
                [
                    ('ParimaryConv', conv3x3(in_channels, out_channels, 2, 1, True, 1)),
                    ('ParimaryBN', nn.BatchNorm2d(out_channels)),
                    ('ParimaryMaxPool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ParimaryModule(x)
        return x

class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, splits_left=2):
        super(ShuffleNetV2Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.splits_left = splits_left

        if stride == 2:
            self.Left = nn.Sequential(
                OrderedDict(
                    [
                        ('DepthwiseConv3x3', conv3x3(in_channels, in_channels, stride, 1, True, in_channels)),
                        ('DepthwiseConv3x3BN', nn.BatchNorm2d(in_channels)),
                        ('UnCompressConv1x1', conv1x1(in_channels, out_channels // 2, True, 1)),
                        ('UnCompressConv1x1BN', nn.BatchNorm2d(out_channels // 2)),
                        ('UnCompressConv1x1ReLU', nn.ReLU())
                    ]
                )
            )
            self.Right = nn.Sequential(
                OrderedDict(
                    [
                        ('NoCompressConv1x1', conv1x1(in_channels, in_channels, True, 1)),
                        ('NoCompressConv1x1BN', nn.BatchNorm2d(in_channels)),
                        ('NoCompressConv1x1ReLU', nn.ReLU()),
                        ('DepthwiseConv3x3', conv3x3(in_channels, in_channels, stride, 1, True, in_channels)),
                        ('DepthwiseConv3x3BN', nn.BatchNorm2d(in_channels)),
                        ('UnCompressConv1x1', conv1x1(in_channels, out_channels // 2, True, 1)),
                        ('UnCompressConv1x1BN', nn.BatchNorm2d(out_channels // 2)),
                        ('UnCompressConv1x1ReLU', nn.ReLU())
                    ]
                )
            )
        elif stride == 1:
            in_channels = in_channels - in_channels // splits_left
            self.Right = nn.Sequential(
                OrderedDict(
                    [
                        ('NoCompressConv1x1', conv1x1(in_channels, in_channels, True, 1)),
                        ('NoCompressConv1x1BN', nn.BatchNorm2d(in_channels)),
                        ('NoCompressConv1x1ReLU', nn.ReLU()),
                        ('DepthwiseConv3x3', conv3x3(in_channels, in_channels, stride, 1, True, in_channels)),
                        ('DepthwiseConv3x3BN', nn.BatchNorm2d(in_channels)),
                        ('UnCompressConv1x1', conv1x1(in_channels, in_channels, True, 1)),
                        ('UnCompressConv1x1BN', nn.BatchNorm2d(in_channels)),
                        ('UnCompressConv1x1ReLU', nn.ReLU())
                    ]
                )
            )
        else:
            raise ValueError('stride must be 1 or 2')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        if self.stride == 2:
            x_left, x_right = x, x
            x_left = self.Left(x_left)
            x_right = self.Right(x_right)
        elif self.stride == 1:
            x_split = channel_split(x, [self.in_channels // self.splits_left,
                                        self.in_channels - self.in_channels // self.splits_left])
            x_left, x_right = x_split[0], x_split[1]
            x_right = self.Right(x_right)

        x = torch.cat((x_left, x_right), dim=1)
        x = channel_shuffle(x, 2)
        return x


class Shuffle_Skip(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, net_scale=1.0, splits_left=2):
        super(Shuffle_Skip, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net_scale = net_scale
        self.splits_left = splits_left

        if net_scale == 0.5:
            self.out_channels = [24, 48, 96, 192, 1024]
        elif net_scale == 1.0:
            self.out_channels = [24, 116, 232, 464, 1024]
        elif net_scale == 1.5:
            self.out_channels = [24, 176, 352, 704, 1024]
        elif net_scale == 2.0:
            self.out_channels = [24, 244, 488, 976, 2048]
        else:
            raise ValueError('net_scale must be 0.5,1.0,1.5 or 2.0')

        self.ParimaryModule = ParimaryModule(in_channels, self.out_channels[0])

        self.Stage1 = self.Stage(1, [1, 3])
        self.Stage2 = self.Stage(2, [1, 3])
        self.Stage3 = self.Stage(3, [1, 3])

        # self.FinalModule = FinalModule(self.out_channels[3], self.out_channels[4], num_classes)


        #decoder
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(96)
        self.deconv2 = nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(48)
        self.deconv3 = nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(16)
        self.deconv5 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def Stage(self, stage=1, BlockRepeat=[1, 3]):
        modules = OrderedDict()
        name = 'ShuffleNetV2Stage{}'.format(stage)

        if BlockRepeat[0] == 1:
            modules[name + '_0'] = ShuffleNetV2Block(self.out_channels[stage - 1], self.out_channels[stage], 2,
                                                     self.splits_left)
        else:
            raise ValueError('stage first block must only repeat 1 time')

        for i in range(BlockRepeat[1]):
            modules[name + '_{}'.format(i + 1)] = ShuffleNetV2Block(self.out_channels[stage], self.out_channels[stage],
                                                                    1, self.splits_left)

        return nn.Sequential(modules)

    def forward(self, x):
        #encoder
        # start=time.time()
        x = self.ParimaryModule(x)
        x1 = self.Stage1(x)#48*32*32 H/8
        x2 = self.Stage2(x1)#96*16*16 H/16
        x3 = self.Stage3(x2)#192*8*8  H/32
        # end = time.time()
        score = self.relu(self.deconv1(x3))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x2)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x1)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)


        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)W)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 64, x.H/2, x.W/2)W)
        # score=torch.nn.functional.interpolate(score, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        score = self.classifier(score)
        # f=time.time()
        # print('time-',f-start,end-start,f-end)
        return score  # size=(N, n_class, x.H/1, x.W/1)


if __name__ == '__main__':
    from torchscope import scope

    net = Shuffle_Skip(3, 1, 0.5)
    scope(net, input_size=(3, 256, 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    dummy_input = torch.rand(3, 3, 256, 256).float().to(device)
    net.cuda()
    net(dummy_input)
    with torch.no_grad():
        start = time.time()
        result = net(dummy_input)
        end = time.time()
        print('time',end-start,'s')

        start = time.time()
        result = net(dummy_input)
        end = time.time()
        print('time',end-start,'s')

        start = time.time()
        result = net(dummy_input)
        end = time.time()
        print('time',end-start,'s')


