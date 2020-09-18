# Refer to: https://github.com/xiaolai-sqlai/mobilenetv3/blob/master/mobilenetv3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        # self.bneck = nn.Sequential(
        #     Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
        #     Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
        #     Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
        #     Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
        #     Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
        #     Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
        #     Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
        #     Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
        #     Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
        #     Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        #     Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        # )

        self.bneck1=Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2)
        self.bneck2=nn.Sequential(
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
        )
        self.bneck3=nn.Sequential(
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),

        )
        self.bneck4=nn.Sequential(
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))
        c1 = self.bneck1(x)
        c2 = self.bneck2(c1)
        c3 = self.bneck3(c2)
        c4 = self.bneck4(c3)
        # c5 = self.hs2(self.bn2(self.conv2(c4)))

        return c1,c2,c3,c4#,c5


if __name__ =="__main__":
    net = MobileNetV3_Small()
    x = torch.randn(2, 3, 512, 512)
    c1,c2,c3,c4 = net(x)
    print(c1.shape) # 2x16x128x128
    print(c2.shape) # 2x24x64x64
    print(c3.shape) # 2x48x32x32
    print(c4.shape) # 2x96x16x16
    # print(c5.shape)
