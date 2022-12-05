import numpy as np
import torch
from torch import nn


class ConvUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bn: bool = True,
                 act=nn.ReLU
                 ):
        super(ConvUnit, self).__init__()
        if bn:
            bias = False
        else:
            bias = True

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Sequential(),
            act(inplace=True) if isinstance(act, nn.Module) else nn.Sequential()
        )

    def forward(self, x):
        return self.layer(x)


# # 全连接实现
# class SEModule(nn.Module):
#     def __init__(self, in_channels, ratio=4):
#         super(SEModule, self).__init__()
#         self.in_channels = in_channels
#         hidden_channels = in_channels // ratio
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.excitation_layer = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_channels, in_channels),
#             nn.Hardsigmoid(inplace=True)
#         )
#
#     def forward(self, x):
#         out_ = self.pool(x)
#         out_ = out_.view(-1, self.in_channels)
#         return self.excitation_layer(out_).view(-1, self.in_channels, 1, 1) * x


# 卷积实现
class SEModule(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(SEModule, self).__init__()
        hidden_channels = in_channels // ratio
        # self.squeeze_layer = nn.AdaptiveAvgPool2d(1)
        self.excitation_layer = nn.Sequential(
            ConvUnit(in_channels, hidden_channels, kernel_size=1, stride=1),
            ConvUnit(hidden_channels, in_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        out_ = ConvUnit(c, c, (h, w), stride=1)(x)
        return self.excitation_layer(out_) * x


class GhostModule(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 ratio=2,
                 dw_size=3,
                 act=nn.ReLU
                 ):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        hidden_channels = np.ceil(out_channels / ratio).astype(np.int8)
        self.primary_layer = ConvUnit(in_channels, hidden_channels, kernel_size, stride=stride,
                                      padding=kernel_size // 2, act=act)
        self.cheap_layer = ConvUnit(hidden_channels, hidden_channels, dw_size, stride=stride, padding=dw_size // 2,
                                    groups=hidden_channels, act=act)

    def forward(self, x):
        out1 = self.primary_layer(x)
        out2 = self.cheap_layer(out1)
        out = torch.cat([out1, out2], dim=1)

        return out[:, :self.out_channels, :, :]


class DFC(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=1,
                 scale=2):
        super(DFC, self).__init__()
        self.dfc_layer = nn.Sequential(
            nn.AvgPool2d(scale, scale),
            ConvUnit(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, act=False),
            ConvUnit(out_channels, out_channels, kernel_size=(1, 5), stride=1, padding=(0, 2), act=False),
            ConvUnit(out_channels, out_channels, kernel_size=(5, 1), stride=1, padding=(2, 0), act=False),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=scale)
        )

    def forward(self, x):
        return self.dfc_layer(x)


class GhostV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, stride=1, use_se=False):
        super().__init__()
        assert stride in [1, 2]
        self.ghost1 = GhostModule(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.dfc_layer = DFC(in_channels, hidden_channels)

        self.ghost2 = GhostModule(hidden_channels, out_channels, kernel_size=1, stride=1, act=False)

        if stride == 1:
            self.dw_layer = nn.Sequential()

        else:
            self.dw_layer = ConvUnit(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride,
                                     padding=kernel_size // 2, groups=hidden_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                ConvUnit(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                         padding=kernel_size // 2, groups=in_channels, act=False),
                ConvUnit(in_channels, out_channels, kernel_size=1, stride=1, act=False)
            )

        if use_se:
            self.se_layer = SEModule(hidden_channels)
        else:
            self.se_layer = nn.Sequential()

    def forward(self, x):
        g_out = self.ghost1(x)
        dfc_out = self.dfc_layer(x)
        out = torch.mul(g_out, dfc_out)
        out = self.se_layer(out)

        shortcut = self.shortcut(x)

        out = self.ghost2(self.dw_layer(out))
        return torch.add(shortcut, out)


if __name__ == '__main__':
    data = torch.randn(2, 32, 224, 224)
    net = GhostV2Block(32, 32, 16, 3, use_se=True)
    out = net(data)
    print(out.shape)
