"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 08日 星期四 01:39:22 CST
# ***
# ************************************************************************************/
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import pdb


def calc_mean_std(feat, eps: float = 1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


vgg19 = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1,
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
)

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),  # relu4_1
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),  # relu3_1
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),  # relu2_1
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


class AttentionUnit(nn.Module):
    def __init__(self, channels):
        super(AttentionUnit, self).__init__()
        self.relu6 = nn.ReLU6()
        self.f = nn.Conv2d(channels, channels // 2, (1, 1))
        self.g = nn.Conv2d(channels, channels // 2, (1, 1))
        self.h = nn.Conv2d(channels, channels // 2, (1, 1))

        self.out_conv = nn.Conv2d(channels // 2, channels, (1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Fc, Fs):
        B, C, H, W = Fc.shape
        f_Fc = self.relu6(self.f(mean_variance_norm(Fc)))
        g_Fs = self.relu6(self.g(mean_variance_norm(Fs)))
        h_Fs = self.relu6(self.h(Fs))
        f_Fc = f_Fc.view(f_Fc.shape[0], f_Fc.shape[1], -1).permute(0, 2, 1)
        g_Fs = g_Fs.view(g_Fs.shape[0], g_Fs.shape[1], -1)

        Attention = self.softmax(torch.bmm(f_Fc, g_Fs))

        h_Fs = h_Fs.view(h_Fs.shape[0], h_Fs.shape[1], -1)

        Fcs = torch.bmm(h_Fs, Attention.permute(0, 2, 1))
        Fcs = Fcs.view(B, C // 2, H, W)
        Fcs = self.relu6(self.out_conv(Fcs))

        return Fcs


class FuseUnit(nn.Module):
    def __init__(self, channels):
        super(FuseUnit, self).__init__()
        self.proj1 = nn.Conv2d(2 * channels, channels, (1, 1))
        self.proj2 = nn.Conv2d(channels, channels, (1, 1))
        self.proj3 = nn.Conv2d(channels, channels, (1, 1))

        self.fuse1x = nn.Conv2d(channels, 1, (1, 1), stride=1)
        self.fuse3x = nn.Conv2d(channels, 1, (3, 3), stride=1)
        self.fuse5x = nn.Conv2d(channels, 1, (5, 5), stride=1)

        self.pad3x = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pad5x = nn.ReflectionPad2d((2, 2, 2, 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, F1, F2):
        Fcat = self.proj1(torch.cat((F1, F2), dim=1))
        F1 = self.proj2(F1)
        F2 = self.proj3(F2)

        fusion1 = self.sigmoid(self.fuse1x(Fcat))
        fusion3 = self.sigmoid(self.fuse3x(self.pad3x(Fcat)))
        fusion5 = self.sigmoid(self.fuse5x(self.pad5x(Fcat)))
        fusion = (fusion1 + fusion3 + fusion5) / 3.0

        return torch.clamp(fusion, min=0, max=1.0) * F1 + torch.clamp(1.0 - fusion, min=0, max=1.0) * F2


class PAMA(nn.Module):
    def __init__(self, channels):
        super(PAMA, self).__init__()
        self.conv_in = nn.Conv2d(channels, channels, (3, 3), stride=1)
        self.attn = AttentionUnit(channels)
        self.fuse = FuseUnit(channels)
        self.conv_out = nn.Conv2d(channels, channels, (3, 3), stride=1)

        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.relu6 = nn.ReLU6()

    def forward(self, Fc, Fs):
        Fc = self.relu6(self.conv_in(self.pad(Fc)))
        Fs = self.relu6(self.conv_in(self.pad(Fs)))
        Fcs = self.attn(Fc, Fs)
        Fcs = self.relu6(self.conv_out(self.pad(Fcs)))
        Fcs = self.fuse(Fc, Fcs)

        return Fcs


class StyleModel(nn.Module):
    def __init__(self):
        super(StyleModel, self).__init__()

        self.vgg = vgg19[:44]

        self.vgg_0_4 = self.vgg[0:4]
        self.vgg_4_11 = self.vgg[4:11]
        self.vgg_11_18 = self.vgg[11:18]
        self.vgg_18_31 = self.vgg[18:31]
        self.vgg_31_44 = self.vgg[31:44]

        self.align1 = PAMA(512)
        self.align2 = PAMA(512)
        self.align3 = PAMA(512)

        self.decoder = decoder

    def forward(self, Ic, Is):
        # Ic.size(), Is.size() -- [1, 3, 512, 512], [1, 3, 512, 512]
        feat_c = self.forward_vgg(Ic)
        feat_s = self.forward_vgg(Is)
        Fc, Fs = feat_c[3], feat_s[3]

        # Fc.size(), Fs.size() -- [1, 512, 64, 64], [1, 512, 64, 64]
        Fcs1 = self.align1(Fc, Fs)
        Fcs2 = self.align2(Fcs1, Fs)
        Fcs3 = self.align3(Fcs2, Fs)

        return self.decoder(Fcs3)  # -- [1, 3, 512, 512]

    def forward_vgg(self, x) -> List[torch.Tensor]:
        relu1_1 = self.vgg_0_4(x)
        relu2_1 = self.vgg_4_11(relu1_1)
        relu3_1 = self.vgg_11_18(relu2_1)
        relu4_1 = self.vgg_18_31(relu3_1)
        relu5_1 = self.vgg_31_44(relu4_1)

        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]
