import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.layers import DeformUnfold
from maskrcnn_benchmark.layers import DeformConv
# import scipy.io as io
# import numpy as np


class DynamicWeights(nn.Module):
    def __init__(self, channels):
        super(DynamicWeights, self).__init__()
        self.cata = nn.Conv2d(channels, 9, 3, padding=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.unfold1 = nn.Unfold(kernel_size=(3, 3), padding=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        blur_depth = x
        dynamic_filter1 = self.cata(blur_depth)
        N, C, H, W = blur_depth.size()

        dynamic_filter1 = self.softmax(dynamic_filter1.permute(0, 2, 3, 1).contiguous().view(N * H * W, -1))
        xd_unfold1 = self.unfold1(blur_depth)

        xd_unfold1 = xd_unfold1.contiguous().view(N, blur_depth.size()[1], 9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W, blur_depth.size()[1], 9)

        out1 = torch.bmm(xd_unfold1, dynamic_filter1.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out1 = out1.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2).contiguous()  # N, xd.size()[1], H, W

        out = self.gamma*out1 + x

        return out


class DynamicWeightsCat11(nn.Module):
    r'''' not a rigrous implementation but faster speed

    '''''

    def __init__(self, channels, group=1, kernel=3, dilation=(1, 4, 8, 12), shuffle=False, deform=None):
        super(DynamicWeightsCat11, self).__init__()
        in_channel = channels // 4
        self.scale1 = nn.Sequential(nn.Conv2d(channels, in_channel, 1, padding=0, bias=False),
                                    group_norm(in_channel),
                                    nn.ReLU(inplace=True))

        if deform == 'deformatt':
            self.cata = nn.Conv2d(in_channel, group * kernel * kernel + 27, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.catb = nn.Conv2d(in_channel, group * kernel * kernel + 27, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.catc = nn.Conv2d(in_channel, group * kernel * kernel + 27, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.catd = nn.Conv2d(in_channel, group * kernel * kernel + 27, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = DeformUnfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])
        elif deform == 'deform':
            self.cata = nn.Conv2d(in_channel, group * kernel * kernel + 18, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.catb = nn.Conv2d(in_channel, group * kernel * kernel + 18, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.catc = nn.Conv2d(in_channel, group * kernel * kernel + 18, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.catd = nn.Conv2d(in_channel, group * kernel * kernel + 18, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = DeformUnfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])
        else:
            self.cata = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.catb = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.catc = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.catd = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = nn.Unfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = nn.Unfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = nn.Unfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = nn.Unfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])

        self.softmax = nn.Softmax(dim=-1)

        self.shuffle = shuffle
        self.deform = deform
        self.group = group
        self.K = kernel * kernel

        self.scale2 = nn.Sequential(nn.Conv2d(in_channel * 5, in_channel, 1, padding=0, bias=False),
                                    group_norm(in_channel),
                                    nn.ReLU(inplace=True))

        self.scale3 = nn.Sequential(nn.Conv2d(in_channel, channels, 1, padding=0, bias=False),
                                    group_norm(channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        xd = self.scale1(x)
        blur_depth = xd

        N, C, H, W = xd.size()
        R = C // self.group

        if self.deform == 'deformatt':
            dynamic_filter_offset_att1 = self.cata(blur_depth)
            dynamic_filter_offset_att2 = self.catb(blur_depth)
            dynamic_filter_offset_att3 = self.catc(blur_depth)
            dynamic_filter_offset_att4 = self.catd(blur_depth)

            dynamic_filter1 = dynamic_filter_offset_att1[:, :9 * self.group, :, :]
            offset1 = dynamic_filter_offset_att1[:, 9 * self.group:9 * self.group+18, :, :]
            att1 = dynamic_filter_offset_att1[:, -9:, :, :]  # N, 9, H, W
            att1 = att1.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            att1 = att1.sigmoid()

            dynamic_filter2 = dynamic_filter_offset_att2[:, :9 * self.group, :, :]
            offset2 = dynamic_filter_offset_att2[:, 9 * self.group:9 * self.group + 18, :, :]
            att2 = dynamic_filter_offset_att2[:, -9:, :, :]  # N, 9, H, W
            att2 = att2.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            att2 = att2.sigmoid()

            dynamic_filter3 = dynamic_filter_offset_att3[:, :9 * self.group, :, :]
            offset3 = dynamic_filter_offset_att3[:, 9 * self.group:9 * self.group + 18, :, :]
            att3 = dynamic_filter_offset_att3[:, -9:, :, :]  # N, 9, H, W
            att3 = att3.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            att3 = att3.sigmoid()

            dynamic_filter4 = dynamic_filter_offset_att4[:, :9 * self.group, :, :]
            offset4 = dynamic_filter_offset_att4[:, 9 * self.group:9 * self.group + 18, :, :]
            att4 = dynamic_filter_offset_att4[:, -9:, :, :]  # N, 9, H, W
            att4 = att4.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            att4 = att4.sigmoid()

        elif self.deform == 'deform':
            dynamic_filter_offset1 = self.cata(blur_depth)
            dynamic_filter_offset2 = self.catb(blur_depth)
            dynamic_filter_offset3 = self.catc(blur_depth)
            dynamic_filter_offset4 = self.catd(blur_depth)

            dynamic_filter1 = dynamic_filter_offset1[:, :9, :, :]
            offset1 = dynamic_filter_offset1[:, -18:, :, :]

            dynamic_filter2 = dynamic_filter_offset2[:, :9, :, :]
            offset2 = dynamic_filter_offset2[:, -18:, :, :]

            dynamic_filter3 = dynamic_filter_offset3[:, :9, :, :]
            offset3 = dynamic_filter_offset3[:, -18:, :, :]

            dynamic_filter4 = dynamic_filter_offset4[:, :9, :, :]
            offset4 = dynamic_filter_offset4[:, -18:, :, :]
        else:
            dynamic_filter1 = self.cata(blur_depth)
            dynamic_filter2 = self.catb(blur_depth)
            dynamic_filter3 = self.catc(blur_depth)
            dynamic_filter4 = self.catd(blur_depth)

        dynamic_filter1 = self.softmax(dynamic_filter1.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
        dynamic_filter2 = self.softmax(dynamic_filter2.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
        dynamic_filter3 = self.softmax(dynamic_filter3.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
        dynamic_filter4 = self.softmax(dynamic_filter4.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)

        if self.training and self.shuffle:
            dynamic_filter1 = dynamic_filter1.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx1 = torch.randperm(self.group)
            dynamic_filter1 = dynamic_filter1[idx1].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter2 = dynamic_filter2.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx2 = torch.randperm(self.group)
            dynamic_filter2 = dynamic_filter2[idx2].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter3 = dynamic_filter3.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx3 = torch.randperm(self.group)
            dynamic_filter3 = dynamic_filter3[idx3].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter4 = dynamic_filter4.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx4 = torch.randperm(self.group)
            dynamic_filter4 = dynamic_filter4[idx4].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

        if self.deform == 'none':
            xd_unfold1 = self.unfold1(blur_depth)
            xd_unfold2 = self.unfold2(blur_depth)
            xd_unfold3 = self.unfold3(blur_depth)
            xd_unfold4 = self.unfold4(blur_depth)
        else:
            xd_unfold1 = self.unfold1(blur_depth, offset1)
            xd_unfold2 = self.unfold2(blur_depth, offset2)
            xd_unfold3 = self.unfold3(blur_depth, offset3)
            xd_unfold4 = self.unfold4(blur_depth, offset4)

        if self.deform == 'deformatt':
            xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W)
            xd_unfold2 = xd_unfold2.view(N, C, self.K, H * W)
            xd_unfold3 = xd_unfold3.view(N, C, self.K, H * W)
            xd_unfold4 = xd_unfold4.view(N, C, self.K, H * W)

            xd_unfold1 *= att1
            xd_unfold2 *= att2
            xd_unfold3 *= att3
            xd_unfold4 *= att4

            xd_unfold1 = xd_unfold1.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold2 = xd_unfold2.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold3 = xd_unfold3.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold4 = xd_unfold4.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
        else:
            xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold2 = xd_unfold2.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold3 = xd_unfold3.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold4 = xd_unfold4.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)

        out1 = torch.bmm(xd_unfold1, dynamic_filter1.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out1 = out1.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out2 = torch.bmm(xd_unfold2, dynamic_filter2.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out2 = out2.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out3 = torch.bmm(xd_unfold3, dynamic_filter3.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out3 = out3.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out4 = torch.bmm(xd_unfold4, dynamic_filter4.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out4 = out4.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)

        out = self.scale3(self.scale2(torch.cat((xd, out1, out2, out3, out4), 1))) + x

        return out


class DynamicWeightsCat33(nn.Module):
    r'''' not a rigrous implementation but faster speed

    '''''

    def __init__(self, channels, group=1, kernel=3, dilation=(1, 4, 8, 12), shuffle=False, deform=None):
        super(DynamicWeightsCat33, self).__init__()
        in_channel = channels

        if deform == 'deformatt':
            self.cata = nn.Conv2d(in_channel, group * kernel * kernel + 27, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.catb = nn.Conv2d(in_channel, group * kernel * kernel + 27, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.catc = nn.Conv2d(in_channel, group * kernel * kernel + 27, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.catd = nn.Conv2d(in_channel, group * kernel * kernel + 27, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = DeformUnfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])
        elif deform == 'deform':
            self.cata = nn.Conv2d(in_channel, group * kernel * kernel + 18, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.catb = nn.Conv2d(in_channel, group * kernel * kernel + 18, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.catc = nn.Conv2d(in_channel, group * kernel * kernel + 18, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.catd = nn.Conv2d(in_channel, group * kernel * kernel + 18, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = DeformUnfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])
        else:
            self.cata = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.catb = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.catc = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.catd = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = nn.Unfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = nn.Unfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = nn.Unfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = nn.Unfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])

        self.softmax = nn.Softmax(dim=-1)

        self.shuffle = shuffle
        self.deform = deform
        self.group = group
        self.K = kernel * kernel

        self.scale2 = nn.Sequential(nn.Conv2d(in_channel * 5, in_channel, 1, padding=0, bias=True),
                                    group_norm(in_channel),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        blur_depth = x

        N, C, H, W = x.size()
        R = C // self.group

        if self.deform == 'deformatt':
            dynamic_filter_offset_att1 = self.cata(blur_depth)
            dynamic_filter_offset_att2 = self.catb(blur_depth)
            dynamic_filter_offset_att3 = self.catc(blur_depth)
            dynamic_filter_offset_att4 = self.catd(blur_depth)

            dynamic_filter1 = dynamic_filter_offset_att1[:, :9 * self.group, :, :]
            offset1 = dynamic_filter_offset_att1[:, 9 * self.group:9 * self.group+18, :, :]
            att1 = dynamic_filter_offset_att1[:, -9:, :, :]  # N, 9, H, W
            att1 = att1.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            att1 = att1.sigmoid()

            dynamic_filter2 = dynamic_filter_offset_att2[:, :9 * self.group, :, :]
            offset2 = dynamic_filter_offset_att2[:, 9 * self.group:9 * self.group + 18, :, :]
            att2 = dynamic_filter_offset_att2[:, -9:, :, :]  # N, 9, H, W
            att2 = att2.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            att2 = att2.sigmoid()

            dynamic_filter3 = dynamic_filter_offset_att3[:, :9 * self.group, :, :]
            offset3 = dynamic_filter_offset_att3[:, 9 * self.group:9 * self.group + 18, :, :]
            att3 = dynamic_filter_offset_att3[:, -9:, :, :]  # N, 9, H, W
            att3 = att3.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            att3 = att3.sigmoid()

            dynamic_filter4 = dynamic_filter_offset_att4[:, :9 * self.group, :, :]
            offset4 = dynamic_filter_offset_att4[:, 9 * self.group:9 * self.group + 18, :, :]
            att4 = dynamic_filter_offset_att4[:, -9:, :, :]  # N, 9, H, W
            att4 = att4.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            att4 = att4.sigmoid()

        elif self.deform == 'deform':
            dynamic_filter_offset1 = self.cata(blur_depth)
            dynamic_filter_offset2 = self.catb(blur_depth)
            dynamic_filter_offset3 = self.catc(blur_depth)
            dynamic_filter_offset4 = self.catd(blur_depth)

            dynamic_filter1 = dynamic_filter_offset1[:, :9, :, :]
            offset1 = dynamic_filter_offset1[:, -18:, :, :]

            dynamic_filter2 = dynamic_filter_offset2[:, :9, :, :]
            offset2 = dynamic_filter_offset2[:, -18:, :, :]

            dynamic_filter3 = dynamic_filter_offset3[:, :9, :, :]
            offset3 = dynamic_filter_offset3[:, -18:, :, :]

            dynamic_filter4 = dynamic_filter_offset4[:, :9, :, :]
            offset4 = dynamic_filter_offset4[:, -18:, :, :]
        else:
            dynamic_filter1 = self.cata(blur_depth)
            dynamic_filter2 = self.catb(blur_depth)
            dynamic_filter3 = self.catc(blur_depth)
            dynamic_filter4 = self.catd(blur_depth)

        dynamic_filter1 = self.softmax(dynamic_filter1.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
        dynamic_filter2 = self.softmax(dynamic_filter2.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
        dynamic_filter3 = self.softmax(dynamic_filter3.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
        dynamic_filter4 = self.softmax(dynamic_filter4.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)

        if self.training and self.shuffle:
            dynamic_filter1 = dynamic_filter1.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx1 = torch.randperm(self.group)
            dynamic_filter1 = dynamic_filter1[idx1].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter2 = dynamic_filter2.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx2 = torch.randperm(self.group)
            dynamic_filter2 = dynamic_filter2[idx2].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter3 = dynamic_filter3.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx3 = torch.randperm(self.group)
            dynamic_filter3 = dynamic_filter3[idx3].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter4 = dynamic_filter4.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx4 = torch.randperm(self.group)
            dynamic_filter4 = dynamic_filter4[idx4].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

        if self.deform == 'none':
            xd_unfold1 = self.unfold1(blur_depth)
            xd_unfold2 = self.unfold2(blur_depth)
            xd_unfold3 = self.unfold3(blur_depth)
            xd_unfold4 = self.unfold4(blur_depth)
        else:
            xd_unfold1 = self.unfold1(blur_depth, offset1)
            xd_unfold2 = self.unfold2(blur_depth, offset2)
            xd_unfold3 = self.unfold3(blur_depth, offset3)
            xd_unfold4 = self.unfold4(blur_depth, offset4)

        if self.deform == 'deformatt':
            xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W)
            xd_unfold2 = xd_unfold2.view(N, C, self.K, H * W)
            xd_unfold3 = xd_unfold3.view(N, C, self.K, H * W)
            xd_unfold4 = xd_unfold4.view(N, C, self.K, H * W)

            xd_unfold1 *= att1
            xd_unfold2 *= att2
            xd_unfold3 *= att3
            xd_unfold4 *= att4

            xd_unfold1 = xd_unfold1.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold2 = xd_unfold2.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold3 = xd_unfold3.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold4 = xd_unfold4.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
        else:
            xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold2 = xd_unfold2.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold3 = xd_unfold3.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold4 = xd_unfold4.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)

        out1 = torch.bmm(xd_unfold1, dynamic_filter1.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out1 = out1.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out2 = torch.bmm(xd_unfold2, dynamic_filter2.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out2 = out2.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out3 = torch.bmm(xd_unfold3, dynamic_filter3.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out3 = out3.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out4 = torch.bmm(xd_unfold4, dynamic_filter4.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out4 = out4.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)

        out = self.scale2(torch.cat((x, out1, out2, out3, out4), 1)) + x

        return out


class ReDynamicWeightsCat11(nn.Module):
    r'''' a rigrous implementation but slow

    '''''
    def __init__(self, channels, group=1, kernel=3, dilation=(1, 4, 8, 12), shuffle=False, deform=None):
        super(ReDynamicWeightsCat11, self).__init__()
        in_channel = channels // 4
        self.scale1 = nn.Sequential(nn.Conv2d(channels, in_channel, 1, padding=0, bias=False),
                                    group_norm(in_channel),
                                    nn.ReLU(inplace=True))

        if deform == 'deformatt':
            self.off_conva = nn.Conv2d(in_channel, 18, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.off_convb = nn.Conv2d(in_channel, 18, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.off_convc = nn.Conv2d(in_channel, 18, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.off_convd = nn.Conv2d(in_channel, 18, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.kernel_conva = DeformConv(in_channel, group * kernel * kernel + 9, kernel_size=3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.kernel_convb = DeformConv(in_channel, group * kernel * kernel + 9, kernel_size=3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.kernel_convc = DeformConv(in_channel, group * kernel * kernel + 9, kernel_size=3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.kernel_convd = DeformConv(in_channel, group * kernel * kernel + 9, kernel_size=3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = DeformUnfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])

        elif deform == 'deform':
            self.off_conva = nn.Conv2d(in_channel, 18, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.off_convb = nn.Conv2d(in_channel, 18, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.off_convc = nn.Conv2d(in_channel, 18, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.off_convd = nn.Conv2d(in_channel, 18, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.kernel_conva = DeformConv(in_channel, group * kernel * kernel, kernel_size=3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.kernel_convb = DeformConv(in_channel, group * kernel * kernel, kernel_size=3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.kernel_convc = DeformConv(in_channel, group * kernel * kernel, kernel_size=3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.kernel_convd = DeformConv(in_channel, group * kernel * kernel, kernel_size=3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = DeformUnfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])
        else:
            self.cata = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.catb = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.catc = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.catd = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = nn.Unfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = nn.Unfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = nn.Unfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = nn.Unfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])

        self.softmax = nn.Softmax(dim=-1)

        self.shuffle = shuffle
        self.deform = deform
        self.group = group
        self.K = kernel * kernel

        self.gamma1 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))
        self.gamma2 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))
        self.gamma3 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))
        self.gamma4 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))

        self.scale3 = nn.Sequential(nn.Conv2d(in_channel, channels, 1, padding=0, bias=False),
                                    group_norm(channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        xd = self.scale1(x)
        blur_depth = xd

        N, C, H, W = xd.size()
        R = C // self.group

        if self.deform == 'deformatt':
            offset1 = self.off_conva(blur_depth)
            offset2 = self.off_convb(blur_depth)
            offset3 = self.off_convc(blur_depth)
            offset4 = self.off_convd(blur_depth)

            xd_unfold1 = self.unfold1(blur_depth, offset1)
            xd_unfold2 = self.unfold2(blur_depth, offset2)
            xd_unfold3 = self.unfold3(blur_depth, offset3)
            xd_unfold4 = self.unfold4(blur_depth, offset4)

            dynamic_filter_att1 = self.kernel_conva(blur_depth, offset1)
            dynamic_filter_att2 = self.kernel_convb(blur_depth, offset2)
            dynamic_filter_att3 = self.kernel_convc(blur_depth, offset3)
            dynamic_filter_att4 = self.kernel_convd(blur_depth, offset4)

            dynamic_filter1 = dynamic_filter_att1[:, :9*self.group, :, :]
            att1 = dynamic_filter_att1[:, -9:, :, :]  # N, 9, H, W
            att1 = att1.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0, 1, 3, 2).contiguous()  # N, 1, HW, K
            att1 = self.softmax(att1)

            dynamic_filter2 = dynamic_filter_att2[:, :9*self.group, :, :]
            att2 = dynamic_filter_att2[:, -9:, :, :]  # N, 9, H, W
            att2 = att2.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0, 1, 3, 2).contiguous()  # N, 1, HW, K
            att2 = self.softmax(att2)

            dynamic_filter3 = dynamic_filter_att3[:, :9*self.group, :, :]
            att3 = dynamic_filter_att3[:, -9:, :, :]  # N, 9, H, W
            att3 = att3.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0, 1, 3, 2).contiguous()  # N, 1, HW, K
            att3 = self.softmax(att3)

            dynamic_filter4 = dynamic_filter_att4[:, :9 * self.group, :, :]
            att4 = dynamic_filter4[:, -9:, :, :]  # N, 9, H, W
            att4 = att4.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0, 1, 3, 2).contiguous()  # N, 1, HW, K
            att4 = self.softmax(att4)

        elif self.deform == 'deform':
            offset1 = self.off_conva(blur_depth)
            offset2 = self.off_convb(blur_depth)
            offset3 = self.off_convc(blur_depth)
            offset4 = self.off_convd(blur_depth)

            xd_unfold1 = self.unfold1(blur_depth, offset1)
            xd_unfold2 = self.unfold2(blur_depth, offset2)
            xd_unfold3 = self.unfold3(blur_depth, offset3)
            xd_unfold4 = self.unfold4(blur_depth, offset4)

            dynamic_filter1 = self.kernel_conva(blur_depth, offset1)
            dynamic_filter2 = self.kernel_convb(blur_depth, offset2)
            dynamic_filter3 = self.kernel_convc(blur_depth, offset3)
            dynamic_filter4 = self.kernel_convd(blur_depth, offset4)
        else:
            dynamic_filter1 = self.cata(blur_depth)
            dynamic_filter2 = self.catb(blur_depth)
            dynamic_filter3 = self.catc(blur_depth)
            dynamic_filter4 = self.catd(blur_depth)

            xd_unfold1 = self.unfold1(blur_depth)
            xd_unfold2 = self.unfold2(blur_depth)
            xd_unfold3 = self.unfold3(blur_depth)
            xd_unfold4 = self.unfold4(blur_depth)

        if self.deform == 'deformatt':
            dynamic_filter1 = dynamic_filter1.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous()  # N, G, HW, K
            dynamic_filter2 = dynamic_filter2.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous()  # N, G, HW, K
            dynamic_filter3 = dynamic_filter3.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous()  # N, G, HW, K
            dynamic_filter4 = dynamic_filter4.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous()  # N, G, HW, K
        else:
            dynamic_filter1 = self.softmax(dynamic_filter1.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
            dynamic_filter2 = self.softmax(dynamic_filter2.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
            dynamic_filter3 = self.softmax(dynamic_filter3.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
            dynamic_filter4 = self.softmax(dynamic_filter4.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)

        if self.training and self.shuffle:
            dynamic_filter1 = dynamic_filter1.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx1 = torch.randperm(self.group)
            dynamic_filter1 = dynamic_filter1[idx1].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter2 = dynamic_filter2.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx2 = torch.randperm(self.group)
            dynamic_filter2 = dynamic_filter2[idx2].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter3 = dynamic_filter3.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx3 = torch.randperm(self.group)
            dynamic_filter3 = dynamic_filter3[idx3].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter4 = dynamic_filter4.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx4 = torch.randperm(self.group)
            dynamic_filter4 = dynamic_filter4[idx4].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

        if self.deform == 'deformatt':
            dynamic_filter1 = dynamic_filter1 * att1
            dynamic_filter2 = dynamic_filter2 * att2
            dynamic_filter3 = dynamic_filter3 * att3
            dynamic_filter4 = dynamic_filter4 * att4

            dynamic_filter1 = dynamic_filter1.view(-1, self.K)  # (NGHW, K)
            dynamic_filter2 = dynamic_filter2.view(-1, self.K)  # (NGHW, K)
            dynamic_filter3 = dynamic_filter3.view(-1, self.K)  # (NGHW, K)
            dynamic_filter4 = dynamic_filter4.view(-1, self.K)  # (NGHW, K)

        xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(N * self.group * H * W, R, self.K)  # (BGHW, R, K)
        xd_unfold2 = xd_unfold2.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(N * self.group * H * W, R, self.K)  # (BGHW, R, K)
        xd_unfold3 = xd_unfold3.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(N * self.group * H * W, R, self.K)  # (BGHW, R, K)
        xd_unfold4 = xd_unfold4.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(N * self.group * H * W, R, self.K)  # (BGHW, R, K)

        out1 = torch.bmm(xd_unfold1, dynamic_filter1.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out1 = out1.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out2 = torch.bmm(xd_unfold2, dynamic_filter2.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out2 = out2.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out3 = torch.bmm(xd_unfold3, dynamic_filter3.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out3 = out3.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out4 = torch.bmm(xd_unfold4, dynamic_filter4.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out4 = out4.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)

        # out = self.scale3(self.scale2(torch.cat((xd, out1, out2, out3, out4), 1))) + x
        out = xd + self.gamma1 * out1 + self.gamma2 * out2 + self.gamma3 * out3 + self.gamma4 * out4
        out = self.scale3(out) + x

        return out


class ReDynamicWeightsCat33(nn.Module):
    r'''' a rigrous implementation but slow

    '''''
    def __init__(self, channels, group=1, kernel=3, dilation=(1, 4, 8, 12), shuffle=False, deform=None):
        super(ReDynamicWeightsCat33, self).__init__()
        in_channel = channels

        if deform == 'deformatt':
            self.off_conva = nn.Conv2d(in_channel, 18, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.off_convb = nn.Conv2d(in_channel, 18, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.off_convc = nn.Conv2d(in_channel, 18, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.off_convd = nn.Conv2d(in_channel, 18, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.kernel_conva = DeformConv(in_channel, group * kernel * kernel + 9, kernel_size=3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.kernel_convb = DeformConv(in_channel, group * kernel * kernel + 9, kernel_size=3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.kernel_convc = DeformConv(in_channel, group * kernel * kernel + 9, kernel_size=3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.kernel_convd = DeformConv(in_channel, group * kernel * kernel + 9, kernel_size=3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = DeformUnfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])

        elif deform == 'deform':
            self.off_conva = nn.Conv2d(in_channel, 18, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.off_convb = nn.Conv2d(in_channel, 18, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.off_convc = nn.Conv2d(in_channel, 18, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.off_convd = nn.Conv2d(in_channel, 18, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.kernel_conva = DeformConv(in_channel, group * kernel * kernel, kernel_size=3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.kernel_convb = DeformConv(in_channel, group * kernel * kernel, kernel_size=3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.kernel_convc = DeformConv(in_channel, group * kernel * kernel, kernel_size=3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.kernel_convd = DeformConv(in_channel, group * kernel * kernel, kernel_size=3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = DeformUnfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])
        else:
            self.cata = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.catb = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[1], dilation=dilation[1], bias=False)
            self.catc = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[2], dilation=dilation[2], bias=False)
            self.catd = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[3], dilation=dilation[3], bias=False)

            self.unfold1 = nn.Unfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = nn.Unfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = nn.Unfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])
            self.unfold4 = nn.Unfold(kernel_size=(3, 3), padding=dilation[3], dilation=dilation[3])

        self.softmax = nn.Softmax(dim=-1)

        self.shuffle = shuffle
        self.deform = deform
        self.group = group
        self.K = kernel * kernel

        self.gamma1 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))
        self.gamma2 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))
        self.gamma3 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))
        self.gamma4 = nn.Parameter(torch.FloatTensor(1).fill_(1.0))

        # self.scale2 = nn.Sequential(nn.Conv2d(in_channel * 5, in_channel, 1, padding=0, bias=False),
        #                             group_norm(in_channel),
        #                             nn.ReLU(inplace=True))

    def forward(self, x):
        blur_depth = x

        N, C, H, W = x.size()
        R = C // self.group

        if self.deform == 'deformatt':
            offset1 = self.off_conva(blur_depth)
            offset2 = self.off_convb(blur_depth)
            offset3 = self.off_convc(blur_depth)
            offset4 = self.off_convd(blur_depth)

            xd_unfold1 = self.unfold1(blur_depth, offset1)
            xd_unfold2 = self.unfold2(blur_depth, offset2)
            xd_unfold3 = self.unfold3(blur_depth, offset3)
            xd_unfold4 = self.unfold4(blur_depth, offset4)

            dynamic_filter_att1 = self.kernel_conva(blur_depth, offset1)
            dynamic_filter_att2 = self.kernel_convb(blur_depth, offset2)
            dynamic_filter_att3 = self.kernel_convc(blur_depth, offset3)
            dynamic_filter_att4 = self.kernel_convd(blur_depth, offset4)

            dynamic_filter1 = dynamic_filter_att1[:, :9*self.group, :, :]
            att1 = dynamic_filter_att1[:, -9:, :, :]  # N, 9, H, W
            att1 = att1.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0, 1, 3, 2).contiguous()  # N, 1, HW, K
            att1 = self.softmax(att1)

            dynamic_filter2 = dynamic_filter_att2[:, :9*self.group, :, :]
            att2 = dynamic_filter_att2[:, -9:, :, :]  # N, 9, H, W
            att2 = att2.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0, 1, 3, 2).contiguous()  # N, 1, HW, K
            att2 = self.softmax(att2)

            dynamic_filter3 = dynamic_filter_att3[:, :9*self.group, :, :]
            att3 = dynamic_filter_att3[:, -9:, :, :]  # N, 9, H, W
            att3 = att3.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0, 1, 3, 2).contiguous()  # N, 1, HW, K
            att3 = self.softmax(att3)

            dynamic_filter4 = dynamic_filter_att4[:, :9 * self.group, :, :]
            att4 = dynamic_filter4[:, -9:, :, :]  # N, 9, H, W
            att4 = att4.view(N, -1, H * W).view(N, 1, -1, H * W).permute(0, 1, 3, 2).contiguous()  # N, 1, HW, K
            att4 = self.softmax(att4)

        elif self.deform == 'deform':
            offset1 = self.off_conva(blur_depth)
            offset2 = self.off_convb(blur_depth)
            offset3 = self.off_convc(blur_depth)
            offset4 = self.off_convd(blur_depth)

            xd_unfold1 = self.unfold1(blur_depth, offset1)
            xd_unfold2 = self.unfold2(blur_depth, offset2)
            xd_unfold3 = self.unfold3(blur_depth, offset3)
            xd_unfold4 = self.unfold4(blur_depth, offset4)

            dynamic_filter1 = self.kernel_conva(blur_depth, offset1)
            dynamic_filter2 = self.kernel_convb(blur_depth, offset2)
            dynamic_filter3 = self.kernel_convc(blur_depth, offset3)
            dynamic_filter4 = self.kernel_convd(blur_depth, offset4)
        else:
            dynamic_filter1 = self.cata(blur_depth)
            dynamic_filter2 = self.catb(blur_depth)
            dynamic_filter3 = self.catc(blur_depth)
            dynamic_filter4 = self.catd(blur_depth)

            xd_unfold1 = self.unfold1(blur_depth)
            xd_unfold2 = self.unfold2(blur_depth)
            xd_unfold3 = self.unfold3(blur_depth)
            xd_unfold4 = self.unfold4(blur_depth)

        if self.deform == 'deformatt':
            dynamic_filter1 = dynamic_filter1.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous()  # N, G, HW, K
            dynamic_filter2 = dynamic_filter2.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous()  # N, G, HW, K
            dynamic_filter3 = dynamic_filter3.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous()  # N, G, HW, K
            dynamic_filter4 = dynamic_filter4.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous()  # N, G, HW, K
        else:
            dynamic_filter1 = self.softmax(dynamic_filter1.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
            dynamic_filter2 = self.softmax(dynamic_filter2.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
            dynamic_filter3 = self.softmax(dynamic_filter3.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)
            dynamic_filter4 = self.softmax(
                dynamic_filter4.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)

        if self.training and self.shuffle:
            dynamic_filter1 = dynamic_filter1.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx1 = torch.randperm(self.group)
            dynamic_filter1 = dynamic_filter1[idx1].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter2 = dynamic_filter2.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx2 = torch.randperm(self.group)
            dynamic_filter2 = dynamic_filter2[idx2].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter3 = dynamic_filter3.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx3 = torch.randperm(self.group)
            dynamic_filter3 = dynamic_filter3[idx3].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

            dynamic_filter4 = dynamic_filter4.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx4 = torch.randperm(self.group)
            dynamic_filter4 = dynamic_filter4[idx4].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

        if self.deform == 'deformatt':
            dynamic_filter1 = dynamic_filter1 * att1
            dynamic_filter2 = dynamic_filter2 * att2
            dynamic_filter3 = dynamic_filter3 * att3
            dynamic_filter4 = dynamic_filter4 * att4

            dynamic_filter1 = dynamic_filter1.view(-1, self.K)  # (NGHW, K)
            dynamic_filter2 = dynamic_filter2.view(-1, self.K)  # (NGHW, K)
            dynamic_filter3 = dynamic_filter3.view(-1, self.K)  # (NGHW, K)
            dynamic_filter4 = dynamic_filter4.view(-1, self.K)  # (NGHW, K)

        xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
            N * self.group * H * W, R, self.K)  # (BGHW, R, K)
        xd_unfold2 = xd_unfold2.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
            N * self.group * H * W, R, self.K)  # (BGHW, R, K)
        xd_unfold3 = xd_unfold3.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
            N * self.group * H * W, R, self.K)  # (BGHW, R, K)
        xd_unfold4 = xd_unfold4.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2,
                                                                                                                                         4).contiguous().view(
            N * self.group * H * W, R, self.K)  # (BGHW, R, K)

        out1 = torch.bmm(xd_unfold1, dynamic_filter1.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out1 = out1.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out2 = torch.bmm(xd_unfold2, dynamic_filter2.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out2 = out2.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out3 = torch.bmm(xd_unfold3, dynamic_filter3.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out3 = out3.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)
        out4 = torch.bmm(xd_unfold4, dynamic_filter4.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out4 = out4.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)

        # out = self.scale2(torch.cat((x, out1, out2, out3, out4), 1)) + x
        out = x + self.gamma1 * out1 + self.gamma2 * out2 + self.gamma3 * out3 + self.gamma4 * out4

        return out


