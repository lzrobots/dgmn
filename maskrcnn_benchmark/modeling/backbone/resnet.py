# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple, OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import DFConv2d
from maskrcnn_benchmark.layers import DynamicWeightsCat33, DynamicWeightsCat11, ReDynamicWeightsCat33, DeformDGMN, ReDynamicWeightsCat11 #, GloReLocalModule
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry
from maskrcnn_benchmark.layers import DeformUnfold
from maskrcnn_benchmark.layers import DeformConv


# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)


class PAM_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.W = nn.Conv2d(in_channels=in_dim//4, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)
        out = self.W(out)

        out = out + x

        return out


class OneDynamicWeightsCat11(nn.Module):
    r'''' not a rigrous implementation but faster speed

    '''''

    def __init__(self, channels, group=1, kernel=3, dilation=(1, 4, 8, 12), shuffle=False, deform=None):
        super(OneDynamicWeightsCat11, self).__init__()
        in_channel = channels // 4
        self.scale1 = nn.Sequential(nn.Conv2d(channels, in_channel, 1, padding=0, bias=False),
                                    group_norm(in_channel),
                                    nn.ReLU(inplace=True))

        if deform == 'deform':
            self.cata = nn.Conv2d(in_channel, group * kernel * kernel + 18, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])

        else:
            self.cata = nn.Conv2d(in_channel, group * kernel * kernel, 3, padding=dilation[0], dilation=dilation[0], bias=False)
            self.unfold1 = nn.Unfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])

        self.softmax = nn.Softmax(dim=-1)

        self.shuffle = shuffle
        self.deform = deform
        self.group = group
        self.K = kernel * kernel

        self.scale2 = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, 1, padding=0, bias=True),
                                    group_norm(in_channel),
                                    nn.ReLU(inplace=True))

        self.scale3 = nn.Sequential(nn.Conv2d(in_channel, channels, 1, padding=0, bias=True),
                                    group_norm(channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        xd = self.scale1(x)
        blur_depth = xd

        N, C, H, W = xd.size()
        R = C // self.group

        if self.deform == 'deform':
            dynamic_filter_offset1 = self.cata(blur_depth)
            dynamic_filter1 = dynamic_filter_offset1[:, :9, :, :]
            offset1 = dynamic_filter_offset1[:, -18:, :, :]

        else:
            dynamic_filter1 = self.cata(blur_depth)

        dynamic_filter1 = self.softmax(dynamic_filter1.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1, self.K))  # (NGHW, K)

        if self.training and self.shuffle:
            dynamic_filter1 = dynamic_filter1.view(N, self.group, H * W, self.K).permute(1, 0, 2, 3).contiguous()
            idx1 = torch.randperm(self.group)
            dynamic_filter1 = dynamic_filter1[idx1].permute(1, 0, 2, 3).contiguous().view(-1, self.K)

        if self.deform == 'none':
            xd_unfold1 = self.unfold1(blur_depth)
        else:
            xd_unfold1 = self.unfold1(blur_depth, offset1)

        xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(
            N * self.group * H * W, R, self.K)  # (BGHW, R, K)

        out1 = torch.bmm(xd_unfold1, dynamic_filter1.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out1 = out1.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)

        out = self.scale3(self.scale2(torch.cat((xd, out1), 1))) + x

        return out


class ConvMP(nn.Module):
    def __init__(self, channels):
        super(ConvMP, self).__init__()
        in_channel = channels // 4
        self.scale1 = nn.Sequential(nn.Conv2d(channels, in_channel, 1, padding=0, bias=False),
                                    group_norm(in_channel),
                                    nn.ReLU(inplace=True))

        self.cata = nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel)

        self.scale2 = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, 1, padding=0, bias=False),
                                    group_norm(in_channel),
                                    nn.ReLU(inplace=True))

        self.scale3 = nn.Sequential(nn.Conv2d(in_channel, channels, 1, padding=0, bias=False),
                                    group_norm(channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        xd = self.scale1(x)
        blur_depth = xd

        out1 = self.cata(blur_depth)

        out = self.scale3(self.scale2(torch.cat((xd, out1), 1))) + x

        return out


class deformMP(nn.Module):
    def __init__(self, channels):
        super(deformMP, self).__init__()
        in_channel = channels // 4
        self.scale1 = nn.Sequential(nn.Conv2d(channels, in_channel, 1, padding=0, bias=False),
                                    group_norm(in_channel),
                                    nn.ReLU(inplace=True))

        self.off_conva = nn.Conv2d(in_channel, 18, 3, padding=1, bias=False)
        self.kernel_conva = DeformConv(in_channel, in_channel, kernel_size=3, padding=1, bias=False)

        self.scale3 = nn.Sequential(nn.Conv2d(in_channel, channels, 1, padding=0, bias=False),
                                    group_norm(channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        xd = self.scale1(x)

        offset1 = self.off_conva(xd)

        out1 = self.kernel_conva(xd, offset1)

        out = out1 + xd

        out = self.scale3(out) + x

        return out


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # Construct the stem module
        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index -1]
            stage_with_dw = cfg.MODEL.RESNETS.STAGE_WITH_DW[stage_spec.index -1]
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                stage_spec.index,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1,
                dcn_config={
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
                    "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,
                },
                dw_config={
                    "stage_with_dw": stage_with_dw,
                    "domain": cfg.MODEL.RESNETS.DYNAMIC_WEIGHT.DOMAIN,
                    "group": cfg.MODEL.RESNETS.DYNAMIC_WEIGHT.GROUP,
                    "kernel": cfg.MODEL.RESNETS.DYNAMIC_WEIGHT.KERNEL,
                    "dilation": cfg.MODEL.RESNETS.DYNAMIC_WEIGHT.DILATION,
                    "shuffle": cfg.MODEL.RESNETS.DYNAMIC_WEIGHT.SHUFFLE,
                    "deform": cfg.MODEL.RESNETS.DYNAMIC_WEIGHT.DEFORM,
                    "insert_pos": cfg.MODEL.RESNETS.DYNAMIC_WEIGHT.INSERT_POS,
                }
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
        dilation=1,
        dcn_config={}
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    index,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1,
    dcn_config={},
    dw_config={}
):
    blocks = []
    stride = first_stride

    if dw_config is not None:
        dw_dilation = dw_config.get('dilation', ((1, 4, 8, 12), (1, 4, 8, 12), (1, 4, 8, 12), (1, 2, 4, 8)))
        dw_config["dilation"] = dw_dilation[index-1]

    dw_block = None
    if dw_config is not None:
        dw_domain = dw_config.get('domain', 'block')
        assert dw_domain in ['stage', 'block']
        if dw_domain == 'stage':
            dw_group = dw_config.get('group', 1)
            dw_kernel = dw_config.get('kernel', 3)
            dw_dilation = dw_config.get('dilation', (1, 4, 8, 12))
            dw_shuffle = dw_config.get('shuffle', False)
            dw_deform = dw_config.get('deform', "none")
            dw_block = DynamicWeightsCat11(channels=out_channels,
                                           group=dw_group,
                                           kernel=dw_kernel,
                                           dilation=dw_dilation,
                                           shuffle=dw_shuffle,
                                           deform=dw_deform)
            dw_config = None

    for idx in range(block_count):
        # if idx == 5 and block_count == 6:
        #     blocks.append(('NL{}'.format(idx), PAM_Module(in_channels)))
        # if idx == 5 and block_count == 6:
        #     blocks.append(('DW{}'.format(idx), OneDynamicWeightsCat11(channels=in_channels, group=1, dilation=(1, 4, 8, 12), deform='none')))
        # if idx == 5 and block_count == 6:
        #     blocks.append(('DW{}'.format(idx), DynamicWeightsCat11(channels=in_channels, group=4, dilation=(1, 4, 8, 12), deform='deformatt')))
        # if idx == 5 and block_count == 6:
        #     blocks.append(('MP{}'.format(idx), ConvMP(channels=in_channels)))
        # if idx == 5 and block_count == 6:
        #     blocks.append(('deformMP{}'.format(idx), deformMP(channels=in_channels)))
        # if idx == 5 and block_count == 6:
        #     blocks.append(('DG{}'.format(idx), GloReLocalModule(planes=in_channels)))
        blocks.append(
            (str(idx),
                transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config,
                dw_config=dw_config
                )
            )
        )
        # if index == 4 and idx == 2:
        #     blocks.append(('DW{}'.format(idx), DynamicWeightsCat11(channels=in_channels, group=1, dilation=(1, 4, 8, 12), deform='none')))
        stride = 1
        in_channels = out_channels

    if dw_block is not None:
        blocks.append(('DW', dw_block))

    return nn.Sequential(OrderedDict(blocks))


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
        dcn_config,
        dw_config
    ):
        super(Bottleneck, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above
        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv2d(
                bottleneck_channels, 
                bottleneck_channels, 
                with_modulated_dcn=with_modulated_dcn, 
                kernel_size=3, 
                stride=stride_3x3, 
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        else:
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=dilation,
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)
        self.with_dw = dw_config.get("stage_with_dw", False)
        if self.with_dw:
            self.insert_pos = dw_config.get('insert_pos', 'after1x1')
            assert self.insert_pos in ['after1x1', 'after3x3', 'afterAdd']
            if self.insert_pos == 'afterAdd':
                dw_block = DynamicWeightsCat11
                dw_channels = out_channels
            elif self.insert_pos == 'after3x3':
                dw_block = ReDynamicWeightsCat33 #ReDynamicWeightsCat33, DeformDGMN
                dw_channels = bottleneck_channels
            dw_group = dw_config.get('group', 1)
            dw_kernel = dw_config.get('kernel', 3)
            dw_dilation = dw_config.get('dilation', (1, 4, 8, 12))
            dw_shuffle = dw_config.get('shuffle', False)
            dw_deform = dw_config.get('deform', 'none')
            self.dw_block = dw_block(channels=dw_channels,
                                     group=dw_group,
                                     kernel=dw_kernel,
                                     dilation=dw_dilation,
                                     shuffle=dw_shuffle,
                                     deform=dw_deform)
        else:
            self.dw_block = None

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        if self.with_dw and self.insert_pos == 'after3x3':
            out = self.dw_block(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.with_dw and self.insert_pos == 'after1x1':
            out = self.dw_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        if self.with_dw and self.insert_pos == 'afterAdd':
            out = self.dw_block(out)

        return out


class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={},
        dw_config={}

    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config,
            dw_config=dw_config
        )


class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class BottleneckWithGN(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm,
            dcn_config=dcn_config
        )


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)


_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})
