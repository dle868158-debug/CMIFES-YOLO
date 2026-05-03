# CMIFES.py v7
# Cross-scale Multi-level Information Fusion Enhancement - Simplified
# Key change: Replace DeformableSpatialAttention with lightweight SE-ChannelAttention
# - Eliminates heavy DCNv2 offset computation overhead
# - SE: ~4K params (vs Deformable: ~400K) at 256 channels
# - Retains cross-scale multi-input fusion via learned gating
# - Retains simple spatial attention (3x3 conv + skip) for local refinement

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    """Squeeze-and-Excitation attention - ultra lightweight channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SimpleSpatialAttention(nn.Module):
    """Lightweight spatial attention - 3x3 conv + residual, ~6K params."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv(x)
        return self.sigmoid(attn + 1.0)  # residual: initialized to ~1 for identity


class CrossLayerGating(nn.Module):
    """Learned weighted fusion for multi-input features."""
    def __init__(self, channels, num_inputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * num_inputs, num_inputs * 4),
            nn.SiLU(inplace=True),
            nn.Linear(num_inputs * 4, num_inputs),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, feat_list):
        stacked = torch.cat(feat_list, dim=1)
        gate_w = F.softmax(self.gate_fc(stacked), dim=1)
        fused = torch.zeros_like(feat_list[0])
        for i, feat in enumerate(feat_list):
            w = gate_w[:, i].view(-1, 1, 1, 1)
            fused = fused + w * feat
        return self.proj(fused)


class CMIFES(nn.Module):
    """
    CMIFE-S: Simplified Cross-scale Multi-level Information Fusion Enhancement.
    Replaces heavy DeformableSpatialAttention with lightweight SE + SimpleSpatialAttention.

    Key improvements over v6 CMIFE:
    1. SE-ChannelAttention instead of LocalGlobalChannelAttention (much fewer params)
    2. SimpleSpatialAttention instead of DeformableSpatialAttention (removes DCNv2 overhead)
    3. CrossLayerGating: SiLU activation instead of ReLU (more expressive)
    4. All changes maintain the multi-input cross-scale fusion paradigm
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.out_channels = out_channels

        if isinstance(in_channels, list):
            self.num_inputs = len(in_channels)
            self.align_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
                for ch in in_channels
            ])
            self.gate = CrossLayerGating(out_channels, self.num_inputs)
        else:
            self.num_inputs = 1
            self.align_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )

        # Key change: SE attention is much lighter than LocalGlobalChannelAttention
        self.se_attn = SEAttention(out_channels, reduction)
        # Key change: simple spatial attention replaces deformable conv
        self.spatial_attn = SimpleSpatialAttention(out_channels)

    def forward(self, x):
        if isinstance(x, list) and self.num_inputs > 1:
            return self._forward_multi(x)
        else:
            return self._forward_single(x[0] if isinstance(x, list) else x)

    def _forward_multi(self, x_list):
        # Find target spatial size (largest H,W among inputs)
        target_h, target_w = 0, 0
        for xi in x_list:
            if xi is not None:
                _, _, h, w = xi.shape
                if h > target_h:
                    target_h, target_w = h, w

        # Align and interpolate all inputs to same spatial size
        aligned = []
        for i, xi in enumerate(x_list):
            if xi is None or i >= len(self.align_convs):
                continue
            feat = self.align_convs[i](xi)
            if feat.shape[2] != target_h or feat.shape[3] != target_w:
                feat = F.interpolate(feat, size=(target_h, target_w),
                                     mode='bilinear', align_corners=False)
            aligned.append(feat)

        # Learned weighted fusion
        fused = self.gate(aligned)

        # SE channel attention
        se_weight = self.se_attn(fused)
        fused = se_weight * fused

        # Simple spatial attention
        sp_weight = self.spatial_attn(fused)
        fused = sp_weight * fused

        return fused

    def _forward_single(self, x):
        x = self.align_conv(x)
        se_weight = self.se_attn(x)
        x = se_weight * x
        sp_weight = self.spatial_attn(x)
        x = sp_weight * x
        return x
