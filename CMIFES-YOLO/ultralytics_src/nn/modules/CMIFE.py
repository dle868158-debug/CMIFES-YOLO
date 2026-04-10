# CMIFE.py v3
# Cross-scale Multi-level Information Fusion Enhancement Module
# v3 fixes: AMP/HalfTensor dtype mismatch in DeformableSpatialAttention

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMIFE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.out_channels = out_channels

        if isinstance(in_channels, list):
            self.num_inputs = len(in_channels)
            self.in_channels_list = in_channels
            self.align_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
                for ch in in_channels
            ])
            self.gate = CrossLayerFeatureGating(out_channels, self.num_inputs)
        else:
            self.num_inputs = 1
            self.align_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
        self.channel_attn = LocalGlobalChannelAttention(out_channels, reduction)
        self.spatial_attn = DeformableSpatialAttention(out_channels)

    def forward(self, x):
        if isinstance(x, list) and self.num_inputs > 1:
            return self._forward_multi(x)
        else:
            return self._forward_single(x[0] if isinstance(x, list) else x)

    def _forward_multi(self, x_list):
        target_h, target_w = 0, 0
        for xi in x_list:
            if xi is not None:
                _, _, h, w = xi.shape
                if h > target_h:
                    target_h, target_w = h, w
        aligned = []
        for i, xi in enumerate(x_list):
            if xi is None or i >= len(self.align_convs):
                continue
            feat = self.align_convs[i](xi)
            if feat.shape[2] != target_h or feat.shape[3] != target_w:
                feat = F.interpolate(feat, size=(target_h, target_w),
                                     mode='bilinear', align_corners=False)
            aligned.append(feat)
        fused = self.gate(aligned)
        ca_weight = self.channel_attn(fused)
        fused = ca_weight * fused
        sa_weight = self.spatial_attn(fused)
        fused = sa_weight * fused
        return fused

    def _forward_single(self, x):
        x = self.align_conv(x)
        ca_weight = self.channel_attn(x)
        x = ca_weight * x
        sa_weight = self.spatial_attn(x)
        x = sa_weight * x
        return x


class LocalGlobalChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, grid_size=4):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        mid_channels = max(channels // reduction, 8)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d(grid_size)
        self.local_fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
        )
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
        )
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        g_avg = self.shared_mlp(self.global_avg_pool(x))
        g_max = self.shared_mlp(self.global_max_pool(x))
        global_attn = g_avg + g_max
        local_feat = self.local_pool(x)
        local_attn = self.local_fc(local_feat)
        local_attn = local_attn.mean(dim=(2, 3), keepdim=True)
        alpha = self.sigmoid(self.fusion_weight)
        attn = alpha * global_attn + (1 - alpha) * local_attn
        return self.sigmoid(attn)


class DeformableSpatialAttention(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.offset_conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2 * kernel_size * kernel_size, 3, padding=1, bias=True),
        )
        self.attn_conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_out, max_out], dim=1)
        offset = self.offset_conv(spatial_feat)
        attn_map = self._stable_deformable_attention(spatial_feat, offset)
        return self.sigmoid(attn_map)

    def _stable_deformable_attention(self, feat, offset):
        # v3 FIX: ensure grid dtype matches feat to avoid HalfTensor vs FloatTensor error
        b, c, h, w = feat.shape
        k = self.kernel_size

        # Key fix: specify dtype=feat.dtype so grid matches input tensor type
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1, 1, w, device=feat.device, dtype=feat.dtype),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(b, -1, -1, -1).contiguous()

        # Aggregate k*k offsets into per-pixel offset
        offset = offset.view(b, 2, k * k, h, w)
        offset_mean = offset.mean(dim=2)

        # Use tanh to bound offsets, preventing sampling out of bounds
        offset_norm = torch.tanh(offset_mean) * 0.5
        offset_norm = offset_norm.permute(0, 2, 3, 1)

        deformed_grid = base_grid + offset_norm
        deformed_grid = deformed_grid.clamp(-1, 1)

        # align_corners=False is more stable
        sampled = F.grid_sample(feat, deformed_grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        attn = self.attn_conv(sampled)
        return attn


class CrossLayerFeatureGating(nn.Module):
    def __init__(self, channels, num_inputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.channels = channels
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * num_inputs, num_inputs * 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_inputs * 4, num_inputs),
        )
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, feat_list):
        assert len(feat_list) == self.num_inputs
        stacked = torch.cat(feat_list, dim=1)
        gate_weights = self.gate_fc(stacked)
        gate_weights = F.softmax(gate_weights, dim=1)
        fused = torch.zeros_like(feat_list[0])
        for i, feat in enumerate(feat_list):
            w = gate_weights[:, i].view(-1, 1, 1, 1)
            fused = fused + w * feat
        fused = self.fusion_proj(fused)
        return fused
