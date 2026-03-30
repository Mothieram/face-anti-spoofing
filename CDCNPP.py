"""CDCN++ model for face anti-spoofing.

This module implements a compact CDCN++-style network using central-difference
convolutions. The network predicts both:
1) a depth map
2) a spoof probability score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


class CDCConv2d(nn.Module):
    """Central-Difference Convolution.

    Standard convolution is corrected by subtracting the response to a spatially
    averaged kernel (controlled by theta), encouraging gradient-like responses.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        theta: float = 0.7,
    ) -> None:
        super().__init__()
        self.theta = float(theta)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_normal = self.conv(x)
        if self.theta <= 0.0:
            return out_normal

        # Depthwise-style correction using kernel mean.
        kernel_mean = self.conv.weight.mean(dim=(2, 3), keepdim=True)
        out_diff = F.conv2d(
            x,
            kernel_mean,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=0,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return out_normal - self.theta * out_diff


class CDCBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, theta: float = 0.7, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            CDCConv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CDCConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CDCNpp(nn.Module):
    """CDCN++ network that predicts depth map and spoof score.

    Input:  (B, 3, 256, 256)
    Output: dict with keys:
      - depth_map:  (B, 1, 256, 256)
      - spoof_logit:(B, 1)
      - spoof_prob: (B, 1)
    """

    def __init__(self, base_channels: int = 32, theta: float = 0.7) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = nn.Sequential(
            CDCConv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = CDCBlock(c1, c1, theta=theta, stride=1)
        self.stage2 = CDCBlock(c1, c2, theta=theta, stride=2)
        self.stage3 = CDCBlock(c2, c3, theta=theta, stride=2)
        self.stage4 = CDCBlock(c3, c4, theta=theta, stride=2)

        self.depth_head = nn.Sequential(
            CDCConv2d(c1 + c2 + c3 + c4, c2, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, 1, kernel_size=1, bias=True),
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, c2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(c2, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = x.shape[-2:]

        x0 = self.stem(x)
        f1 = self.stage1(x0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        # Multi-scale fusion for depth prediction.
        f2u = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f3u = F.interpolate(f3, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f4u = F.interpolate(f4, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        depth_feat = torch.cat([f1, f2u, f3u, f4u], dim=1)

        depth_map = self.depth_head(depth_feat)
        depth_map = F.interpolate(depth_map, size=(h, w), mode="bilinear", align_corners=False)

        spoof_logit = self.cls_head(f4)
        spoof_prob = torch.sigmoid(spoof_logit)

        return {
            "depth_map": depth_map,
            "spoof_logit": spoof_logit,
            "spoof_prob": spoof_prob,
        }


@dataclass
class CDCNppCheckpoint:
    state_dict: Dict[str, torch.Tensor]
    base_channels: int = 32
    theta: float = 0.7


def save_cdcnpp(model: CDCNpp, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "base_channels": 32,
        "theta": 0.7,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_cdcnpp(path: str, device: Optional[torch.device] = None) -> CDCNpp:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)

    if "state_dict" in ckpt:
        base_channels = int(ckpt.get("base_channels", 32))
        theta = float(ckpt.get("theta", 0.7))
        state_dict = ckpt["state_dict"]
    else:
        base_channels = 32
        theta = 0.7
        state_dict = ckpt

    model = CDCNpp(base_channels=base_channels, theta=theta).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_cdcnpp(device: Optional[torch.device] = None) -> CDCNpp:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return CDCNpp().to(device)
