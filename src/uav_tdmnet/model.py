from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DetailGate(nn.Module):
    """Preserve fine texture before feature fusion erases tiny-target cues."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(x)


class DetailPreservingPyramidNeck(nn.Module):
    """
    A lightweight high-resolution neck with an extra P2 branch.

    Input order is expected to be [C2, C3, C4, C5].
    Output order is [P2, P3, P4, P5].
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.detail_gates = nn.ModuleList([DetailGate(out_channels) for _ in in_channels])
        self.smooth = nn.ModuleList([ConvBNAct(out_channels, out_channels, 3) for _ in in_channels])

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        c2, c3, c4, c5 = [proj(feat) for proj, feat in zip(self.lateral, features)]
        p5 = self.detail_gates[3](c5)
        p4 = self.detail_gates[2](c4 + nn.functional.interpolate(p5, scale_factor=2, mode="nearest"))
        p3 = self.detail_gates[1](c3 + nn.functional.interpolate(p4, scale_factor=2, mode="nearest"))
        p2 = self.detail_gates[0](c2 + nn.functional.interpolate(p3, scale_factor=2, mode="nearest"))
        return [self.smooth[0](p2), self.smooth[1](p3), self.smooth[2](p4), self.smooth[3](p5)]


class MotionGate(nn.Module):
    """Fuse current and previous features using a learned motion confidence map."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            ConvBNAct(channels * 3, channels, 3),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.memory_refine = ConvBNAct(channels, channels, 3)

    def forward(self, current: Tensor, previous: Optional[Tensor]) -> Tensor:
        if previous is None:
            return current
        aligned_previous = self.memory_refine(previous)
        gate = self.gate(torch.cat([current, aligned_previous, torch.abs(current - aligned_previous)], dim=1))
        return current + gate * aligned_previous


class MotionGatedTemporalFusion(nn.Module):
    def __init__(self, channels: Sequence[int]):
        super().__init__()
        self.blocks = nn.ModuleList([MotionGate(ch) for ch in channels])

    def forward(
        self,
        current_features: Sequence[Tensor],
        previous_features: Optional[Sequence[Tensor]] = None,
    ) -> List[Tensor]:
        if previous_features is None:
            return list(current_features)
        return [
            block(cur, prev) for block, cur, prev in zip(self.blocks, current_features, previous_features)
        ]


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_head = nn.Conv2d(in_channels, num_classes, 1)
        self.box_head = nn.Conv2d(in_channels, 4, 1)
        self.obj_head = nn.Conv2d(in_channels, 1, 1)

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        return {
            "cls_logits": self.cls_head(feat),
            "box_reg": self.box_head(feat),
            "obj_logits": self.obj_head(feat),
        }


class EmbeddingOcclusionHead(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3),
            nn.Conv2d(in_channels, embedding_dim, 1),
        )
        self.occlusion = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        return {
            "embeddings": self.embedding(feat),
            "occlusion_logits": self.occlusion(feat),
        }


class UAVTDMHead(nn.Module):
    def __init__(self, channels: Sequence[int], num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.det_heads = nn.ModuleList([DetectionHead(ch, num_classes) for ch in channels])
        self.assoc_heads = nn.ModuleList([EmbeddingOcclusionHead(ch, embedding_dim) for ch in channels])

    def forward(self, pyramid_features: Sequence[Tensor]) -> List[Dict[str, Tensor]]:
        outputs = []
        for feat, det_head, assoc_head in zip(pyramid_features, self.det_heads, self.assoc_heads):
            merged = det_head(feat)
            merged.update(assoc_head(feat))
            outputs.append(merged)
        return outputs


@dataclass
class TrackerMatchWeights:
    iou: float = 0.5
    embedding: float = 0.35
    occlusion: float = 0.15


class OcclusionAwareAssociation(nn.Module):
    """Score helper for a ByteTrack-style tracker."""

    def __init__(self, weights: TrackerMatchWeights = TrackerMatchWeights()):
        super().__init__()
        self.weights = weights

    def forward(self, iou_score: Tensor, embedding_score: Tensor, occlusion_score: Tensor) -> Tensor:
        return (
            self.weights.iou * iou_score
            + self.weights.embedding * embedding_score
            + self.weights.occlusion * occlusion_score
        )


class DummyBackbone(nn.Module):
    """
    Placeholder backbone for the scaffold.

    Replace this with CSPDarknet or the upstream ByteTrack backbone when integrating.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.stem = ConvBNAct(in_channels, base_channels, 3, stride=2)
        self.stage2 = ConvBNAct(base_channels, 128, 3, stride=2)
        self.stage3 = ConvBNAct(128, 256, 3, stride=2)
        self.stage4 = ConvBNAct(256, 512, 3, stride=2)
        self.stage5 = ConvBNAct(512, 1024, 3, stride=2)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return [c2, c3, c4, c5]


class UAVTDMNet(nn.Module):
    """
    Research scaffold for the proposed detector-tracker.

    This file is intentionally lightweight: it defines the interfaces and the
    proposed module boundaries so the model can be migrated into a ByteTrack-like
    codebase with minimal ambiguity.
    """

    def __init__(self, num_classes: int = 10, embedding_dim: int = 128):
        super().__init__()
        self.backbone = DummyBackbone()
        self.neck = DetailPreservingPyramidNeck([128, 256, 512, 1024], out_channels=256)
        self.temporal = MotionGatedTemporalFusion([256, 256, 256, 256])
        self.head = UAVTDMHead([256, 256, 256, 256], num_classes=num_classes, embedding_dim=embedding_dim)
        self.association = OcclusionAwareAssociation()

    def forward(
        self,
        current_frame: Tensor,
        previous_frame: Optional[Tensor] = None,
    ) -> Dict[str, List[Dict[str, Tensor]]]:
        current_feats = self.neck(self.backbone(current_frame))
        previous_feats = None
        if previous_frame is not None:
            previous_feats = self.neck(self.backbone(previous_frame))
        fused_feats = self.temporal(current_feats, previous_feats)
        outputs = self.head(fused_feats)
        return {"multi_scale_outputs": outputs}


if __name__ == "__main__":
    model = UAVTDMNet(num_classes=10, embedding_dim=128)
    current = torch.randn(1, 3, 640, 640)
    previous = torch.randn(1, 3, 640, 640)
    result = model(current, previous)
    print("Scales:", len(result["multi_scale_outputs"]))
    print("Keys:", sorted(result["multi_scale_outputs"][0].keys()))
