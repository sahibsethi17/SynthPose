"""ResNet backbone with separate rotation and translation heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

import rotation as rot

BACKBONES = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (torchvision.models.resnet34, torchvision.models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V2),
}


class PoseNet(nn.Module):
    """Single RGB image -> (rotation matrix, t/s).

    Two heads rather than one six-number output: rotation and translation are
    geometrically different quantities. Rotation is emitted as the continuous 6D
    representation and projected onto SO(3) with Gram-Schmidt, so the network can never
    produce an invalid rotation and never has to learn the orthonormality constraint.
    Translation is regressed in standardised units (see ``--- standardisation`` below).

    ImageNet pretraining matters more than usual here. The renders are untextured
    objects on flat backgrounds, so there is little low-level texture statistic to learn
    from scratch; the pretrained early layers supply edge and shading filters that
    shading-based 3D reasoning depends on.
    """

    def __init__(self, backbone="resnet18", pretrained=True, dropout=0.0):
        super().__init__()
        ctor, weights = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net

        def head(out_dim):
            layers = [nn.Linear(feat_dim, 256), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(256, out_dim))
            return nn.Sequential(*layers)

        self.rot_head = head(6)
        self.trans_head = head(3)

    def forward(self, x):
        """Returns ``(R[B,3,3], t_norm[B,3])``; ``t_norm`` is in standardised units."""
        f = self.backbone(x)
        return rot.rot6d_to_matrix(self.rot_head(f)), self.trans_head(f)
