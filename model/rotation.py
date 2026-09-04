"""Differentiable rotation utilities (PyTorch).

These mirror the numpy functions in ``blender_gen/geometry.py`` that produced the
labels. They are deliberately a separate implementation -- autograd needs torch ops --
so ``tests/test_rotation_parity.py`` cross-checks the two against each other. If the
training-time math ever drifts from the label-time math, that test fails rather than
the model quietly learning a slightly wrong target.

Rotations are predicted in the continuous 6D representation of Zhou et al. (CVPR 2019).
Quaternions and Euler angles are discontinuous as maps onto SO(3): near the
discontinuity a tiny change in rotation demands a huge change in the network's output,
which caps achievable regression accuracy no matter how long you train. The 6D form has
no such seam -- the network emits two arbitrary 3-vectors and Gram-Schmidt projects them
onto the nearest valid rotation.
"""

from __future__ import annotations

import torch


def rot6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """``(B, 6)`` -> ``(B, 3, 3)`` via Gram-Schmidt. Output is always a valid rotation."""
    a1, a2 = r6[:, :3], r6[:, 3:]
    b1 = torch.nn.functional.normalize(a1, dim=1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)      # columns, matching geometry.py


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """``(B, 3, 3)`` -> ``(B, 6)``: the first two columns, flattened."""
    return R[:, :, :2].transpose(1, 2).reshape(R.shape[0], 6)


def geodesic_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Angle of the relative rotation, in degrees -- the headline pose metric.

    This is the honest way to score a rotation: it is the single angle you would have
    to turn the prediction through to land on the truth, so it is invariant to how the
    rotation happens to be parameterised.

    This is a **metric, not a loss** -- ``acos`` has unbounded gradient at 0 and pi, so
    optimise :func:`frobenius_loss` instead and report this. Being gradient-free lets it
    clamp to exactly [-1, 1] and evaluate in float64; backing off the clamp by an epsilon
    instead would put an artificial floor (~0.026 deg at 1e-7) under every number reported.

    Evaluated on CPU in float64 and returned on CPU: MPS has no float64 at all, and in
    float32 ``acos`` near 1 resolves only to ~0.026 deg, which would floor exactly the
    small errors a converged model produces. Metric-only, so the device round-trip is free.
    """
    a = R_pred.detach().cpu().double()
    b = R_gt.detach().cpu().double()
    cos = ((a.transpose(1, 2) @ b).diagonal(dim1=1, dim2=2).sum(1) - 1.0) / 2.0
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


def frobenius_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Mean squared Frobenius distance between rotations (the chordal distance).

    Preferred over differentiating the geodesic angle directly: ``acos`` has infinite
    gradient at 0 and pi, so a geodesic loss is numerically unstable exactly when the
    prediction is nearly right. Frobenius is smooth everywhere and shares its minimum.
    """
    return ((R_pred - R_gt) ** 2).sum(dim=(1, 2)).mean()
