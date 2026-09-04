"""The training-time rotation math must agree with the label-time rotation math.

``blender_gen/geometry.py`` (numpy) produced every label; ``model/rotation.py`` (torch)
consumes them. They are separate implementations because autograd needs torch ops, so
nothing but a test keeps them in sync. A silent divergence here would have the model
optimising toward a subtly different target than the one on disk.

    .venv/bin/python -m pytest tests/ -q      (or just run this file)
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender_gen"))
sys.path.insert(0, str(ROOT / "model"))

import geometry as geo      # noqa: E402  (numpy, label-time)
import rotation as rot      # noqa: E402  (torch, training-time)


def _random_rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([geo.random_rotation(rng) for _ in range(n)])


def test_rot6d_roundtrip_matches_numpy():
    R = _random_rotations(256)
    r6_np = np.stack([geo.matrix_to_rot6d(r) for r in R])
    r6_t = rot.matrix_to_rot6d(torch.from_numpy(R)).numpy()
    assert np.allclose(r6_np, r6_t, atol=1e-6), "6D encoding differs"

    back_np = np.stack([geo.rot6d_to_matrix(v) for v in r6_np])
    back_t = rot.rot6d_to_matrix(torch.from_numpy(r6_t)).numpy()
    assert np.allclose(back_np, back_t, atol=1e-6), "6D decoding differs"
    assert np.allclose(back_t, R, atol=1e-6), "6D round-trip is not identity"


def test_gram_schmidt_yields_valid_rotations():
    """Arbitrary network output must project to a genuine rotation, not just any matrix."""
    torch.manual_seed(0)
    R = rot.rot6d_to_matrix(torch.randn(512, 6) * 5.0)
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-5), "not orthonormal"
    assert torch.allclose(torch.det(R), torch.ones(512), atol=1e-5), "det != +1 (reflection)"


def test_geodesic_error_matches_numpy():
    A, B = _random_rotations(128, 1), _random_rotations(128, 2)
    ref = np.array([geo.geodesic_error_deg(a, b) for a, b in zip(A, B)])
    got = rot.geodesic_error_deg(torch.from_numpy(A), torch.from_numpy(B)).numpy()
    assert np.allclose(ref, got, atol=1e-3), "geodesic error differs"


def test_geodesic_error_is_zero_for_identical_rotations():
    R = torch.from_numpy(_random_rotations(64, 3))
    assert rot.geodesic_error_deg(R, R).max() < 1e-2


def test_frobenius_loss_minimised_at_truth():
    R = torch.from_numpy(_random_rotations(64, 4))
    perturbed = rot.rot6d_to_matrix(rot.matrix_to_rot6d(R) + torch.randn(64, 6) * 0.1)
    assert rot.frobenius_loss(R, R) < rot.frobenius_loss(perturbed, R)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS  {name}")
    print("\nall rotation parity tests pass")
