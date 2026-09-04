"""3D geometry helpers shared by the Blender generator and every downstream consumer.

This module is deliberately free of ``bpy`` so that the verification tool and the
PyTorch training code can import the *same* pose math the renderer used. Any
convention bug therefore shows up identically on both sides, where the
reprojection check in ``tools/verify_dataset.py`` will catch it.

Conventions
-----------
Blender camera : right-handed, looks down its local ``-Z``, local ``+Y`` is up.
OpenCV camera  : right-handed, looks down its local ``+Z``, local ``+Y`` is down.

The two differ by a 180 deg rotation about X, i.e. ``diag(1, -1, -1)``. All labels
written to disk are in the **OpenCV** convention, because that is what virtually
every vision codebase (and every PnP implementation) expects.

An object pose is stored decomposed as ``(R, t, s)`` with ``R`` orthonormal and
``s`` a uniform scale, so that a point in object-local coordinates maps to the
camera frame as::

    X_cam = s * (R @ X_local) + t
"""

from __future__ import annotations

import numpy as np

# Maps a Blender camera frame to an OpenCV camera frame (flip Y and Z).
BLENDER_TO_CV = np.diag([1.0, -1.0, -1.0])


# --------------------------------------------------------------------------
# Rotation utilities
# --------------------------------------------------------------------------
def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed rotation matrix via Shoemake's quaternion method.

    Sampling Euler angles uniformly would bias orientations toward the poles,
    which quietly skews the pose-estimation training distribution.
    """
    u1, u2, u3 = rng.random(3)
    a, b = np.sqrt(1.0 - u1), np.sqrt(u1)
    q = np.array([
        a * np.sin(2 * np.pi * u2),   # w
        a * np.cos(2 * np.pi * u2),   # x
        b * np.sin(2 * np.pi * u3),   # y
        b * np.cos(2 * np.pi * u3),   # z
    ])
    return matrix_from_quat(q)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a ``(w, x, y, z)`` quaternion."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` quaternion from a rotation matrix, sign-canonicalised.

    Uses the branch with the largest denominator for numerical stability, then
    forces ``w >= 0``. Without that canonicalisation ``q`` and ``-q`` denote the
    same rotation but give a large L2 loss during training.
    """
    m, tr = R, np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of ``R`` — the 6D rotation representation.

    Quaternions and Euler angles are both discontinuous as functions on SO(3),
    which caps regression accuracy. The 6D form (Zhou et al., CVPR 2019) is
    continuous, so Phase 2 regresses this and recovers ``R`` via Gram-Schmidt.
    """
    return R[:, :2].T.reshape(6)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Inverse of :func:`matrix_to_rot6d` via Gram-Schmidt orthonormalisation."""
    a1, a2 = r6.reshape(2, 3)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - (b1 @ a2) * b1
    b2 /= np.linalg.norm(b2)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def geodesic_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle of the relative rotation, in degrees — the Phase 3 pose metric."""
    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# --------------------------------------------------------------------------
# Camera placement
# --------------------------------------------------------------------------
def sample_viewpoint(rng, elev_deg, azim_deg, dist):
    """Sample a camera position on a spherical shell around the origin.

    Elevation is sampled uniformly in ``sin(elev)`` rather than in the angle, so
    viewpoints are uniform per unit *area* of the sphere instead of clustering
    near the poles.
    """
    s_lo, s_hi = np.sin(np.radians(elev_deg))
    elev = np.arcsin(rng.uniform(s_lo, s_hi))
    azim = np.radians(rng.uniform(*azim_deg))
    r = rng.uniform(*dist)
    return np.array([
        r * np.cos(elev) * np.cos(azim),
        r * np.cos(elev) * np.sin(azim),
        r * np.sin(elev),
    ])


def look_at(eye, target=(0.0, 0.0, 0.0), roll=0.0, world_up=(0.0, 0.0, 1.0)):
    """4x4 camera-to-world matrix in **Blender** convention (camera looks down -Z).

    ``roll`` rotates about the viewing axis, which is what actually gives the
    dataset in-plane rotation coverage; without it every render is gravity-aligned
    and the model never learns to handle a tilted camera.
    """
    eye = np.asarray(eye, dtype=float)
    fwd = np.asarray(target, dtype=float) - eye
    fwd /= np.linalg.norm(fwd)

    up = np.asarray(world_up, dtype=float)
    if abs(fwd @ up) > 0.999:          # looking straight up/down: pick a new up
        up = np.array([0.0, 1.0, 0.0])

    z_axis = -fwd                       # Blender camera's local +Z points backwards
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    c, s = np.cos(roll), np.sin(roll)
    R = R @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, eye
    return M


def world_to_cv_camera(cam_to_world: np.ndarray) -> np.ndarray:
    """Blender camera-to-world -> OpenCV world-to-camera (4x4)."""
    world_to_cam = np.linalg.inv(cam_to_world)
    flip = np.eye(4)
    flip[:3, :3] = BLENDER_TO_CV
    return flip @ world_to_cam


# --------------------------------------------------------------------------
# Intrinsics and projection
# --------------------------------------------------------------------------
def intrinsics(lens_mm, sensor_mm, res_x, res_y, sensor_fit="AUTO"):
    """Pinhole intrinsics ``K`` from Blender camera parameters.

    Under ``AUTO`` fit Blender applies the sensor size to the *larger* image
    dimension, which is the subtlety that silently produces a wrong focal length
    on non-square renders.
    """
    if sensor_fit == "AUTO":
        fit_px = max(res_x, res_y)
    elif sensor_fit == "HORIZONTAL":
        fit_px = res_x
    else:
        fit_px = res_y
    f_px = lens_mm / sensor_mm * fit_px
    return np.array([
        [f_px, 0.0, res_x / 2.0],
        [0.0, f_px, res_y / 2.0],
        [0.0, 0.0, 1.0],
    ])


def project(K, R, t, s, pts_local):
    """Project object-local points to pixels. Returns ``(uv[N,2], z[N])``.

    ``z`` is the camera-frame depth; callers should treat ``z <= 0`` as behind
    the camera and discard those points rather than trusting ``uv``.
    """
    pts = np.asarray(pts_local, dtype=float).reshape(-1, 3)
    cam = s * (pts @ R.T) + np.asarray(t, dtype=float)
    z = cam[:, 2]
    uv = (cam @ K.T)[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]
    return uv, z
