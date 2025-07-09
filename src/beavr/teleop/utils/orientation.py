from __future__ import annotations
from typing import Final
import numpy as np

"""Orientation / rotation helper functions.

Centralises quaternion ↔ axis-angle conversions so individual modules no longer
need to keep local re-implementations.  All quaternions follow the
``[x, y, z, w]`` convention and are *normalised with a non-negative* scalar part
(``w >= 0``) to guarantee a unique representation.

The minimal set of helpers intentionally avoids heavy dependencies – only NumPy
is required.  SciPy is used optionally for validation in unit tests but not at
run-time.
"""

_EPS: Final = 1e-8

# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def quat_normalise(q: np.ndarray) -> np.ndarray:
    """Return *unit* quaternion with the same sign as the input.

    No hemisphere enforcement here – use :pyfunc:`quat_positive` if you need a
    unique representative.
    """
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < _EPS:
        # treat as identity rotation
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / n


def quat_positive(q: np.ndarray) -> np.ndarray:
    """Normalise quaternion and ensure ``w >= 0`` (positive hemisphere)."""
    q = quat_normalise(q)
    if q[3] < 0:
        q = -q
    return q

# ---------------------------------------------------------------------------
# Quaternion ↔ axis–angle
# ---------------------------------------------------------------------------

def axis_angle_to_quat(v: np.ndarray) -> np.ndarray:
    """Convert axis–angle *vector* to quaternion.

    ``v = θ * a`` where *θ* is the rotation magnitude and *a* is the unit axis.
    """
    v = np.asarray(v, dtype=np.float32)
    theta = np.linalg.norm(v)
    if theta < _EPS:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = v / theta
    half = 0.5 * theta
    sin_h = np.sin(half)
    return quat_positive(np.concatenate([axis * sin_h, [np.cos(half)]]))


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert quaternion → *axis–angle vector* (rx, ry, rz)."""
    q = quat_positive(q)
    w = q[3]
    xyz = q[:3]
    sin_h = np.linalg.norm(xyz)
    theta = 2.0 * np.arctan2(sin_h, w)
    if theta > np.pi:
        theta -= 2.0 * np.pi  # map to (−π, π]
    if abs(theta) < _EPS:
        return np.zeros(3, dtype=np.float32)
    axis = xyz / sin_h
    return axis * theta

# ---------------------------------------------------------------------------
# Quaternion → Euler (roll, pitch, yaw)
# ---------------------------------------------------------------------------

def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert *unit* quaternion to Euler angles (roll, pitch, yaw).

    The convention follows *intrinsic* rotations about the **X → Y → Z** axes
    (often labelled "XYZ" or "roll–pitch–yaw").  The returned angles are in
    **radians** and lie within the usual ranges

        roll  ∈ (−π, π]
        pitch ∈ [−π/2, π/2]
        yaw   ∈ (−π, π]

    The input quaternion is first normalised and forced into the positive
    hemisphere (``w >= 0``) so the mapping is unique.
    """
    q = quat_positive(q)  # guarantees normalised, w >= 0

    x, y, z, w = q.astype(np.float32)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp_clamped = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp_clamped)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)

# ---------------------------------------------------------------------------
# Canonicalisation helper (kept for legacy callers)
# ---------------------------------------------------------------------------

def canonical_axis_angle(v: np.ndarray) -> np.ndarray:
    """Return a *canonical* axis–angle vector with θ ∈ [0, π]."""
    v = np.asarray(v, dtype=np.float32)
    theta = np.linalg.norm(v)
    if theta < _EPS:
        return v

    axis = v / theta
    half_theta = 0.5 * theta
    w = np.cos(half_theta)
    xyz = axis * np.sin(half_theta)

    # positive hemisphere
    if w < 0.0:
        w = -w
        xyz = -xyz

    new_theta = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < _EPS:
        new_axis = axis
    else:
        new_axis = xyz / s
    return new_axis * new_theta 