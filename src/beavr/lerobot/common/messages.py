from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping

import numpy as np

"""Common typed messages flowing between the three BeAVR layers.

This module defines two lightweight, **typed** containers that formalise the
contract between:

Operator  →  Interface  →  Controller

1. `OperatorCommand`
   Encapsulates the **desired** robot state produced by an *Operator*
   implementation (e.g. VR retargeting).

2. `RobotObservation`
   Represents the **measured** robot state returned by an *Interface*.

Both classes are implemented as `dataclasses` so they are by design
lightweight and have zero runtime-dependency overhead.  They also provide
small helper methods (`to_dict` / `from_dict`) so the rest of the, currently
_dict-based_, codebase can progressively migrate to the new contract without
breaking.

Usage example
-------------
>>> cmd = OperatorCommand(
...     pose=np.zeros(7),
...     gripper=0.0,
...     metadata={"frame": 123}
... )
>>> interface.apply(cmd)  # type: RobotObservation
"""

__all__ = [
    "Action",
    "Observation",
]


@dataclass(slots=True)
class Action:  # noqa: D101 – public API, docstring above.
    pose: np.ndarray  #: Either 7-D (x, y, z, qx, qy, qz, qw) or 4×4 hom. matrix.
    gripper: np.ndarray  #: Normalised gripper opening ∈ [0, 1] or physical unit.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # Serialisation helpers – *optional* usage while the codebase migrates.
    # ---------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation (uses `np.ndarray.tolist`)."""
        d = asdict(self)
        # Convert ndarray to list for JSON / msg-pack friendliness.
        d["pose"] = self.pose.tolist() if isinstance(self.pose, np.ndarray) else self.pose
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Action":
        """Create an :class:`Action` from a mapping produced by
        :py:meth:`to_dict`.
        """
        pose = np.asarray(data["pose"], dtype=np.float32)
        gripper = np.asarray(data["gripper"], dtype=np.float32)
        metadata = dict(data.get("metadata", {}))
        return cls(pose=pose, gripper=gripper, metadata=metadata)


@dataclass(slots=True)
class Observation:  # noqa: D101 – public API, docstring above.
    joint_state: np.ndarray  #: Shape (n_joints,) – *position* in [rad] / [m].
    end_effector_pose: np.ndarray  #: 7-D (pos + quat) or 4×4 hom. matrix.
    flags: Dict[str, Any] = field(default_factory=dict)

    # Similar helper utilities as for Action
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["joint_state"] = (
            self.joint_state.tolist() if isinstance(self.joint_state, np.ndarray) else self.joint_state
        )
        d["end_effector_pose"] = (
            self.end_effector_pose.tolist()
            if isinstance(self.end_effector_pose, np.ndarray)
            else self.end_effector_pose
        )
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Observation":
        joint_state = np.asarray(data["joint_state"], dtype=np.float32)
        end_effector_pose = np.asarray(data["end_effector_pose"], dtype=np.float32)
        flags = dict(data.get("flags", {}))
        return cls(joint_state=joint_state, end_effector_pose=end_effector_pose, flags=flags)