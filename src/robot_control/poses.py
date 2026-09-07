"""Taught poses: a YAML file per profile, one entry per name.

A pose is saved with the groups it was taught on — an arm, or an arm and its
gripper together — and the gravity scale that was holding it, because a pose
taught with compensation on sits a droop away from the same pose without it.
``teach goto`` reads both back so the arm is put where it was, not where the
same numbers land under a different torque.

The file is read and written whole. Every function returns a new mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .layout import repository_root

SCHEMA = 1
POSES_DIR = "poses"


class PoseStoreError(ValueError):
    pass


@dataclass(frozen=True)
class TaughtPose:
    name: str
    #: The groups whose joints this pose carries, in the order they were taught.
    groups: tuple[str, ...]
    #: Canonical joint name -> value, across every group.
    joints: Mapping[str, float]
    #: Gravity feedforward scale per compensated joint, by joint name, or None.
    gravity: Mapping[str, float] | None
    saved_at: str

    def __post_init__(self) -> None:
        groups = tuple(str(group) for group in self.groups)
        if not self.name or not groups or not all(groups):
            raise PoseStoreError("a taught pose needs a name and at least one group")
        object.__setattr__(self, "groups", groups)
        joints = {}
        for key, value in dict(self.joints).items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise PoseStoreError(f"pose {self.name!r}: {key} is not a number") from None
            if not math.isfinite(number):
                raise PoseStoreError(f"pose {self.name!r}: {key} is not finite")
            joints[str(key)] = number
        if not joints:
            raise PoseStoreError(f"pose {self.name!r} has no joints")
        object.__setattr__(self, "joints", joints)
        if self.gravity is not None:
            gravity = {str(key): float(value) for key, value in dict(self.gravity).items()}
            if not all(math.isfinite(value) for value in gravity.values()):
                raise PoseStoreError(f"pose {self.name!r}: gravity scale is not finite")
            object.__setattr__(self, "gravity", gravity)


def default_poses_path(profile_name: str) -> Path:
    return repository_root() / POSES_DIR / f"{profile_name}.yaml"


def load_poses(path: Path, profile: str) -> dict[str, TaughtPose]:
    """Read the store, or an empty one if the file does not exist yet."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise PoseStoreError(f"cannot read {path}: {error}") from error
    if not isinstance(raw, dict):
        raise PoseStoreError(f"{path} is not a pose store (expected a mapping)")
    if raw.get("schema") != SCHEMA:
        raise PoseStoreError(f"{path} has schema {raw.get('schema')!r}, expected {SCHEMA}")
    if raw.get("profile") != profile:
        raise PoseStoreError(
            f"{path} holds poses for profile {raw.get('profile')!r}, not {profile!r}"
        )
    entries = raw.get("poses")
    if not isinstance(entries, dict):
        raise PoseStoreError(f"{path}: 'poses' must be a mapping of name to pose")
    return {str(name): _from_yaml(str(name), body) for name, body in entries.items()}


def save_poses(path: Path, poses: Mapping[str, TaughtPose], profile: str) -> None:
    """Write the whole store, replacing the file only once it is complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": SCHEMA,
        "profile": profile,
        "poses": {name: _to_yaml(pose) for name, pose in sorted(poses.items())},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
    os.replace(temporary, path)


def with_pose(poses: Mapping[str, TaughtPose], pose: TaughtPose) -> dict[str, TaughtPose]:
    return {**dict(poses), pose.name: pose}


def pose_values(pose: TaughtPose, group: Any) -> np.ndarray:
    """The pose in *group*'s canonical joint order, refusing a mismatch."""
    if group.name not in pose.groups:
        raise PoseStoreError(
            f"pose {pose.name!r} was taught on {', '.join(pose.groups)}, not {group.name}"
        )
    missing = [joint for joint in group.joints if joint not in pose.joints]
    if missing:
        raise PoseStoreError(f"pose {pose.name!r} lacks joints {missing}")
    return np.array([pose.joints[joint] for joint in group.joints], dtype=float)


def _to_yaml(pose: TaughtPose) -> dict[str, Any]:
    return {
        "groups": list(pose.groups),
        "saved_at": pose.saved_at,
        "gravity": None if pose.gravity is None else dict(pose.gravity),
        "joints": dict(pose.joints),
    }


def _from_yaml(name: str, body: Any) -> TaughtPose:
    if not isinstance(body, dict):
        raise PoseStoreError(f"pose {name!r} is not a mapping")
    joints = body.get("joints")
    if not isinstance(joints, dict):
        raise PoseStoreError(f"pose {name!r}: 'joints' must be a mapping")
    gravity = body.get("gravity")
    if gravity is not None and not isinstance(gravity, dict):
        raise PoseStoreError(f"pose {name!r}: 'gravity' must be a mapping or null")
    groups = body.get("groups", [])
    if not isinstance(groups, list):
        raise PoseStoreError(f"pose {name!r}: 'groups' must be a list")
    return TaughtPose(
        name=name,
        groups=tuple(groups),
        joints=joints,
        gravity=gravity,
        saved_at=str(body.get("saved_at", "")),
    )
