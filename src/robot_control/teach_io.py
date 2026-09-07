"""Files a teaching session leaves behind: recordings and taught poses.

A recording is one ``.npz`` holding every joint of every group the session
drove, side by side, so an arm and its gripper replay from one file. The
gravity scale that was holding each compensated joint travels with it, keyed
by joint name, because a pose taught under compensation sits a droop away from
the same numbers without it.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .poses import TaughtPose
from .teaching import trim_still
from .track import Recording, TrackError

RECORDING_SCHEMA = 2


def write_recording(
    path: Path,
    recording: Recording,
    profile_name: str,
    groups: Sequence[str],
    gravity: Mapping[str, float] | None,
    trim_rad: float,
) -> Recording:
    """Trim the still time off both ends and write. Returns what was written."""
    if path.suffix != ".npz":
        raise ValueError(f"a recording must end in .npz (numpy appends it otherwise): {path}")
    trimmed = trim_still(recording, trim_rad)
    gravity = {} if gravity is None else dict(gravity)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schema=RECORDING_SCHEMA,
        profile=profile_name,
        groups=np.array(list(groups)),
        joint_names=np.array(trimmed.joint_names),
        timestamps_ns=trimmed.timestamps_ns,
        values=trimmed.values,
        gravity_joints=np.array(list(gravity)),
        gravity_scales=np.array(list(gravity.values()), dtype=float),
    )
    return trimmed


def read_recording(
    path: Path, profile_name: str
) -> tuple[Recording, tuple[str, ...], dict[str, float]]:
    """The recording, the groups it covers, and the gravity scale by joint."""
    try:
        raw = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise TrackError(f"cannot read recording {path}: {error}") from error
    try:
        if int(raw["schema"]) != RECORDING_SCHEMA:
            raise TrackError(
                f"{path} has schema {raw['schema']}, expected {RECORDING_SCHEMA}"
            )
        if str(raw["profile"]) != profile_name:
            raise TrackError(
                f"{path} was recorded on profile {str(raw['profile'])!r}, "
                f"not {profile_name!r}"
            )
        recording = Recording(
            raw["timestamps_ns"],
            raw["values"],
            tuple(str(name) for name in raw["joint_names"]),
        )
        groups = tuple(str(name) for name in raw["groups"])
        gravity = {
            str(joint): float(scale)
            for joint, scale in zip(raw["gravity_joints"], raw["gravity_scales"])
        }
    except KeyError as error:
        raise TrackError(f"{path} is not a teach recording: missing {error}") from None
    return recording, groups, gravity


def merge_recordings(primary: Recording, others: Sequence[Recording]) -> Recording:
    """Put recordings from other joint-state topics on the primary's stamps.

    A hand under its own controller_manager publishes on its own clock, so its
    samples never line up with the arm's. Each other recording is interpolated
    onto the primary's stamps; the primary keeps its arrival pattern, and its
    stamps are what the merged recording carries.
    """
    stamps = primary.timestamps_ns
    columns = [primary.values]
    names = list(primary.joint_names)
    for other in others:
        order = np.unique(other.timestamps_ns, return_index=True)[1]
        source_stamps = other.timestamps_ns[order].astype(float)
        source = other.values[order]
        if len(source_stamps) < 2:
            raise TrackError(
                f"too few samples of {', '.join(other.joint_names)} to merge"
            )
        columns.append(
            np.column_stack(
                [
                    np.interp(stamps.astype(float), source_stamps, source[:, index])
                    for index in range(source.shape[1])
                ]
            )
        )
        names.extend(other.joint_names)
    return Recording(
        stamps, np.hstack(columns), tuple(names),
        incomplete=primary.incomplete + sum(other.incomplete for other in others),
    )


def group_columns(joint_names: Sequence[str], group: Any) -> np.ndarray:
    """Where *group*'s joints sit in a recording's columns, by name."""
    lookup = {name: index for index, name in enumerate(joint_names)}
    missing = [joint for joint in group.joints if joint not in lookup]
    if missing:
        raise TrackError(
            f"the recording carries no {', '.join(missing)}; it covers "
            f"{', '.join(joint_names)}"
        )
    return np.array([lookup[joint] for joint in group.joints])


def taught_pose(
    name: str,
    measured: Sequence[tuple[Any, np.ndarray]],
    gravity: Mapping[str, float] | None,
) -> TaughtPose:
    """A pose spanning every (group, values) pair given."""
    joints = {}
    for group, values in measured:
        joints.update(zip(group.joints, (float(value) for value in values)))
    return TaughtPose(
        name=name,
        groups=tuple(group.name for group, _ in measured),
        joints=joints,
        gravity=None if not gravity else dict(gravity),
        saved_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )


def gravity_by_joint(lanes: Sequence[Any]) -> dict[str, float]:
    """The scale holding each compensated joint, across lanes, by joint name."""
    scales = {}
    for lane in lanes:
        if lane.scales is not None:
            scales.update(zip(lane.group.joints, (float(s) for s in lane.scales)))
    return scales
