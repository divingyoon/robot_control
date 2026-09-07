"""`robotctl teach`: poses and trajectories taught by hand.

    teach hold    let a hand move the arm; it keeps whatever pose it is left in
    teach record  the same, recording the motion to a file
    teach replay  play a recording back, repeatedly if asked
    teach save    name the pose the arm is in
    teach goto    put the arm back in a named pose
    teach list    the poses on file

Every session is the servo loop `pose follow` runs — a command per period on
the trajectory controller's stream topic, gravity feedforward beside it — with
a different source of commands: the teaching state machine, or a recording.
The gate clamps every streamed sample; a replay is also authorized whole,
before the arm moves, so a recording that asks for more than the profile
allows is refused rather than played back slowly.

A session drives one **lane** per group given — an arm, an arm and its
gripper, an arm and a hand — so all are taught, recorded and replayed
together. Lanes on the same joint-state topic share one backend; a hand under
its own controller_manager gets a backend of its own, and its recording is
merged onto the arm's clock. Gravity feedforward goes only to lanes that have
an effort controller; a gripper or a hand has none and needs none.

The Tesollo hand is one lane, ``tesollo_hand``, made here from the profile's
four phalanx groups. The profile itself keeps every joint in exactly one
group, which the calibration pipeline relies on; and a hand streamed as four
partial trajectories would have each message refill the other three's joints
with their measured position, undoing their hold.

The gains this works against are fixed in the vendor hardware (see
``teaching``), which is why nothing here talks to the motors about stiffness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from . import cli as _cli
from .keys import KeyReader
from .poses import (
    PoseStoreError,
    default_poses_path,
    load_poses,
    pose_values,
    save_poses,
    with_pose,
)
from .profile import load_builtin_profile
from .safety import SafetyError
from .teach_io import (
    gravity_by_joint,
    group_columns,
    merge_recordings,
    read_recording,
    taught_pose,
    write_recording,
)
from .teaching import (
    TeachLimits,
    TeachState,
    acknowledge,
    resample,
    start,
    step,
    toggle_lock,
)

# Deflection beyond the standing droop that counts as a hand pushing. At the
# shoulders' kp 70 this is about 1.4 N.m; at the wrists' kp 10, 0.2 N.m. A
# joint whose whole range is small — a gripper's stroke is 0.044 — gets a
# fraction of that range instead, or one push would be half its travel.
DEFAULT_PUSH_RAD = 0.02
DEFAULT_STILL_RAD_S = 0.03
RANGE_FRACTION = 0.05
DEFAULT_STILL_SEC = 0.5
DEFAULT_SETTLE_SEC = 1.0
# A push that goes on this long without the arm ever resting is drift. It
# also caps one taught motion: a demonstration longer than this has to pause.
DEFAULT_MAX_FOLLOW_SEC = 30.0
# A session ends on its own: the right wrist has overheated holding a pose for
# eighteen minutes, and a compliant arm left running is one that moves when
# somebody brushes it later.
DEFAULT_SESSION_SEC = 300.0
DEFAULT_TRIM_RAD = 0.005
DEFAULT_SPEED = 1.0
DEFAULT_REPEAT = 1
DEFAULT_HOLD_SEC = 0.0
DEFAULT_NAME_PREFIX = "taught"
NODE_NAME = "robot_control_teach"
# How long one lane's joint state may stay silent before the session ends. A
# hand under its own driver drops a message now and then; ending the whole
# session — and releasing the arm's gravity torque mid-teach — over one
# missed sample is worse than skipping that lane for a cycle.
STATE_TIMEOUT_SEC = 0.2
STALE_LIMIT_SEC = 2.0
#: Lanes made by joining profile groups that share a controller, by name.
COMPOSITE_GROUPS = {
    "tesollo_hand": ("tesollo_abduction", "tesollo_curl", "tesollo_pip", "tesollo_dip"),
}

KEY_LOCK = " "
KEY_SAVE = "s"
KEY_QUIT = "q"


def add_parser(commands: argparse._SubParsersAction) -> None:
    teach = commands.add_parser("teach", help="teach poses and trajectories by hand")
    stages = teach.add_subparsers(dest="stage", required=True)

    hold = stages.add_parser("hold", help="move the arm by hand; it keeps the pose")
    _add_session_args(hold, groups_required=True)
    _add_teach_args(hold)

    record = stages.add_parser("record", help="teach hold, recording the motion")
    _add_session_args(record, groups_required=True)
    _add_teach_args(record)
    record.add_argument("--output", type=Path, required=True, help="recording (.npz)")
    record.add_argument(
        "--trim-rad", type=float, default=DEFAULT_TRIM_RAD,
        help="drop the still time before and after the motion, judged at this",
    )

    replay = stages.add_parser("replay", help="play a recording back")
    _add_session_args(replay, groups_required=False)
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    replay.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="time scale")
    replay.add_argument("--hold-sec", type=float, default=DEFAULT_HOLD_SEC)
    _add_approach_arg(replay)

    save = stages.add_parser("save", help="name the pose the arm is in")
    save.add_argument("--profile", default="openarm_tesollo")
    _add_group_arg(save, required=True)
    save.add_argument("--name", required=True)
    save.add_argument("--poses", type=Path, help="pose store; default poses/<profile>.yaml")
    save.add_argument("--gravity", help="scale that was holding it, recorded with the pose")

    goto = stages.add_parser("goto", help="put the arm back in a named pose")
    _add_session_args(goto, groups_required=False)
    goto.add_argument("--name", required=True)
    goto.add_argument("--poses", type=Path)
    goto.add_argument("--hold-sec", type=float, default=DEFAULT_HOLD_SEC)
    _add_approach_arg(goto)

    listing = stages.add_parser("list", help="the poses on file")
    listing.add_argument("--profile", default="openarm_tesollo")
    listing.add_argument("--poses", type=Path)


def _add_group_arg(parser: argparse.ArgumentParser, required: bool) -> None:
    parser.add_argument(
        "--group", action="append", required=required,
        help="a group to drive; repeatable, e.g. an arm and its gripper"
        + ("" if required else ". Default: the groups the file was made with"),
    )


def _add_session_args(parser: argparse.ArgumentParser, groups_required: bool) -> None:
    parser.add_argument("--profile", default="openarm_tesollo")
    _add_group_arg(parser, required=groups_required)
    parser.add_argument(
        "--gravity",
        help="gravity feedforward scale for every compensable group: one value, "
        "or one per joint of each",
    )
    parser.add_argument("--payload", help=_cli._PAYLOAD_HELP)
    parser.add_argument("--urdf", type=Path, help=_cli._URDF_OVERRIDE_HELP)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SESSION_SEC)
    parser.add_argument(
        "--keep-gravity", action="store_true",
        help="leave the feedforward torque published when the session ends",
    )
    parser.add_argument("--execute", action="store_true")


def _add_teach_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--push-rad",
        help=f"deflection that counts as a push: one value for every joint, or "
        f"one per joint across the groups in order. Default {DEFAULT_PUSH_RAD:g}, "
        f"or {RANGE_FRACTION:.0%} of a joint's range if that is smaller",
    )
    parser.add_argument("--still-sec", type=float, default=DEFAULT_STILL_SEC)
    parser.add_argument("--settle-sec", type=float, default=DEFAULT_SETTLE_SEC)
    parser.add_argument(
        "--max-follow-sec", type=float, default=DEFAULT_MAX_FOLLOW_SEC,
        help="a push lasting this long with no rest is latched as drift; it also "
        "caps one uninterrupted taught motion",
    )
    parser.add_argument("--poses", type=Path, help="where the s key saves poses")
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument(
        "--soft-p", action="append", metavar="GROUP=P",
        help="lower a group's controller PID p to P for the session and restore "
        "it after, for a hand whose own loop makes it too stiff to push; repeatable",
    )


def _add_approach_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--approach-rad-s", type=float, default=_cli.PARK_SPEED_RAD_PER_SEC,
        help="joint speed of the move to the first pose",
    )


def _with_composites(profile):
    """The profile plus every composite group whose members it has."""
    import dataclasses

    groups = dict(profile.groups)
    for name, members in COMPOSITE_GROUPS.items():
        if name in groups or not all(member in groups for member in members):
            continue
        first = groups[members[0]]
        for member in members[1:]:
            other = groups[member]
            if other.controller != first.controller or other.state_topic != first.state_topic:
                raise ValueError(
                    f"composite group {name!r}: {member} is not on {members[0]}'s "
                    "controller and state topic"
                )
        groups[name] = dataclasses.replace(
            first,
            name=name,
            joints=tuple(j for member in members for j in groups[member].joints),
            moveit_group=None,
            tip_link=None,
            asset_tip_link=None,
            hdgp_group=None,
        )
    return dataclasses.replace(profile, groups=groups)


def run(args: argparse.Namespace) -> int:
    from .ros_adapter import AdapterUnavailable

    try:
        profile = _with_composites(load_builtin_profile(args.profile))
        handler = {
            "hold": _hold,
            "record": _record,
            "replay": _replay,
            "save": _save,
            "goto": _goto,
            "list": _list,
        }[args.stage]
        return handler(args, profile)
    except SafetyError as error:
        print(f"refused: {error}")
        return _cli.REFUSED
    except AdapterUnavailable as error:
        print(f"unavailable: {error}")
        return _cli.UNUSABLE
    except (ValueError, OSError) as error:
        # PoseStoreError, TrackError, ProfileError are all ValueError.
        print(f"error: {error}")
        return _cli.UNUSABLE


# --- lanes and the session ---------------------------------------------------------


@dataclass(frozen=True)
class Lane:
    """One group in a session: its adapter, its gate, and its gravity model."""

    group: Any
    adapter: Any
    gate: Any
    chain: Any
    scales: np.ndarray | None


@dataclass(frozen=True)
class Session:
    lanes: tuple[Lane, ...]
    period: float
    seconds: float
    keep_gravity: bool

    @property
    def topic_lanes(self) -> list[Lane]:
        """One lane per joint-state topic: the one whose adapter pumps and
        records that topic's backend, on behalf of every lane sharing it."""
        seen, first = set(), []
        for lane in self.lanes:
            topic = lane.group.state_topic
            if topic not in seen:
                seen.add(topic)
                first.append(lane)
        return first


class TeachController:
    """The teaching state machine, given the clock."""

    def __init__(self, measured, limits: TeachLimits, settle_sec: float):
        self.limits = limits
        self.state: TeachState = start(measured, time.monotonic(), settle_sec)

    def step(self, measured, now: float):
        self.state, desired, event = step(self.state, self.limits, measured, now)
        return desired, False, event

    def acknowledge(self, command) -> None:
        self.state = acknowledge(self.state, command)

    def toggle_lock(self) -> str:
        self.state = toggle_lock(self.state)
        return "locked: the arm will not follow a push" if self.state.locked else "unlocked"


class TrajectoryController:
    """Streams segments of points in turn, then holds the last for a while."""

    def __init__(self, segments: Sequence[np.ndarray], hold_sec: float):
        self.points = [point for segment in segments for point in segment]
        self.index = 0
        self.hold_sec = hold_sec
        self.finished_at: float | None = None

    def step(self, measured, now: float):
        if self.index < len(self.points):
            point = self.points[self.index]
            self.index += 1
            return point, False, None
        if self.finished_at is None:
            self.finished_at = now
            return None, self.hold_sec <= 0, "played"
        return None, now - self.finished_at >= self.hold_sec, None

    def acknowledge(self, command) -> None:
        return None


def _groups(profile, names: Sequence[str]):
    groups = [_cli._group(profile, name) for name in names]
    if len({group.name for group in groups}) != len(groups):
        raise ValueError(f"a group is listed twice in {list(names)}")
    controllers = {}
    for group in groups:
        if group.controller in controllers:
            raise ValueError(
                f"{group.name} and {controllers[group.controller]} stream to the same "
                f"controller {group.controller}; a partial stream refills the joints it "
                "does not name with their measured position, so drive them as one group"
            )
        controllers[group.controller] = group.name
    return groups


def _lane_scales(gravity: str | None, groups) -> list[np.ndarray | None]:
    """Gravity scales per group: --gravity for every compensable one, None for
    the rest. Validated before anything opens the robot: a bad scale is a
    typed mistake, and the answer to it should not be 'no ROS'."""
    if gravity is None:
        return [None] * len(groups)
    if not any(group.compensable for group in groups):
        raise ValueError(
            "--gravity needs a group with an effort_controller; none of "
            f"{[group.name for group in groups]} declares one"
        )
    scales = []
    for group in groups:
        if not group.compensable:
            scales.append(None)
            continue
        vector = _cli._scale_vector(gravity, group)
        _cli._check_scales(vector, group)
        scales.append(vector if np.any(vector) else None)
    return scales


class _Backends:
    """One backend per joint-state topic the session's groups publish on."""

    def __init__(self, groups):
        from .ros_adapter import JOINT_STATES_TOPIC, make_backend

        self._by_topic = {}
        for group in groups:
            topic = group.state_topic or JOINT_STATES_TOPIC
            if topic not in self._by_topic:
                self._by_topic[topic] = make_backend(
                    f"{NODE_NAME}_{len(self._by_topic)}", topic
                )

    def for_group(self, group):
        from .ros_adapter import JOINT_STATES_TOPIC

        return self._by_topic[group.state_topic or JOINT_STATES_TOPIC]

    def close(self) -> None:
        for one in self._by_topic.values():
            one.close()


def _open(profile, groups, execute: bool):
    """The session's backends, and one adapter per group on its topic's."""
    from .ros_adapter import RosAdapter

    backends = _Backends(groups)
    adapters = [
        RosAdapter(profile, group.name, execute=execute, backend=backends.for_group(group))
        for group in groups
    ]
    return backends, adapters


def _lanes(args, profile, groups, adapters, scales, seeds) -> tuple[Lane, ...]:
    lanes = []
    for group, adapter, scale, seed in zip(groups, adapters, scales, seeds):
        chain = None
        if scale is not None:
            chain = _cli._gravity_chain(
                adapter, profile, group, args.urdf, getattr(args, "payload", None)
            )
        lanes.append(Lane(group, adapter, _cli._gate(profile, group, seed=seed), chain, scale))
    return tuple(lanes)


def _session(args, profile, lanes) -> Session:
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    return Session(
        lanes=tuple(lanes),
        period=1.0 / profile.endpoint().command_rate_hz,
        seconds=args.seconds,
        keep_gravity=args.keep_gravity,
    )


def _run_session(
    session: Session,
    controllers: Sequence[Any],
    on_key: Callable[[str, dict[str, np.ndarray]], str | None],
) -> int:
    """Stream commands until every controller is done, a key quits, or time runs out."""
    from .ros_adapter import AdapterUnavailable

    samples, notes = 0, {}
    stale_since: dict[str, float] = {}
    at_limit: dict[str, float] = {}
    deadline = time.monotonic() + session.seconds
    try:
        with KeyReader() as keys:
            while time.monotonic() < deadline:
                cycle = time.monotonic()
                for lane in session.topic_lanes:
                    lane.adapter.pump(timeout_sec=0.0)
                measured = {}
                for lane in session.lanes:
                    name = lane.group.name
                    try:
                        measured[name] = lane.adapter.read_state(timeout_sec=STATE_TIMEOUT_SEC)
                    except AdapterUnavailable as error:
                        first = stale_since.setdefault(name, cycle)
                        if cycle - first > STALE_LIMIT_SEC:
                            raise AdapterUnavailable(
                                f"{name}: no joint state for {cycle - first:.1f} s: {error}"
                            ) from error
                        continue
                    if name in stale_since:
                        print(f"{name}: joint state back after {cycle - stale_since.pop(name):.2f} s")
                key = keys.poll()
                if key == KEY_QUIT:
                    print("quit")
                    break
                if key is not None:
                    note = on_key(key, measured)
                    if note:
                        print(note)
                done = True
                for lane, controller in zip(session.lanes, controllers):
                    here = measured.get(lane.group.name)
                    if here is None:
                        done = False  # a silent lane is not a finished one
                        continue
                    _push_gravity(lane, here)
                    desired, lane_done, event = controller.step(here, cycle)
                    done = done and lane_done
                    if desired is not None:
                        command, limited = lane.gate.follow(desired, here, session.period)
                        if limited is not None:
                            notes[limited] = notes.get(limited, 0) + 1
                            if "position" in limited:
                                _say_at_limit(lane, command, cycle, at_limit)
                        lane.adapter.stream_positions(command)
                        controller.acknowledge(command)
                        samples += 1
                    if event:
                        elapsed = time.monotonic() - deadline + session.seconds
                        print(f"{elapsed:7.2f} s  {lane.group.name}: {event}")
                if done:
                    break
                time.sleep(max(0.0, session.period - (time.monotonic() - cycle)))
            else:
                print(f"time is up after {session.seconds:g} s")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        _release(session)
        print(f"streamed {samples} samples; the arm holds its last commanded pose")
        for note, count in sorted(notes.items()):
            print(f"  {note} clamped on {count} of {samples} samples")
    return 0


# A joint held at its limit feels like a joint that has seized: the command
# stops following the hand there and the full stiffness pushes back. Say which
# one, and which limit, as it happens — the end-of-session count cannot.
LIMIT_NOTICE_SEC = 2.0


def _say_at_limit(lane: Lane, command: np.ndarray, now: float, said: dict[str, float]) -> None:
    gate = lane.gate
    for index in np.flatnonzero((command <= gate.lower + 1e-9) | (command >= gate.upper - 1e-9)):
        name = lane.group.joints[index]
        if now - said.get(name, -LIMIT_NOTICE_SEC) < LIMIT_NOTICE_SEC:
            continue
        said[name] = now
        which = "lower" if command[index] <= gate.lower[index] + 1e-9 else "upper"
        bound = gate.lower[index] if which == "lower" else gate.upper[index]
        print(f"{name} is at its {which} limit ({bound:+.3f}); it will not follow past it")


def _push_gravity(lane: Lane, measured: np.ndarray) -> None:
    if lane.scales is None:
        return
    lane.adapter.send_effort(
        lane.gate.authorize_effort(lane.chain.gravity_torque(measured) * lane.scales)
    )


def _release(session: Session) -> None:
    compensated = [lane for lane in session.lanes if lane.scales is not None]
    if not compensated:
        return
    if session.keep_gravity:
        for lane in compensated:
            print(
                "gravity feedforward is still published for the pose the arm is in. "
                f"Release it before moving the arm any other way: robotctl pose "
                f"gravity --group {lane.group.name} --scale 0 --execute"
            )
        return
    for lane in compensated:
        lane.adapter.send_effort(np.zeros(len(lane.group.joints)))
    print(
        "gravity feedforward released: the arm will sag by its droop. "
        "--keep-gravity leaves the torque on."
    )


# --- hold / record ---------------------------------------------------------------


def _default_thresholds(profile, group, default: float) -> np.ndarray:
    """*default*, or RANGE_FRACTION of the joint's range where that is smaller."""
    limits = _cli._joint_limits(profile, group)
    return np.array(
        [min(default, RANGE_FRACTION * (joint.upper - joint.lower)) for joint in limits]
    )


def _limits_for(args, profile, groups) -> list[TeachLimits]:
    pushes = [_default_thresholds(profile, group, DEFAULT_PUSH_RAD) for group in groups]
    if args.push_rad is not None:
        given = _cli._parse_floats(args.push_rad, "--push-rad")
        total = sum(len(group.joints) for group in groups)
        if len(given) == 1:
            given = given * total
        if len(given) != total:
            raise ValueError(
                f"--push-rad needs one value or one per joint: the groups have "
                f"{total} joints, got {len(given)}"
            )
        pushes, offset = [], 0
        for group in groups:
            pushes.append(np.asarray(given[offset : offset + len(group.joints)]))
            offset += len(group.joints)
    return [
        TeachLimits(
            push_rad=push,
            still_rad_s=_default_thresholds(profile, group, DEFAULT_STILL_RAD_S),
            still_sec=args.still_sec,
            settle_sec=args.settle_sec,
            max_follow_sec=args.max_follow_sec,
        )
        for group, push in zip(groups, pushes)
    ]


def _hold(args, profile) -> int:
    return _teach(args, profile, output=None)


def _record(args, profile) -> int:
    if args.trim_rad <= 0:
        raise ValueError("--trim-rad must be positive")
    if args.output.suffix != ".npz":
        raise ValueError(f"--output must end in .npz (numpy appends it otherwise): {args.output}")
    return _teach(args, profile, output=args.output)


def _soft_gains(given: Sequence[str] | None, groups) -> dict[str, float]:
    """--soft-p as {group name: p}, refusing a group not in the session."""
    soft = {}
    for item in given or ():
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--soft-p needs GROUP=P, got {item!r}")
        if name not in {group.name for group in groups}:
            raise ValueError(f"--soft-p {name!r} is not one of the session's groups")
        try:
            p = float(value)
        except ValueError:
            raise ValueError(f"--soft-p {name}: {value!r} is not a number") from None
        if not p > 0:
            raise ValueError(f"--soft-p {name}: p must be positive, got {p:g}")
        soft[name] = p
    return soft


class _SoftGains:
    """Lower the named controllers' p for the session; put it back afterwards.

    The restore runs whether the session ended, quit, or failed: a hand left
    at a teaching gain cannot hold what it is later told to grasp.
    """

    def __init__(self, lanes, soft: dict[str, float]):
        self._lanes = [lane for lane in lanes if lane.group.name in soft]
        self._soft = soft
        self._previous: dict[str, dict[str, float]] = {}

    def __enter__(self):
        for lane in self._lanes:
            before = lane.adapter.read_gain_p()
            if not before:
                raise ValueError(
                    f"{lane.group.controller} declares no PID gains to soften"
                )
            self._previous[lane.group.name] = before
            p = self._soft[lane.group.name]
            lane.adapter.write_gain_p({joint: p for joint in before})
            print(
                f"{lane.group.name}: controller p {_cli._scale_label(np.array(list(before.values())), None)} "
                f"-> {p:g} for the session"
            )
        return self

    def __exit__(self, *_exception):
        for lane in self._lanes:
            before = self._previous.get(lane.group.name)
            if before is None:
                continue
            lane.adapter.write_gain_p(before)
            print(f"{lane.group.name}: controller p restored")


def _teach(args, profile, output: Path | None) -> int:
    groups = _groups(profile, args.group)
    limits = _limits_for(args, profile, groups)
    scales = _lane_scales(args.gravity, groups)
    soft = _soft_gains(args.soft_p, groups)
    poses_path = args.poses or default_poses_path(profile.name)
    backends, adapters = _open(profile, groups, args.execute)
    try:
        measured = [
            _cli._start_pose(profile, group, adapter.read_state())
            for group, adapter in zip(groups, adapters)
        ]
        lanes = _lanes(args, profile, groups, adapters, scales, [None] * len(groups))
        session = _session(args, profile, lanes)
        _announce(session, limits, poses_path, output, soft)
        if not args.execute:
            print("DRY RUN: nothing is published; pass --execute to teach")
            return 0
        controllers = [
            TeachController(here, limit, args.settle_sec)
            for here, limit in zip(measured, limits)
        ]
        saver = _Saver(profile, lanes, poses_path, args.name_prefix)

        def on_key(key: str, here: dict[str, np.ndarray]) -> str | None:
            if key == KEY_LOCK:
                return "; ".join(controller.toggle_lock() for controller in controllers)
            if key == KEY_SAVE:
                return saver.save(here)
            return None

        if output is not None:
            for lane in session.topic_lanes:
                lane.adapter.start_recording()
        try:
            with _SoftGains(lanes, soft):
                code = _run_session(session, controllers, on_key)
        finally:
            # What was taught before a failure is still worth keeping: a hand
            # that drops its joint states mid-session took the demonstration
            # so far with it otherwise.
            if output is not None:
                _save_recording(session, output, profile, groups, lanes, args.trim_rad)
    finally:
        backends.close()
    return code


def _save_recording(session, output, profile, groups, lanes, trim_rad) -> None:
    from .ros_adapter import AdapterUnavailable
    from .track import TrackError

    try:
        recording = _stop_recording(session)
        trimmed = write_recording(
            output, recording, profile.name, [g.name for g in groups],
            gravity_by_joint(lanes), trim_rad,
        )
    except (AdapterUnavailable, TrackError) as error:
        print(f"nothing recorded: {error}")
        return
    _report_recording(output, recording, trimmed)


def _announce(session, limits, poses_path, output, soft) -> None:
    names = ", ".join(lane.group.name for lane in session.lanes)
    print(
        f"teaching {names} at {1.0 / session.period:g} Hz for up to "
        f"{session.seconds:g} s"
    )
    for lane, limit in zip(session.lanes, limits):
        gravity = "off" if lane.scales is None else _cli._scale_label(lane.scales, None)
        print(
            f"  {lane.group.name}: gravity scale {gravity}; push threshold "
            f"{' '.join(f'{v:.4g}' for v in limit.push_rad)}"
        )
    print(
        f"  latch after {limits[0].still_sec:g} s still; drift cut at "
        f"{limits[0].max_follow_sec:g} s"
    )
    print(f"  keys: space = lock/unlock, {KEY_SAVE} = save pose to {poses_path}, "
          f"{KEY_QUIT} = quit")
    if output is not None:
        print(f"  recording to {output}")
    for name, p in soft.items():
        print(f"  {name}: controller p lowered to {p:g} while teaching")


class _Saver:
    def __init__(self, profile, lanes, path: Path, prefix: str):
        self.profile, self.lanes, self.path, self.prefix = profile, lanes, path, prefix
        self.poses = load_poses(path, profile.name)

    def save(self, measured: dict[str, np.ndarray]) -> str:
        index = 1
        while f"{self.prefix}_{index}" in self.poses:
            index += 1
        pose = taught_pose(
            f"{self.prefix}_{index}",
            [(lane.group, measured[lane.group.name]) for lane in self.lanes],
            gravity_by_joint(self.lanes),
        )
        self.poses = with_pose(self.poses, pose)
        save_poses(self.path, self.poses, self.profile.name)
        return f"saved {pose.name} to {self.path}"


def _stop_recording(session: Session):
    """Every lane's joints, on the first topic's clock, in lane order."""
    from .track import Recording

    per_topic = {
        lane.group.state_topic: lane.adapter.stop_recording_groups(
            [other.group.name for other in session.lanes
             if other.group.state_topic == lane.group.state_topic]
        )
        for lane in session.topic_lanes
    }
    first = session.topic_lanes[0].group.state_topic
    others = [recording for topic, recording in per_topic.items() if topic != first]
    merged = merge_recordings(per_topic[first], others) if others else per_topic[first]
    # Lane order, not topic order, so the file reads the way --group was given.
    columns = np.concatenate(
        [group_columns(merged.joint_names, lane.group) for lane in session.lanes]
    )
    return Recording(
        merged.timestamps_ns, merged.values[:, columns],
        tuple(merged.joint_names[index] for index in columns), incomplete=merged.incomplete,
    )


def _report_recording(path: Path, recording, trimmed) -> None:
    seconds = (trimmed.timestamps_ns[-1] - trimmed.timestamps_ns[0]) / 1e9
    print(
        f"recorded {len(trimmed)} samples over {seconds:.2f} s to {path} "
        f"({len(recording) - len(trimmed)} still samples trimmed, "
        f"{recording.incomplete} incomplete messages)"
    )


# --- replay / goto -------------------------------------------------------------------


def _ramp_steps(start_pose, target, steps: int) -> np.ndarray:
    start_pose, target = np.asarray(start_pose, dtype=float), np.asarray(target, dtype=float)
    return np.array([start_pose + (target - start_pose) * (s / steps) for s in range(1, steps + 1)])


def _approach_steps(froms, tos, rate: float, speed: float) -> int:
    """How many command periods the move from *froms* to *tos* takes, across
    every lane, so lanes moving different distances arrive together."""
    if speed <= 0:
        raise ValueError("--approach-rad-s must be positive")
    travel = max(float(np.abs(np.asarray(b) - np.asarray(a)).max()) for a, b in zip(froms, tos))
    if travel < 1e-4:
        return 0
    return max(1, int(round(max(travel / speed, 1.0) * rate)))


def _segments(heres, lane_points, rate: float, speed: float, repeat: int):
    """Per lane: the approach to its first point, then its motion *repeat*
    times, returning to the start between runs. Every lane's segments have the
    same lengths, so they stream in step."""
    if repeat < 1:
        raise ValueError("--repeat must be at least 1")
    firsts = [points[0] for points in lane_points]
    lasts = [points[-1] for points in lane_points]
    approach = _approach_steps(heres, firsts, rate, speed)
    back = _approach_steps(lasts, firsts, rate, speed)
    per_lane = []
    for here, points in zip(heres, lane_points):
        segments = []
        if approach:
            segments.append(_ramp_steps(here, points[0], approach))
        for run in range(repeat):
            if run > 0 and back:
                segments.append(_ramp_steps(points[-1], points[0], back))
            segments.append(points)
        per_lane.append(segments)
    return per_lane


def _authorize(profile, group, here, segments) -> int:
    points = [point for segment in segments for point in segment]
    _cli._gate(profile, group, seed=here).authorize_trajectory(
        points, start_time_sec=0.0, period_sec=1.0 / profile.endpoint().command_rate_hz
    )
    return len(points)


def _gravity_text(given: str | None, recorded: dict[str, float] | None, groups) -> str | None:
    """--gravity as typed, else the scale the file was made with — one string
    the compensable groups agree on, since that is what --gravity expresses."""
    if given is not None or not recorded:
        return given
    values = sorted({f"{scale:g}" for scale in recorded.values()})
    if len(values) != 1:
        compensable = [group for group in groups if group.compensable]
        if len(compensable) != 1:
            raise ValueError(
                "the file was made with per-joint gravity scales; pass --gravity "
                "to say which apply"
            )
        return ",".join(f"{recorded[j]:g}" for j in compensable[0].joints if j in recorded)
    return values[0]


def _play(args, profile, groups, lane_points, gravity: str | None, what: str) -> int:
    """Move every lane through its points together, from wherever it is."""
    rate = profile.endpoint().command_rate_hz
    if args.hold_sec < 0:
        raise ValueError("--hold-sec must not be negative")
    scales = _lane_scales(gravity, groups)
    repeat = getattr(args, "repeat", 1)

    if not args.execute:
        firsts = [points[0] for points in lane_points]
        per_lane = _segments(firsts, lane_points, rate, args.approach_rad_s, repeat)
        count = max(
            _authorize(profile, group, first, segments)
            for group, first, segments in zip(groups, firsts, per_lane)
        )
        print(
            f"DRY RUN: {what}: {len(lane_points[0])} points at {rate:g} Hz x{repeat} "
            f"({count / rate:.1f} s from the first pose), gravity "
            f"{'off' if gravity is None else gravity}; pass --execute to play"
        )
        return 0

    backends, adapters = _open(profile, groups, execute=True)
    try:
        heres = [
            _cli._start_pose(profile, group, adapter.read_state())
            for group, adapter in zip(groups, adapters)
        ]
        per_lane = _segments(heres, lane_points, rate, args.approach_rad_s, repeat)
        count = max(
            _authorize(profile, group, here, segments)
            for group, here, segments in zip(groups, heres, per_lane)
        )
        lanes = _lanes(args, profile, groups, adapters, scales, heres)
        session = _session(args, profile, lanes)
        print(
            f"playing {what}: {count} samples ({count / rate:.1f} s), x{repeat}, "
            f"then holding {args.hold_sec:g} s; {KEY_QUIT} quits"
        )
        controllers = [TrajectoryController(segments, args.hold_sec) for segments in per_lane]
        return _run_session(session, controllers, lambda *_: None)
    finally:
        backends.close()


def _replay(args, profile) -> int:
    recording, recorded_groups, recorded_gravity = read_recording(args.input, profile.name)
    groups = _groups(profile, args.group or recorded_groups)
    rate = profile.endpoint().command_rate_hz
    points = resample(recording, rate, args.speed)
    lane_points = [points[:, group_columns(recording.joint_names, group)] for group in groups]
    gravity = _gravity_text(args.gravity, recorded_gravity, groups)
    return _play(args, profile, groups, lane_points, gravity, str(args.input))


def _goto(args, profile) -> int:
    path = args.poses or default_poses_path(profile.name)
    poses = load_poses(path, profile.name)
    if args.name not in poses:
        raise PoseStoreError(f"no pose {args.name!r} in {path}; known: {sorted(poses)}")
    pose = poses[args.name]
    groups = _groups(profile, args.group or pose.groups)
    lane_points = [np.asarray([pose_values(pose, group)]) for group in groups]
    gravity = _gravity_text(args.gravity, pose.gravity, groups)
    return _play(args, profile, groups, lane_points, gravity, f"pose {pose.name}")


# --- save / list ---------------------------------------------------------------------


def _save(args, profile) -> int:
    groups = _groups(profile, args.group)
    scales = _lane_scales(args.gravity, groups)
    path = args.poses or default_poses_path(profile.name)
    poses = load_poses(path, profile.name)
    backends, adapters = _open(profile, groups, execute=False)
    try:
        measured = [(group, adapter.read_state()) for group, adapter in zip(groups, adapters)]
    finally:
        backends.close()
    gravity = {}
    for group, scale in zip(groups, scales):
        if scale is not None:
            gravity.update(zip(group.joints, (float(s) for s in scale)))
    pose = taught_pose(args.name, measured, gravity)
    save_poses(path, with_pose(poses, pose), profile.name)
    for group, values in measured:
        print(f"saved {pose.name} {group.name}: {' '.join(f'{v:+.4f}' for v in values)}")
    print(f"-> {path}")
    return 0


def _list(args, profile) -> int:
    path = args.poses or default_poses_path(profile.name)
    poses = load_poses(path, profile.name)
    if not poses:
        print(f"no poses in {path}")
        return 0
    print(f"{path}:")
    for name, pose in sorted(poses.items()):
        values = " ".join(f"{pose.joints[j]:+.4f}" for j in pose.joints)
        gravity = "off" if not pose.gravity else _cli._scale_label(np.asarray(list(pose.gravity.values())), None)
        print(f"  {name:<20} {', '.join(pose.groups):<40} gravity {gravity:<9} {pose.saved_at}  [{values}]")
    return 0
