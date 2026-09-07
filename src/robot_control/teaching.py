"""Direct teaching: let a hand move the arm, and keep the pose it leaves.

The arms hold position through the DM motors' impedance loop, whose gains are
fixed at bringup. Compliance is therefore made in software: while a hand is
pushing, the trajectory controller's target is re-commanded to where the arm
*is*, so the only resistance left is damping and one period's worth of
stiffness. When the hand lets go the target is frozen and the loop holds.

Two things make that work on a real arm rather than a perfect-tracking stub.

The **baseline**. An impedance-held joint sits behind its command by the droop
that produces its holding torque, and the gravity feedforward the session
publishes only removes most of it. That standing offset is not a push, so it is
measured after every latch and subtracted before deflection is judged — and it
is carried along while following, so the command stays the droop ahead of the
hand and the arm does not sink the moment following starts.

The **settle window**. A latch freezes the target while the droop settles,
and only then is the baseline measured. A hand that is still on the arm shows
up in that window as *motion*, so motion there means follow again — judging
it by deflection would be circular, since the droop being measured is itself
a deflection. Without this a slow, careful adjustment that latched mid-move
felt like an arm that had seized, and a baseline taken with the hand still
pushing carried the hand's displacement into every command after it.

The **drift guard**. Following is the one state in which nothing resists a
gravity model error, so an error the baseline did not capture shows up as an
arm that keeps moving slowly with no hand on it. A push that has gone on for
``max_follow_sec`` without the arm ever coming to rest is treated as that, and
latched. Together with the lock, which refuses to leave HOLD, this is the
software's whole answer to the arm running away; the operator's is the E-stop.

Everything here is pure: a state in, a state out, the clock supplied by the
caller. The servo loop that owns the ROS traffic lives in ``teach_cli``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from .track import Recording, TrackError


class Mode(Enum):
    HOLD = "hold"
    FOLLOW = "follow"


#: Weight of the newest sample in the velocity estimate. One period's
#: difference of an encoder reading is mostly noise; a few periods' worth is a
#: speed.
VELOCITY_SMOOTHING = 0.3
#: While settling, motion faster than this multiple of the rest threshold is a
#: hand, not a droop finding its level.
SETTLE_MOTION_FACTOR = 2.0


@dataclass(frozen=True)
class TeachLimits:
    """What counts as a push, what counts as rest, and how long a push may last."""

    #: Per-joint deflection beyond the baseline that counts as a hand pushing.
    push_rad: np.ndarray
    #: Below this on every joint the arm counts as at rest. One value, or one
    #: per joint: a gripper's whole stroke is a few centimetres, so what is
    #: "still" for it is a fraction of what is still for a shoulder.
    still_rad_s: np.ndarray
    #: How long the arm must rest, while following, before the pose is latched.
    still_sec: float
    #: After a latch, how long to let the droop settle before re-measuring it.
    settle_sec: float
    #: A push that lasts this long without a rest is drift, and is latched.
    max_follow_sec: float
    #: While settling, motion must last this long to count as a hand. A droop
    #: finding its level is over in a fraction of this; a hand keeps going.
    settle_motion_sec: float = 0.3

    def __post_init__(self) -> None:
        push = np.asarray(self.push_rad, dtype=float)
        if push.ndim != 1 or not np.isfinite(push).all() or np.any(push <= 0):
            raise ValueError("push_rad needs one positive radian threshold per joint")
        object.__setattr__(self, "push_rad", push)
        still = np.broadcast_to(np.asarray(self.still_rad_s, dtype=float), push.shape).copy()
        if not np.isfinite(still).all() or np.any(still <= 0):
            raise ValueError("still_rad_s needs positive speed thresholds")
        object.__setattr__(self, "still_rad_s", still)
        for name in ("still_sec", "settle_sec", "max_follow_sec", "settle_motion_sec"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass(frozen=True)
class TeachState:
    mode: Mode
    #: What HOLD commands, and the last command acknowledged while following.
    target: np.ndarray
    #: target - measured at rest: the droop the impedance loop needs to hold.
    baseline: np.ndarray
    #: The previous sample, for the velocity estimate.
    measured: np.ndarray
    time: float
    locked: bool = False
    #: Smoothed joint velocity, what stillness and settle-window motion are
    #: judged by.
    velocity: np.ndarray | None = None
    #: When HOLD should next measure its baseline; None once it has.
    rebaseline_at: float | None = None
    #: While following, when the arm was last seen coming to rest.
    still_since: float | None = None
    follow_since: float | None = None
    #: While settling, when the arm was first seen moving.
    moving_since: float | None = None


def start(measured, time: float, settle_sec: float) -> TeachState:
    """Hold the measured pose, and measure the droop once it has settled."""
    measured = np.asarray(measured, dtype=float).copy()
    return TeachState(
        mode=Mode.HOLD,
        target=measured,
        baseline=np.zeros_like(measured),
        measured=measured,
        time=float(time),
        velocity=np.zeros_like(measured),
        rebaseline_at=float(time) + float(settle_sec),
    )


def toggle_lock(state: TeachState) -> TeachState:
    return replace(state, locked=not state.locked)


def acknowledge(state: TeachState, commanded) -> TeachState:
    """Record what actually went to the controller, after the gate had its say.

    Latching holds the last *acknowledged* command: what the arm was asked to
    be at, not what the hand wanted, which the gate may have clamped.

    Only while following. In HOLD the target is frozen by definition: the gate
    may still bound how far the published command runs ahead of an arm a hand
    is pushing against, and that bound is the hold's resistance limit, not a
    new target. Letting it feed back here would drag the hold along with the
    hand — silently, in HOLD, with none of FOLLOW's guards — for as long as
    the settle window keeps push detection off.
    """
    if state.mode is not Mode.FOLLOW:
        return state
    return replace(state, target=np.asarray(commanded, dtype=float).copy())


def step(
    state: TeachState, limits: TeachLimits, measured, time: float
) -> tuple[TeachState, np.ndarray, str | None]:
    """Advance one sample. Returns the new state, the desired command, and an
    event name when the mode changed or the baseline was taken."""
    measured = np.asarray(measured, dtype=float)
    time = float(time)
    elapsed = time - state.time
    if elapsed < 0:
        raise ValueError("the clock ran backwards between teaching steps")
    raw = (measured - state.measured) / elapsed if elapsed > 0 else np.zeros_like(measured)
    previous = np.zeros_like(measured) if state.velocity is None else state.velocity
    velocity = VELOCITY_SMOOTHING * raw + (1.0 - VELOCITY_SMOOTHING) * previous
    state = replace(state, velocity=velocity)

    if state.mode is Mode.HOLD:
        state, command, event = _hold(state, limits, measured, velocity, time)
    else:
        state, command, event = _follow(state, limits, measured, velocity, time)
    return replace(state, measured=measured.copy(), time=time), command, event


def _hold(state, limits, measured, velocity, time):
    event = None
    if state.rebaseline_at is not None:
        # Settling. The droop is still finding its level, so deflection says
        # nothing yet — but motion does: a hand still on the arm keeps it
        # moving, where a droop settles in a fraction of settle_motion_sec.
        moving = bool(np.any(np.abs(velocity) > SETTLE_MOTION_FACTOR * limits.still_rad_s))
        moving_since = None if not moving else (
            time if state.moving_since is None else state.moving_since
        )
        state = replace(state, moving_since=moving_since)
        sustained = moving_since is not None and time - moving_since >= limits.settle_motion_sec
        at_end = time >= state.rebaseline_at
        if not state.locked and (sustained or (at_end and moving)):
            # Not a baseline worth taking: the hand is in it.
            state = replace(
                state, mode=Mode.FOLLOW, follow_since=time, still_since=None, moving_since=None
            )
            return state, measured + state.baseline, "pushed"
        if not at_end:
            return state, state.target, None
        state = replace(
            state, baseline=state.target - measured, rebaseline_at=None, moving_since=None
        )
        event = "baselined"
    deflection = (state.target - measured) - state.baseline
    if not state.locked and np.any(np.abs(deflection) > limits.push_rad):
        state = replace(state, mode=Mode.FOLLOW, follow_since=time, still_since=None)
        return state, measured + state.baseline, "pushed"
    return state, state.target, event


def _follow(state, limits, measured, velocity, time):
    still = bool(np.all(np.abs(velocity) <= limits.still_rad_s))
    still_since = None if not still else (state.still_since if state.still_since is not None else time)

    event = None
    if state.locked:
        event = "locked"
    elif time - state.follow_since >= limits.max_follow_sec:
        event = "drift"
    elif still_since is not None and time - still_since >= limits.still_sec:
        event = "latched"
    if event is not None:
        latched = replace(
            state,
            mode=Mode.HOLD,
            rebaseline_at=time + limits.settle_sec,
            still_since=None,
            follow_since=None,
        )
        return latched, latched.target, event
    return replace(state, still_since=still_since), measured + state.baseline, None


# --- recordings ---------------------------------------------------------------


def trim_still(recording: Recording, threshold_rad: float) -> Recording:
    """Drop the wait before the hand first moved the arm, and after it last did.

    A teaching session starts with the arm held and the operator walking over,
    and ends with it held again; replaying those is a replay that stands still.
    A recording that never moved keeps its first and last samples, so it is
    still a (zero-length) motion rather than an error here.
    """
    if threshold_rad <= 0:
        raise ValueError("threshold_rad must be positive")
    values = recording.values
    moved_from_start = np.any(np.abs(values - values[0]) > threshold_rad, axis=1)
    moved_from_end = np.any(np.abs(values - values[-1]) > threshold_rad, axis=1)
    if not moved_from_start.any():
        keep = np.array([0, len(values) - 1]) if len(values) > 1 else np.array([0])
    else:
        # The last still sample before motion begins, and the first after it ends.
        first = max(int(np.argmax(moved_from_start)) - 1, 0)
        last = min(len(values) - int(np.argmax(moved_from_end[::-1])), len(values) - 1)
        keep = np.arange(first, max(last, first + 1) + 1)
    return Recording(
        recording.timestamps_ns[keep],
        values[keep],
        recording.joint_names,
        incomplete=recording.incomplete,
    )


def resample(recording: Recording, rate_hz: float, speed: float) -> np.ndarray:
    """Put a recording on the command grid, its clock divided by *speed*.

    Stamps are sorted and de-duplicated first: a recording is what arrived,
    and what arrived is sometimes out of order. Returns one row per command
    period, from the first sample to the last.
    """
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be positive")
    stamps, order = np.unique(recording.timestamps_ns, return_index=True)
    values = recording.values[order]
    if len(stamps) < 2 or stamps[-1] == stamps[0]:
        raise TrackError("a recording needs at least two distinct stamps to replay")
    seconds = (stamps - stamps[0]) / 1e9 / speed
    # Counted rather than stepped: np.arange with a float step includes or
    # drops its last point depending on rounding.
    count = int(np.floor(seconds[-1] * rate_hz + 1e-9)) + 1
    grid = np.arange(count) / rate_hz
    return np.column_stack(
        [np.interp(grid, seconds, values[:, index]) for index in range(values.shape[1])]
    )
