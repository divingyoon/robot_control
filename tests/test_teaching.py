"""Direct teaching: the state machine that lets a hand move the arm and keeps
where the hand left it.

Pure and clockless: every step is given the measured pose and the time, so a
session can be replayed here sample by sample. The stub arm is the impedance
loop in one line — it sits behind its command by a droop — because that droop
is exactly what the baseline exists to absorb.
"""

import numpy as np
import pytest

from robot_control.teaching import (
    Mode,
    TeachLimits,
    acknowledge,
    start,
    step,
    toggle_lock,
    trim_still,
    resample,
)
from robot_control.track import Recording, TrackError

N = 3
DROOP = np.array([0.01, 0.0, -0.005])


def _limits(**overrides):
    base = dict(
        push_rad=np.full(N, 0.02),
        still_rad_s=0.05,
        still_sec=0.2,
        settle_sec=0.1,
        max_follow_sec=5.0,
    )
    base.update(overrides)
    return TeachLimits(**base)


def _settle(state, limits, measured, t):
    """Run HOLD past its settle window so the baseline is taken."""
    state, _, event = step(state, limits, measured, t)
    return state, event


def test_start_holds_the_measured_pose_and_waits_to_baseline():
    measured = np.array([0.1, 0.2, 0.3])
    state = start(measured, time=0.0, settle_sec=0.5)

    assert state.mode is Mode.HOLD
    assert np.allclose(state.target, measured)
    assert np.allclose(state.baseline, 0.0)
    assert state.rebaseline_at == 0.5


def test_droop_within_the_settle_window_is_not_a_push():
    limits = _limits()
    q = np.array([0.1, 0.2, 0.3])
    state = start(q, time=0.0, settle_sec=limits.settle_sec)
    # The arm settles behind its command by a droop larger than push_rad
    # would tolerate, before the baseline has been taken. It settles the way a
    # droop does — a small motion, dying out — not the way a hand moves it.
    for k in range(1, 9):
        t = 0.01 * k
        state, command, event = step(state, limits, q - 3 * DROOP * min(k, 6) / 6, t)
        assert state.mode is Mode.HOLD and event is None, (t, event)
    assert np.allclose(command, q)


def test_motion_during_the_settle_window_is_a_push():
    """A latch mid-adjustment used to leave the arm stiff for the whole settle
    window, and then take the hand's displacement into the baseline."""
    limits = _limits()
    q = np.zeros(N)
    state = start(q, time=0.0, settle_sec=limits.settle_sec)
    # The hand keeps carrying joint 0 at 0.3 rad/s straight through the window.
    t, here, events = 0.0, q.copy(), []
    for _ in range(12):
        t += 0.01
        here = here + [0.003, 0.0, 0.0]
        state, command, event = step(state, limits, here, t)
        state = acknowledge(state, command)
        events.append(event)
    assert "pushed" in events and "baselined" not in events
    assert state.mode is Mode.FOLLOW
    # The baseline was never taken from a moving arm.
    assert np.allclose(state.baseline, 0.0)


def test_sustained_motion_inside_a_long_settle_window_is_a_push_too():
    limits = _limits(settle_sec=1.0, settle_motion_sec=0.2)
    state = start(np.zeros(N), time=0.0, settle_sec=limits.settle_sec)
    t, here, first_push = 0.0, np.zeros(N), None
    for _ in range(40):
        t += 0.01
        here = here + [0.003, 0.0, 0.0]
        state, _, event = step(state, limits, here, t)
        if event == "pushed":
            first_push = t
            break
    assert first_push is not None and 0.2 <= first_push <= 0.35


def test_baseline_is_taken_after_settling_and_absorbs_the_droop():
    limits = _limits()
    q = np.array([0.1, 0.2, 0.3])
    state = start(q, time=0.0, settle_sec=limits.settle_sec)
    state, event = _settle(state, limits, q - DROOP, 0.2)

    assert event == "baselined"
    assert np.allclose(state.baseline, DROOP)
    # Holding at the same droop afterwards is rest, not a push.
    state, _, event = step(state, limits, q - DROOP, 0.3)
    assert state.mode is Mode.HOLD and event is None


def test_a_deflection_past_push_rad_enters_follow():
    limits = _limits()
    q = np.array([0.1, 0.2, 0.3])
    state = start(q, time=0.0, settle_sec=limits.settle_sec)
    state, _ = _settle(state, limits, q - DROOP, 0.2)

    pushed = q - DROOP + np.array([0.0, 0.05, 0.0])
    state, command, event = step(state, limits, pushed, 0.3)

    assert event == "pushed"
    assert state.mode is Mode.FOLLOW
    # The desired command follows the hand, keeping the droop the hold needed.
    assert np.allclose(command, pushed + DROOP)


def test_follow_latches_when_the_arm_stops_and_then_rebaselines():
    limits = _limits()
    q = np.array([0.1, 0.2, 0.3])
    state = start(q, time=0.0, settle_sec=limits.settle_sec)
    state, _ = _settle(state, limits, q - DROOP, 0.2)
    state, _, _ = step(state, limits, q - DROOP + [0.0, 0.05, 0.0], 0.3)
    assert state.mode is Mode.FOLLOW

    # The hand carries the arm, then holds it still.
    here = q - DROOP + np.array([0.0, 0.5, 0.0])
    t = 0.4
    events = []
    for _ in range(24):
        state, command, event = step(state, limits, here, t)
        state = acknowledge(state, command)
        events.append(event)
        t += 0.05
        if event == "latched":
            break
    assert "latched" in events
    assert state.mode is Mode.HOLD
    assert np.allclose(state.target, here + DROOP)
    assert state.rebaseline_at is not None

    # Once settled the new droop becomes the new baseline.
    new_droop = np.array([0.02, 0.0, 0.0])
    state, _, event = step(state, limits, here + DROOP - new_droop, t + limits.settle_sec)
    assert event == "baselined"
    assert np.allclose(state.baseline, new_droop)


def test_follow_that_never_stops_is_cut_off_as_drift():
    limits = _limits(max_follow_sec=1.0)
    q = np.zeros(N)
    state = start(q, time=0.0, settle_sec=limits.settle_sec)
    state, _ = _settle(state, limits, q, 0.2)
    state, _, _ = step(state, limits, np.array([0.05, 0.0, 0.0]), 0.3)
    assert state.mode is Mode.FOLLOW

    # A slow, steady creep in one direction, never still.
    t, here = 0.3, np.array([0.05, 0.0, 0.0])
    event = None
    while t < 1.6 and event != "drift":
        t += 0.05
        here = here + [0.01, 0.0, 0.0]
        state, command, event = step(state, limits, here, t)
        state = acknowledge(state, command)

    assert event == "drift"
    assert state.mode is Mode.HOLD


def test_acknowledge_records_what_was_actually_commanded():
    limits = _limits()
    q = np.zeros(N)
    state = start(q, time=0.0, settle_sec=limits.settle_sec)
    state, _ = _settle(state, limits, q, 0.2)
    state, desired, _ = step(state, limits, np.array([0.1, 0.0, 0.0]), 0.3)
    clamped = desired * 0.5
    state = acknowledge(state, clamped)

    assert np.allclose(state.target, clamped)
    # Latching holds the clamped command, not the unreachable desired one.
    t = 0.35
    for _ in range(24):
        state, command, event = step(state, limits, np.array([0.1, 0.0, 0.0]), t)
        if state.mode is Mode.HOLD:
            break
        state = acknowledge(state, command * 0.5)
        t += 0.05
    assert state.mode is Mode.HOLD


def test_locked_hold_ignores_a_push_and_locking_mid_follow_latches():
    limits = _limits()
    q = np.zeros(N)
    state = start(q, time=0.0, settle_sec=limits.settle_sec)
    state, _ = _settle(state, limits, q, 0.2)

    locked = toggle_lock(state)
    assert locked.locked and not state.locked  # a new state, not a mutation
    locked, _, event = step(locked, limits, np.array([0.1, 0.0, 0.0]), 0.3)
    assert locked.mode is Mode.HOLD and event is None

    state, _, _ = step(state, limits, np.array([0.1, 0.0, 0.0]), 0.3)
    assert state.mode is Mode.FOLLOW
    state, _, event = step(toggle_lock(state), limits, np.array([0.2, 0.0, 0.0]), 0.35)
    assert event == "locked" and state.mode is Mode.HOLD


def test_step_refuses_a_clock_that_runs_backwards():
    limits = _limits()
    state = start(np.zeros(N), time=1.0, settle_sec=0.1)
    with pytest.raises(ValueError, match="backwards"):
        step(state, limits, np.zeros(N), 0.5)


def test_limits_refuse_nonsense():
    with pytest.raises(ValueError):
        _limits(push_rad=np.array([0.02, -0.01, 0.02]))
    with pytest.raises(ValueError):
        _limits(still_sec=0.0)


# --- recordings -------------------------------------------------------------


def _recording(values, period_ns=10_000_000):
    values = np.asarray(values, dtype=float)
    stamps = np.arange(len(values), dtype=np.int64) * period_ns
    return Recording(stamps, values, tuple(f"j{i}" for i in range(values.shape[1])))


def test_trim_still_drops_the_wait_before_and_after_the_motion():
    still = [[0.0, 0.0]] * 5
    motion = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.1], [0.3, 0.1]]
    tail = [[0.3, 0.1]] * 4
    recording = _recording(still + motion + tail)

    trimmed = trim_still(recording, threshold_rad=0.01)

    assert len(trimmed) == len(motion)
    assert np.allclose(trimmed.values[0], motion[0])
    assert np.allclose(trimmed.values[-1], motion[-1])
    assert trimmed.timestamps_ns[0] == 5 * 10_000_000


def test_trim_still_keeps_a_recording_that_never_moved_as_two_samples():
    recording = _recording([[0.0, 0.0]] * 5)
    trimmed = trim_still(recording, threshold_rad=0.01)
    assert len(trimmed) == 2


def test_resample_puts_the_motion_on_the_command_grid():
    recording = _recording([[0.0], [0.1], [0.2]], period_ns=100_000_000)  # 10 Hz
    points = resample(recording, rate_hz=100.0, speed=1.0)

    assert points.shape == (21, 1)
    assert np.allclose(points[:, 0], np.linspace(0.0, 0.2, 21))


def test_resample_speed_compresses_time_not_position():
    recording = _recording([[0.0], [0.1], [0.2]], period_ns=100_000_000)
    points = resample(recording, rate_hz=100.0, speed=2.0)

    assert points.shape == (11, 1)
    assert np.isclose(points[-1, 0], 0.2)


def test_resample_tolerates_reordered_and_duplicate_stamps():
    stamps = np.array([0, 200_000_000, 100_000_000, 100_000_000], dtype=np.int64)
    values = np.array([[0.0], [0.2], [0.1], [0.1]])
    recording = Recording(stamps, values, ("j0",))
    points = resample(recording, rate_hz=10.0, speed=1.0)
    assert np.allclose(points[:, 0], [0.0, 0.1, 0.2])


def test_resample_refuses_a_recording_with_no_duration():
    recording = _recording([[0.0]])
    with pytest.raises(TrackError):
        resample(recording, rate_hz=100.0, speed=1.0)
    with pytest.raises(ValueError):
        resample(_recording([[0.0], [0.1]]), rate_hz=100.0, speed=0.0)


def test_a_push_during_the_settle_window_cannot_drag_the_hold_along():
    """The gate bounds how far a command runs ahead of the arm; acknowledged in
    HOLD, that bound would walk the target after the hand with no event and
    none of FOLLOW's guards. Reproduces the review finding against the real
    gate, not a stub of it."""
    from robot_control.safety import CommandGate

    limits = _limits(settle_sec=1.0)
    gate = CommandGate(
        execute=True,
        lower=np.full(N, -3.0),
        upper=np.full(N, 3.0),
        velocity=np.full(N, 2.0),
        command_period_sec=0.01,
        max_lead=np.full(N, 0.2),
    )
    state = start(np.zeros(N), time=0.0, settle_sec=limits.settle_sec)
    state = toggle_lock(state)  # locked: the settle window may not follow either
    t, measured = 0.0, np.zeros(N)
    # A hand carries joint 0 away at 2 rad/s, entirely inside the settle window.
    for _ in range(50):
        t += 0.01
        measured = measured + [0.02, 0.0, 0.0]
        state, desired, event = step(state, limits, measured, t)
        command, _ = gate.follow(desired, measured, 0.01)
        state = acknowledge(state, command)
        assert state.mode is Mode.HOLD and event is None
        assert np.allclose(state.target, 0.0), "the hold was dragged"
        # The published command may run ahead of the arm only by the lead bound.
        assert abs(command[0] - measured[0]) <= 0.2 + 1e-9


def test_resample_grid_is_counted_not_stepped():
    recording = _recording([[0.0], [0.3]], period_ns=300_000_000)
    for speed in (1.0, 0.7, 1.3, 3.0):
        points = resample(recording, rate_hz=100.0, speed=speed)
        assert len(points) == int(np.floor(0.3 / speed * 100 + 1e-9)) + 1
