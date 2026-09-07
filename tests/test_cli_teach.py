"""`robotctl teach`: poses and trajectories taught by hand.

The stub arm is the impedance loop reduced to what matters here: it sits behind
its command by a droop, and a hand on it shows up as a position the loop did
not command. The push is scripted by read count rather than by wall time so
the same session plays out the same way on a loaded machine. A session opens
one backend and one adapter per group, so the stubs are handed out by group.
"""

import sys

import numpy as np
import pytest

from robot_control.cli import main
from robot_control.track import Recording

ARM = "openarm_left_arm"
GRIPPER = "openarm_left_gripper"
LEFT_ARM = ["--group", ARM]
BOTH = ["--group", ARM, "--group", GRIPPER]
ARM_JOINTS = tuple(f"l_aj_{i}" for i in range(1, 8))
GRIPPER_JOINTS = ("l_hj_gripper_1",)
N = 7
DROOP = np.array([0.01, 0.005, 0.0, 0.01, 0.0, 0.0, 0.0])
FAST = ["--settle-sec", "0.1", "--still-sec", "0.1", "--seconds", "1.5"]


class FlatChain:
    """A gravity model that always asks for the same small torque."""

    def gravity_torque(self, q):
        return np.full(len(q), 0.5)


def _push_joint_two(reads: int):
    """No hand for 15 reads, then a hand that takes the arm where it sits and
    carries joint 2 to +0.05 by read 35, holding it there. A hand imposes a
    position, not an offset from the command: the command follows the hand."""
    if reads < 15:
        return None
    hand = -DROOP.copy()
    hand[1] += 0.05 * min(reads - 15, 20) / 20
    return hand


def _squeeze(reads: int):
    """A hand closing the gripper from 0.04 to 0.01 over reads 15..35."""
    if reads < 15:
        return None
    return np.array([0.04 - 0.03 * min(reads - 15, 20) / 20])


NO_HAND = lambda reads: None  # noqa: E731


class Stub:
    def __init__(self, joints, external=NO_HAND, initial=None, droop=None):
        self.joints = tuple(joints)
        self.initial = np.zeros(len(joints)) if initial is None else np.asarray(initial, dtype=float)
        self.droop = np.zeros(len(joints)) if droop is None else np.asarray(droop, dtype=float)
        self.external = external
        self.command = None
        self.reads = 0
        self.streamed = []
        self.published = []
        self.recording = None

    def pump(self, timeout_sec=0.0):
        return None

    def read_state(self, timeout_sec=None):
        self.reads += 1
        hand = self.external(self.reads)
        base = self.initial if self.command is None else self.command - self.droop
        state = base.copy() if hand is None else np.asarray(hand, dtype=float).copy()
        if self.recording is not None:
            self.recording.append((self.reads * 10_000_000, state.copy()))
        return state

    def stream_positions(self, positions):
        self.command = np.asarray(positions, dtype=float).copy()
        self.streamed.append(self.command)

    def send_effort(self, effort):
        self.published.append(np.asarray(effort, dtype=float).copy())

    gain_p = None  # {joint: p} when the controller declares gains
    gain_writes = ()

    def read_gain_p(self, timeout_sec=None):
        return dict(self.gain_p or {})

    def write_gain_p(self, values):
        self.gain_p = {**self.gain_p, **values}
        self.gain_writes = (*self.gain_writes, dict(values))

    def start_recording(self):
        self.recording = []


PROFILE = None


def _profile():
    global PROFILE
    if PROFILE is None:
        from robot_control.profile import load_builtin_profile
        from robot_control.teach_cli import _with_composites

        PROFILE = _with_composites(load_builtin_profile("openarm_tesollo"))
    return PROFILE


class Robot:
    """Hands out one stub per group and one backend per topic, and stitches
    the stubs' recordings together the way the shared backend would."""

    def __init__(self):
        self.stubs = {}
        self.topics = []
        self.closed = 0
        self.arm = self.stub(ARM, external=_push_joint_two, droop=DROOP)
        self.gripper = self.stub(GRIPPER, initial=[0.04])

    def stub(self, name, **kw):
        stub = Stub(_profile().groups[name].joints, **kw)
        stub.start_recording = self.start_recording
        stub.stop_recording_groups = self.stop_recording
        self.stubs[name] = stub
        return stub

    def make_backend(self, node_name, topic):
        self.topics.append(topic)
        return self

    def adapter(self, profile, name, execute=False, backend=None):
        return self.stubs.get(name) or self.stub(name)

    def start_recording(self):
        for stub in self.stubs.values():
            stub.recording = []

    def stop_recording(self, names):
        stubs = [self.stubs[name] for name in names]
        # Rows are joined by read index; the stubs are read once per cycle.
        rows = [np.concatenate([stub.recording[i][1] for stub in stubs])
                for i in range(min(len(stub.recording) for stub in stubs))]
        stamps = np.array([s for s, _ in stubs[0].recording[: len(rows)]], dtype=np.int64)
        for stub in stubs:
            stub.recording = None
        return Recording(stamps, np.vstack(rows), tuple(j for stub in stubs for j in stub.joints))

    def close(self):
        self.closed += 1


@pytest.fixture
def no_ros(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", None)


@pytest.fixture
def robot(monkeypatch):
    from robot_control import ros_adapter

    stub = Robot()
    monkeypatch.setattr(ros_adapter, "make_backend", stub.make_backend)
    monkeypatch.setattr(ros_adapter, "RosAdapter", stub.adapter)
    monkeypatch.setattr("robot_control.cli._gravity_chain", lambda *a: FlatChain())
    return stub


@pytest.fixture
def offline(monkeypatch):
    """Prove a stage never opens the robot."""
    from robot_control import ros_adapter

    def refuse(*a, **k):
        raise AssertionError("the robot was opened")

    monkeypatch.setattr(ros_adapter, "make_backend", refuse)
    monkeypatch.setattr(ros_adapter, "RosAdapter", refuse)


def _write_recording(path, values, joints=ARM_JOINTS, groups=(ARM,), gravity=None,
                     period_ns=10_000_000):
    values = np.asarray(values, dtype=float)
    gravity = gravity or {}
    np.savez(
        path,
        schema=2,
        profile="openarm_tesollo",
        groups=np.array(list(groups)),
        joint_names=np.array(list(joints)),
        timestamps_ns=np.arange(len(values), dtype=np.int64) * period_ns,
        values=values,
        gravity_joints=np.array(list(gravity)),
        gravity_scales=np.array(list(gravity.values()), dtype=float),
    )
    return path


def _slow_motion(steps=20, width=N):
    """Joint 1 sweeping 0.1 rad over *steps* samples at 100 Hz: 0.5 rad/s."""
    values = np.zeros((steps + 1, width))
    values[:, 0] = np.linspace(0.0, 0.1, steps + 1)
    return values


# --- hold ---------------------------------------------------------------------


def test_hold_dry_run_streams_nothing(robot, capsys):
    assert main(["teach", "hold", *LEFT_ARM, "--gravity", "1.0"]) == 0
    assert robot.arm.streamed == [] and robot.arm.published == []
    assert "DRY RUN" in capsys.readouterr().out
    assert robot.closed == 1


def test_hold_follows_a_push_and_keeps_where_the_hand_left_it(robot, capsys):
    assert main(["teach", "hold", *LEFT_ARM, "--execute", "--gravity", "1.0", *FAST]) == 0

    out = capsys.readouterr().out
    assert "baselined" in out and "pushed" in out and "latched" in out
    # The hold target is the hand's pose plus the droop the arm needs to be there.
    final = robot.arm.streamed[-1]
    assert np.isclose(final[1], 0.05, atol=2e-3), final
    assert np.allclose(np.delete(final, 1), 0.0, atol=2e-3), final
    # The torque was released at the end, and the operator told so.
    assert np.allclose(robot.arm.published[-1], 0.0)
    assert "sag" in out


def test_hold_teaches_an_arm_and_its_gripper_together(robot, capsys):
    robot.gripper.external = _squeeze
    assert main(["teach", "hold", *BOTH, "--execute", "--gravity", "1.0", *FAST]) == 0

    out = capsys.readouterr().out
    assert f"{ARM}: pushed" in out and f"{GRIPPER}: pushed" in out
    assert f"{GRIPPER}: latched" in out
    assert np.isclose(robot.gripper.streamed[-1][0], 0.01, atol=1e-3)
    assert np.isclose(robot.arm.streamed[-1][1], 0.05, atol=2e-3)
    # Gravity went to the arm only; the gripper has no effort controller.
    assert robot.arm.published and robot.gripper.published == []


def test_gripper_push_threshold_is_a_fraction_of_its_stroke(robot, capsys):
    assert main(["teach", "hold", *BOTH]) == 0
    out = capsys.readouterr().out
    assert f"{GRIPPER}: gravity scale off; push threshold 0.0022" in out
    assert f"{ARM}: gravity scale off; push threshold 0.02 " in out


def test_hold_without_a_push_stays_put(robot, capsys):
    robot.arm.external = NO_HAND
    assert main(["teach", "hold", *LEFT_ARM, "--execute", "--seconds", "0.6",
                 "--settle-sec", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "pushed" not in out
    assert np.allclose(robot.arm.streamed[-1], 0.0)
    assert robot.arm.published == []  # no gravity asked for, none published


def test_hold_can_keep_the_gravity_torque_on(robot, capsys):
    assert main(["teach", "hold", *LEFT_ARM, "--execute", "--gravity", "1.0",
                 "--keep-gravity", "--seconds", "0.2"]) == 0
    assert np.allclose(robot.arm.published[-1], 0.5)
    assert f"pose gravity --group {ARM} --scale 0" in capsys.readouterr().out


def test_hold_refuses_an_out_of_range_gravity_scale(no_ros, capsys):
    assert main(["teach", "hold", *LEFT_ARM, "--gravity", "9"]) == 2
    assert "outside" in capsys.readouterr().out


def test_hold_refuses_gravity_on_a_gripper_alone(no_ros, capsys):
    assert main(["teach", "hold", "--group", GRIPPER, "--gravity", "1.0"]) == 2
    assert "effort_controller" in capsys.readouterr().out


def test_hold_refuses_the_wrong_number_of_push_thresholds(no_ros, capsys):
    assert main(["teach", "hold", *LEFT_ARM, "--push-rad", "0.1,0.1"]) == 2
    assert "--push-rad" in capsys.readouterr().out


def test_hold_refuses_two_groups_that_stream_to_one_controller(no_ros, capsys):
    assert main(["teach", "hold", "--group", "tesollo_curl", "--group", "tesollo_pip"]) == 2
    assert "same controller" in capsys.readouterr().out


def test_the_tesollo_hand_is_one_lane_with_thresholds_from_its_ranges(robot, capsys):
    assert main(["teach", "hold", "--group", "openarm_right_arm", "--group", "tesollo_hand"]) == 0
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if "tesollo_hand:" in l)
    assert len(line.split("push threshold ")[1].split()) == 20
    # Two joint-state topics, so two backends were opened.
    assert robot.topics == ["/joint_states", "/dg5f_right/joint_states"]


def test_record_merges_the_hand_topic_onto_the_arms_clock(robot, tmp_path, capsys):
    hand = robot.stub("tesollo_hand")
    limits = {joint.canonical: joint for joint in _profile().joints}
    middle = np.array([(limits[j].lower + limits[j].upper) / 2 for j in hand.joints])
    hand.initial = middle
    robot.stubs["openarm_right_arm"] = robot.stub("openarm_right_arm", external=NO_HAND)
    output = tmp_path / "hand.npz"
    assert main(["teach", "record", "--group", "openarm_right_arm", "--group", "tesollo_hand",
                 "--execute", "--output", str(output), "--seconds", "0.5",
                 "--settle-sec", "0.1"]) == 0
    raw = np.load(output, allow_pickle=False)
    names = list(raw["joint_names"])
    assert names[:7] == [f"r_aj_{i}" for i in range(1, 8)] and len(names) == 27
    assert names[7] == "r_hj_thumb_1"
    assert np.allclose(raw["values"][:, 7:], middle)


# --- record / replay -------------------------------------------------------------


def test_record_writes_the_trimmed_motion_of_every_group(robot, tmp_path, capsys):
    robot.gripper.external = _squeeze
    output = tmp_path / "demo.npz"
    assert main(["teach", "record", *BOTH, "--execute", "--gravity", "1.0",
                 "--output", str(output), *FAST]) == 0

    raw = np.load(output, allow_pickle=False)
    assert list(raw["groups"]) == [ARM, GRIPPER]
    assert list(raw["joint_names"]) == list(ARM_JOINTS + GRIPPER_JOINTS)
    values = raw["values"]
    assert len(values) < robot.arm.reads  # the still time is gone
    assert values[-1, 1] > 0.04 and values[-1, 7] < 0.015
    assert dict(zip(raw["gravity_joints"], raw["gravity_scales"])) == {j: 1.0 for j in ARM_JOINTS}
    assert "trimmed" in capsys.readouterr().out


def test_record_refuses_an_output_without_the_npz_suffix(no_ros, capsys):
    assert main(["teach", "record", *LEFT_ARM, "--output", "demo"]) == 2
    assert ".npz" in capsys.readouterr().out


def test_replay_dry_run_is_offline_and_authorizes_the_motion(offline, tmp_path, capsys):
    path = _write_recording(tmp_path / "demo.npz", _slow_motion())
    assert main(["teach", "replay", "--input", str(path), "--repeat", "2"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "x2" in out


def test_replay_streams_every_group_in_step_and_repeats(robot, tmp_path, capsys):
    values = np.hstack([_slow_motion(), np.linspace(0.04, 0.01, 21)[:, None]])
    path = _write_recording(tmp_path / "demo.npz", values, joints=ARM_JOINTS + GRIPPER_JOINTS,
                            groups=(ARM, GRIPPER))
    robot.arm.external = NO_HAND
    assert main(["teach", "replay", "--execute", "--input", str(path),
                 "--repeat", "2", "--seconds", "10"]) == 0

    # Approach (the arm is at zero, the gripper at 0.04: nothing to do), the
    # motion, the return at approach speed — sized by the arm's 0.1 rad — and
    # the motion again, the same count on both lanes.
    motion, back = 21, 100
    assert len(robot.arm.streamed) == motion + back + motion
    assert len(robot.gripper.streamed) == motion + back + motion
    assert np.isclose(robot.arm.streamed[-1][0], 0.1)
    assert np.isclose(robot.gripper.streamed[-1][0], 0.01)
    assert "played" in capsys.readouterr().out


def test_replay_can_be_narrowed_to_one_of_the_recorded_groups(robot, tmp_path, capsys):
    values = np.hstack([_slow_motion(), np.full((21, 1), 0.04)])
    path = _write_recording(tmp_path / "demo.npz", values, joints=ARM_JOINTS + GRIPPER_JOINTS,
                            groups=(ARM, GRIPPER))
    robot.arm.external = NO_HAND
    assert main(["teach", "replay", *LEFT_ARM, "--execute", "--input", str(path),
                 "--seconds", "10"]) == 0
    assert robot.arm.streamed and robot.gripper.streamed == []


def test_replay_refuses_a_group_the_recording_does_not_cover(offline, tmp_path, capsys):
    path = _write_recording(tmp_path / "left.npz", _slow_motion())
    assert main(["teach", "replay", "--group", "openarm_right_arm", "--input", str(path)]) == 2
    assert "r_aj_1" in capsys.readouterr().out


def test_replay_refuses_a_motion_faster_than_the_profile_allows(offline, tmp_path, capsys):
    values = np.zeros((3, N))
    values[:, 0] = [0.0, 1.0, 0.0]  # 1 rad in 10 ms
    path = _write_recording(tmp_path / "fast.npz", values)
    assert main(["teach", "replay", "--input", str(path)]) == 3
    assert "velocity" in capsys.readouterr().out


def test_replay_uses_the_gravity_scale_it_was_recorded_with(robot, tmp_path, capsys):
    path = _write_recording(tmp_path / "demo.npz", _slow_motion(),
                            gravity={j: 0.9 for j in ARM_JOINTS})
    robot.arm.external = NO_HAND
    assert main(["teach", "replay", "--execute", "--input", str(path), "--seconds", "10"]) == 0
    assert np.allclose(robot.arm.published[0], 0.45)


# --- save / goto / list --------------------------------------------------------------


def test_save_list_goto_round_trip_with_the_gripper(robot, tmp_path, capsys):
    poses = tmp_path / "poses.yaml"
    robot.arm.external = NO_HAND
    robot.arm.initial = np.array([0.1, -0.2, 0.0, 0.8, 0.0, 0.0, 0.5])
    robot.gripper.initial = np.array([0.02])

    assert main(["teach", "save", *BOTH, "--name", "pour_start", "--poses",
                 str(poses), "--gravity", "1.0"]) == 0
    assert robot.arm.streamed == []

    assert main(["teach", "list", "--poses", str(poses)]) == 0
    out = capsys.readouterr().out
    assert "pour_start" in out and GRIPPER in out

    assert main(["teach", "goto", "--name", "pour_start", "--poses", str(poses)]) == 0
    assert "DRY RUN" in capsys.readouterr().out and robot.arm.streamed == []

    robot.arm.initial = np.zeros(N)
    robot.gripper.initial = np.array([0.04])
    assert main(["teach", "goto", "--execute", "--name", "pour_start",
                 "--poses", str(poses), "--seconds", "30"]) == 0
    assert np.allclose(robot.arm.streamed[-1], [0.1, -0.2, 0.0, 0.8, 0.0, 0.0, 0.5])
    assert np.isclose(robot.gripper.streamed[-1][0], 0.02)
    assert len(robot.arm.streamed) == len(robot.gripper.streamed)
    # The pose was saved with gravity on, so goto put it back on — for the arm.
    assert np.allclose(robot.arm.published[0], 0.5) and robot.gripper.published == []


def test_goto_refuses_an_unknown_pose(offline, tmp_path, capsys):
    assert main(["teach", "goto", "--name", "nope", "--poses", str(tmp_path / "p.yaml")]) == 2
    assert "nope" in capsys.readouterr().out


def test_list_of_an_empty_store_is_offline(offline, tmp_path, capsys):
    assert main(["teach", "list", "--poses", str(tmp_path / "none.yaml")]) == 0
    assert "no poses" in capsys.readouterr().out


def test_hold_saves_a_pose_on_the_s_key(robot, tmp_path, monkeypatch, capsys):
    from robot_control import teach_cli

    class OneKey:
        def __init__(self, *a, **k):
            self.sent = False
            self.enabled = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def poll(self):
            if self.sent:
                return None
            self.sent = True
            return "s"

    monkeypatch.setattr(teach_cli, "KeyReader", OneKey)
    poses = tmp_path / "poses.yaml"
    robot.arm.external = NO_HAND
    assert main(["teach", "hold", *BOTH, "--execute", "--seconds", "0.2",
                 "--poses", str(poses), "--name-prefix", "demo"]) == 0
    assert "saved demo_1" in capsys.readouterr().out
    assert main(["teach", "list", "--poses", str(poses)]) == 0
    assert "demo_1" in capsys.readouterr().out


def test_a_lane_that_drops_a_sample_is_skipped_not_fatal(robot, capsys):
    from robot_control.ros_adapter import AdapterUnavailable

    robot.arm.external = NO_HAND
    flaky = robot.gripper
    real_read = flaky.read_state

    def read_state(timeout_sec=None):
        if 10 <= flaky.reads < 20:
            flaky.reads += 1
            raise AdapterUnavailable("no /joint_states")
        return real_read(timeout_sec)

    flaky.read_state = read_state
    assert main(["teach", "hold", *BOTH, "--execute", "--seconds", "0.5",
                 "--settle-sec", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "joint state back" in out and "unavailable" not in out
    assert len(robot.arm.streamed) > len(flaky.streamed)


def test_a_lane_silent_for_too_long_ends_the_session(robot, monkeypatch, capsys):
    from robot_control import teach_cli
    from robot_control.ros_adapter import AdapterUnavailable

    monkeypatch.setattr(teach_cli, "STALE_LIMIT_SEC", 0.1)
    robot.arm.external = NO_HAND

    gripper = robot.gripper
    real_read = gripper.read_state

    def read_state(timeout_sec=None):
        # Fine while the session opens, then silent for good.
        if gripper.reads >= 3:
            raise AdapterUnavailable("no /joint_states")
        return real_read(timeout_sec)

    gripper.read_state = read_state
    assert main(["teach", "hold", *BOTH, "--execute", "--gravity", "1.0", "--seconds", "2"]) == 2
    out = capsys.readouterr().out
    assert "no joint state for" in out
    # The arm's torque was still released on the way out.
    assert np.allclose(robot.arm.published[-1], 0.0)


def test_soft_p_lowers_the_hands_gain_for_the_session_and_restores_it(robot, capsys):
    hand = robot.stub("tesollo_hand")
    hand.gain_p = {joint: 4.5 for joint in hand.joints}
    robot.arm.external = NO_HAND
    assert main(["teach", "hold", "--group", "openarm_right_arm", "--group", "tesollo_hand",
                 "--execute", "--seconds", "0.3", "--settle-sec", "0.2",
                 "--soft-p", "tesollo_hand=1.0"]) == 0
    out = capsys.readouterr().out
    assert "controller p 4.50 -> 1 for the session" in out and "restored" in out
    assert len(hand.gain_writes) == 2
    assert set(hand.gain_writes[0].values()) == {1.0}
    assert hand.gain_p == {joint: 4.5 for joint in hand.joints}


def test_soft_p_is_restored_even_when_the_session_fails(robot, monkeypatch, capsys):
    from robot_control import teach_cli
    from robot_control.ros_adapter import AdapterUnavailable

    monkeypatch.setattr(teach_cli, "STALE_LIMIT_SEC", 0.1)
    hand = robot.stub("tesollo_hand")
    hand.gain_p = {joint: 4.5 for joint in hand.joints}
    real_read = hand.read_state

    def read_state(timeout_sec=None):
        if hand.reads >= 3:
            raise AdapterUnavailable("gone")
        return real_read(timeout_sec)

    hand.read_state = read_state
    robot.arm.external = NO_HAND
    assert main(["teach", "hold", "--group", "openarm_right_arm", "--group", "tesollo_hand",
                 "--execute", "--seconds", "2", "--soft-p", "tesollo_hand=1.0"]) == 2
    assert hand.gain_p == {joint: 4.5 for joint in hand.joints}


def test_soft_p_refuses_a_group_without_gains_or_outside_the_session(robot, capsys):
    assert main(["teach", "hold", *LEFT_ARM, "--execute", "--seconds", "0.2",
                 "--soft-p", "openarm_left_arm=1.0"]) == 2
    assert "no PID gains" in capsys.readouterr().out
    assert main(["teach", "hold", *LEFT_ARM, "--soft-p", "tesollo_hand=1.0"]) == 2
    assert "not one of the session" in capsys.readouterr().out


def test_a_recording_survives_a_session_that_fails(robot, monkeypatch, tmp_path, capsys):
    from robot_control import teach_cli
    from robot_control.ros_adapter import AdapterUnavailable

    monkeypatch.setattr(teach_cli, "STALE_LIMIT_SEC", 0.1)
    robot.gripper.external = _squeeze
    gripper = robot.gripper
    real_read = gripper.read_state

    def read_state(timeout_sec=None):
        if gripper.reads >= 45:
            raise AdapterUnavailable("gone")
        return real_read(timeout_sec)

    gripper.read_state = read_state
    output = tmp_path / "partial.npz"
    assert main(["teach", "record", *BOTH, "--execute", "--output", str(output), *FAST]) == 2
    out = capsys.readouterr().out
    assert "no joint state for" in out and "recorded" in out
    raw = np.load(output, allow_pickle=False)
    assert raw["values"][-1, 7] < 0.02  # the squeeze that was taught is in the file


def test_a_joint_pushed_to_its_limit_is_named_as_it_happens(robot, capsys):
    def past_the_limit(reads):
        if reads < 15:
            return None
        hand = -DROOP.copy()
        hand[5] = 0.9  # l_aj_6 stops at +0.785
        return hand

    robot.arm.external = past_the_limit
    assert main(["teach", "hold", *LEFT_ARM, "--execute", *FAST]) == 0
    out = capsys.readouterr().out
    assert "l_aj_6 is at its upper limit (+0.785)" in out
    assert out.count("l_aj_6 is at its upper limit") <= 2  # not once per cycle
