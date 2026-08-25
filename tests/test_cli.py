import sys

import numpy as np
import pytest

from robot_control.cli import main

RIGHT_ARM = ["--group", "openarm_right_arm"]


@pytest.fixture
def no_ros(monkeypatch):
    """Make every rclpy import fail, as it does with no ROS sourced."""
    monkeypatch.setitem(sys.modules, "rclpy", None)


def test_preflight_reports_profile(capsys):
    assert main(["r2s", "preflight", "--profile", "openarm_tesollo"]) == 0
    output = capsys.readouterr().out
    assert "openarm_tesollo" in output
    assert "publish_enabled: false" in output


def test_collect_needs_to_be_told_which_group(capsys):
    """Group-scoped like everything else that reaches a controller.

    It used to build an excitation over all 43 profile joints — both arms plus
    the whole hand — which no single controller could have accepted.
    """
    assert main(["r2s", "collect", "--profile", "openarm_tesollo"]) == 2
    assert "--group" in capsys.readouterr().out


def test_collect_dry_run_still_needs_the_robot(no_ros, capsys):
    """The excitation is built around the arm's current pose, so a dry run has
    to read it. Building one around the midpoint of the range instead would
    review a track that --execute would not publish."""
    code = main(
        ["r2s", "collect", "--profile", "openarm_tesollo",
         "--group", "openarm_right_arm"]
    )

    assert code == 2
    assert "rclpy" in capsys.readouterr().out


def test_pose_joints_dry_run_prints_the_target_without_touching_ros(no_ros, capsys):
    """A dry run must resolve entirely offline, so it cannot reach the robot."""
    values = ",".join(["0.1"] * 7)

    assert main(["pose", "joints", *RIGHT_ARM, "--values", values]) == 0

    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert "right_joint_trajectory_controller" in output
    assert "openarm_right_joint1" in output
    assert "0.1" in output


def test_pose_joints_execute_without_an_adapter_exits_2(no_ros, capsys):
    values = ",".join(["0.1"] * 7)

    assert main(["pose", "joints", *RIGHT_ARM, "--values", values, "--execute"]) == 2

    assert "rclpy" in capsys.readouterr().out


def test_pose_rejects_an_unknown_group(capsys):
    assert main(["pose", "joints", "--group", "openarm_third_arm", "--values", "0"]) == 2
    assert "unknown group" in capsys.readouterr().out


def test_pose_joints_rejects_a_value_count_mismatch(capsys):
    assert main(["pose", "joints", *RIGHT_ARM, "--values", "0.1,0.2"]) == 2

    output = capsys.readouterr().out
    assert "7" in output and "2" in output


def test_pose_joints_rejects_a_non_numeric_value(capsys):
    assert main(["pose", "joints", *RIGHT_ARM, "--values", "0.1,x,0,0,0,0,0"]) == 2
    assert "x" in capsys.readouterr().out


def test_pose_joints_requires_values_or_a_named_state(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["pose", "joints", *RIGHT_ARM])
    assert exit_info.value.code == 2


def test_pose_joints_resolves_a_named_state_from_the_srdf(capsys):
    assert main(["pose", "joints", *RIGHT_ARM, "--named", "hands_up"]) == 0

    output = capsys.readouterr().out
    assert "DRY RUN" in output
    # hands_up sets joint4 to 2 rad and every other arm joint to zero.
    assert "2.0" in output


def test_pose_joints_rejects_a_named_state_absent_from_the_srdf(capsys):
    assert main(["pose", "joints", *RIGHT_ARM, "--named", "tea_pot"]) == 2

    output = capsys.readouterr().out
    assert "tea_pot" in output
    assert "hands_up" in output  # the error lists what is available


def test_pose_joints_refuses_a_target_outside_the_profile_limits(capsys):
    """Profile limits are authoritative, and a dry run still enforces them."""
    assert main(["pose", "joints", *RIGHT_ARM, "--values", "99,0,0,0,0,0,0"]) == 3
    assert "position limit" in capsys.readouterr().out


def test_the_gripper_can_reach_the_open_state_its_own_srdf_names(capsys):
    """The robot's own `open` has to be commandable, and once was not.

    The SRDF opens the gripper to 0.044 and the description's finger_joint1
    stops at 0.044, but the profile bounded it at 0.04, so the gate refused the
    vendor's own named state. That mismatch is the interesting thing to hold
    still: this asserts the three agree by exercising the one path that reads
    all of them. The gate itself is covered above, by a typed target that is
    genuinely out of range — a named state runs through the same check.
    """
    code = main(["pose", "joints", "--group", "openarm_left_gripper", "--named", "open"])

    output = capsys.readouterr().out
    assert code == 0, output
    assert "position limit" not in output
    assert "+0.0440" in output


def test_pose_ee_requires_a_target():
    with pytest.raises(SystemExit) as exit_info:
        main(["pose", "ee", *RIGHT_ARM])
    assert exit_info.value.code == 2


def test_pose_ee_rejects_both_a_typed_and_a_dragged_target():
    """--from-marker is a whole pose, so it cannot be combined with --xyz."""
    with pytest.raises(SystemExit) as exit_info:
        main(["pose", "ee", *RIGHT_ARM, "--xyz", "0,0,0.03", "--from-marker"])
    assert exit_info.value.code == 2


def test_pose_ee_from_marker_rejects_relative(no_ros, capsys):
    """The marker already carries an absolute pose; an offset from it is a
    request for something the operator cannot see on screen."""
    assert main(["pose", "ee", *RIGHT_ARM, "--from-marker", "--relative"]) == 2
    assert "--relative" in capsys.readouterr().out


def test_pose_ee_from_marker_rejects_rpy(no_ros, capsys):
    assert main(["pose", "ee", *RIGHT_ARM, "--from-marker", "--rpy", "0,0,0"]) == 2
    assert "--rpy" in capsys.readouterr().out


def test_pose_ee_from_marker_needs_ros(no_ros, capsys):
    assert main(["pose", "ee", *RIGHT_ARM, "--from-marker"]) == 2
    assert "rclpy" in capsys.readouterr().out


class DroopingArm:
    """A stub adapter whose joints stop short of every command, as the real
    ones do: the DM motors hold position by *having* a position error, so a
    command repeated unchanged reproduces the same shortfall."""

    #: Radians of droop at zero load angle. The real value is the holding
    #: torque divided by kp, so it follows the pose rather than the step size.
    DROOP = 0.05

    def __init__(self, *_args, **_kwargs):
        from robot_control.ros_adapter import Pose

        self._Pose = Pose
        self.joints = np.zeros(7)
        self.target = np.full(7, 0.2)
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return None

    # Position along a single axis stands in for the tool centre point, so the
    # residual the CLI prints is a readable multiple of the joint error.
    def read_pose(self):
        return self._Pose((float(self.joints.mean()), 0.0, 0.0), (0, 0, 0, 1), "world")

    def read_marker_pose(self):
        return self._Pose((float(self.target.mean()), 0.0, 0.0), (0, 0, 0, 1), "world")

    def read_state(self):
        return self.joints.copy()

    def solve_ik(self, _pose, seed):
        return self.target.copy()

    def send_trajectory(self, points, period_sec):
        command = np.asarray(points[-1], dtype=float)
        self.sent.append(command)
        # The joint settles where kp times the error balances the load, so it
        # stops short by an amount the pose sets, not the size of the step.
        self.joints = command - self.DROOP * np.cos(command)


@pytest.fixture
def drooping(monkeypatch):
    from robot_control import ros_adapter

    arm = DroopingArm()
    monkeypatch.setattr(ros_adapter, "RosAdapter", lambda *a, **k: arm)
    return arm


def test_pose_ee_reports_how_far_short_the_arm_stopped(drooping, capsys):
    """A silent miss is the failure mode; the residual is always printed."""
    assert main(["pose", "ee", *RIGHT_ARM, "--from-marker", "--execute"]) == 0

    output = capsys.readouterr().out
    assert "residual" in output
    assert len(drooping.sent) == 1


def test_pose_ee_settle_drives_the_residual_below_the_tolerance(drooping, capsys):
    assert (
        main(
            [
                "pose",
                "ee",
                *RIGHT_ARM,
                "--from-marker",
                "--execute",
                "--settle",
                "--tolerance",
                "0.002",
            ]
        )
        == 0
    )

    assert len(drooping.sent) > 1, "settle sent no correction"
    np.testing.assert_allclose(drooping.joints, drooping.target, atol=0.002)
    assert "settled" in capsys.readouterr().out


def test_pose_ee_settle_corrects_by_adding_the_shortfall(drooping):
    """Re-sending the IK solution unchanged would reproduce the same droop, so
    each pass has to command past the target by what the last pass missed."""
    main(["pose", "ee", *RIGHT_ARM, "--from-marker", "--execute", "--settle"])

    first, second = drooping.sent[0], drooping.sent[1]
    np.testing.assert_allclose(first, drooping.target)
    assert (second > first).all(), "the correction did not command past the target"


def test_pose_ee_settle_requires_execute(no_ros, capsys):
    """A dry run sends nothing, so there is no shortfall to correct."""
    assert main(["pose", "ee", *RIGHT_ARM, "--from-marker", "--settle"]) == 2
    assert "--execute" in capsys.readouterr().out


def test_pose_ee_needs_ros_even_for_a_dry_run(no_ros, capsys):
    """IK lives in move_group, so there is no offline form of pose ee."""
    assert main(["pose", "ee", *RIGHT_ARM, "--xyz", "0,0,0.03", "--relative"]) == 2
    assert "rclpy" in capsys.readouterr().out


def test_pose_ee_rejects_a_group_without_a_planning_group(no_ros, capsys):
    code = main(["pose", "ee", "--group", "tesollo_curl", "--xyz", "0,0,0.03"])

    assert code == 2
    # The error points at the command that does work for this group.
    assert "pose joints --values" in capsys.readouterr().out


def test_pose_show_reports_every_executable_group_offline(capsys):
    """Without ROS, show still reports the contract it would read against."""
    assert main(["pose", "show"]) == 2

    output = capsys.readouterr().out
    assert "openarm_right_arm" in output
    assert "right_joint_trajectory_controller" in output


class _TopicRecordingBackend:
    """Records which joint topic it was built for, and reads nothing."""

    def __init__(self, joint_topic):
        self.joint_topic = joint_topic

    def joint_states(self, timeout_sec):
        return {}

    def close(self):
        pass


@pytest.fixture
def built_backends(monkeypatch):
    built = []

    def _make(node_name="robot_control_pose", joint_topic="/joint_states"):
        backend = _TopicRecordingBackend(joint_topic)
        built.append(backend)
        return backend

    monkeypatch.setattr("robot_control.ros_adapter.make_backend", _make)
    return built


def test_pose_show_reads_a_group_where_that_group_publishes(built_backends, capsys):
    """A hand on its own controller_manager is not on the robot-wide topic.

    Reading it there does not return a partial pose, it returns nothing at all,
    and reports a bringup that is in fact running as absent.
    """
    main(["pose", "show", "--group", "tesollo_curl"])

    assert [b.joint_topic for b in built_backends] == ["/dg5f_right/joint_states"]


def test_pose_show_across_groups_reads_each_of_their_topics(built_backends, capsys):
    """One subscription cannot serve two topics, so showing everything needs both."""
    main(["pose", "show"])

    assert set(b.joint_topic for b in built_backends) == {
        "/joint_states",
        "/dg5f_right/joint_states",
    }


class StiffArm:
    """A stub adapter whose tracking error shrinks as gravity torque is added.

    Modelled the way the real arm behaves: a joint holds position by sitting
    short of its command by (unmet load / kp), so feedforward torque matching
    the load removes the error and torque past it overshoots.
    """

    #: What the hardware's DEFAULT_KP applies on the wrist joints upward.
    KP = 20.0

    def __init__(self):
        self.published = []
        self.applied = np.zeros(7)
        self.load = np.zeros(7)

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return None

    def read_robot_description(self):
        return "<robot name='stub'/>"  # unused: the chain is injected

    def read_state(self):
        return np.zeros(7)

    def send_effort(self, effort):
        self.applied = np.asarray(effort, dtype=float).copy()
        self.published.append(self.applied)

    def read_tracking_error(self):
        return (self.load - self.applied) / self.KP


def _stub_chain(masses):
    """A seven-joint chain about +y, each link's mass a metre out along +x."""
    from robot_control import kinematics

    joints = tuple(
        kinematics.Revolute(
            f"j{index}",
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0]) if index == 0 else np.array([1e-6, 0.0, 0.0]),
            np.eye(3),
            f"l{index}",
        )
        for index in range(7)
    )
    links = tuple(
        kinematics.Link(f"l{index}", mass, np.array([1.0, 0.0, 0.0]))
        for index, mass in enumerate(masses)
    )
    return kinematics.Chain(joints, links)


@pytest.fixture
def stiff(monkeypatch):
    from robot_control import ros_adapter

    arm = StiffArm()
    # One light link a metre out on the last joint only, so the modelled torque
    # is unambiguous and stays inside the 20 N.m the profile allows the wrist
    # joints even at the largest scale a sweep can ask for.
    chain = _stub_chain([0.0] * 6 + [0.5])
    arm.load = chain.gravity_torque(np.zeros(7))
    monkeypatch.setattr(ros_adapter, "RosAdapter", lambda *a, **k: arm)
    monkeypatch.setattr("robot_control.cli._gravity_chain", lambda *a: chain)
    return arm


def test_pose_gravity_dry_run_publishes_nothing(stiff, capsys):
    assert main(["pose", "gravity", *RIGHT_ARM, "--scale", "1.0"]) == 0

    assert stiff.published == []
    assert "DRY RUN" in capsys.readouterr().out


def test_pose_gravity_releases_the_torque_when_it_finishes(stiff, capsys):
    assert (
        main(
            ["pose", "gravity", *RIGHT_ARM, "--scale", "1.0", "--execute",
             "--hold-sec", "0.02"]
        )
        == 0
    )

    # Whatever was held during the run, the last thing published is zero: torque
    # left applied after the process exits would keep pushing.
    np.testing.assert_allclose(stiff.published[-1], np.zeros(7))
    assert "torque released" in capsys.readouterr().out


def test_pose_gravity_sweep_names_the_scale_that_measured_best(stiff, capsys):
    assert (
        main(
            ["pose", "gravity", *RIGHT_ARM, "--execute", "--hold-sec", "0.02",
             "--sweep", "0,0.5,1.0"]
        )
        == 0
    )

    output = capsys.readouterr().out
    # The stub's error vanishes at full compensation, so 1 must win.
    assert "best measured scale: 1" in output
    for scale in ("0.00", "0.50", "1.00"):
        assert scale in output


def test_pose_gravity_refuses_a_scale_that_would_overcompensate(stiff, capsys):
    assert main(["pose", "gravity", *RIGHT_ARM, "--scale", "3.0", "--execute"]) == 2

    assert "outside 0 to 1.5" in capsys.readouterr().out
    assert stiff.published == []


def test_pose_gravity_refuses_a_group_with_no_effort_controller(no_ros, capsys):
    code = main(["pose", "gravity", "--group", "tesollo_curl", "--scale", "1.0"])

    assert code == 2
    assert "effort_controller" in capsys.readouterr().out


def test_pose_gravity_refuses_torque_over_the_profile_limit(stiff, capsys, monkeypatch):
    """A scale inside range can still ask for more torque than a joint allows."""
    monkeypatch.setattr(
        "robot_control.cli._gravity_chain", lambda *a: _stub_chain([500.0] + [0.0] * 6)
    )

    code = main(["pose", "gravity", *RIGHT_ARM, "--scale", "1.0", "--execute"])

    assert code == 3
    assert "effort limit exceeded" in capsys.readouterr().out


class DraggableArm(StiffArm):
    """A stub that reports a marker being dragged, and follows what it is sent."""

    def __init__(self, target=None):
        super().__init__()
        self.joints = np.zeros(7)
        self.streamed = []
        self._target = target
        self.pumped = 0

    def watch_marker(self):
        self.watching = True

    def latest_marker_target(self):
        return self._target

    def pump(self, timeout_sec=0.0):
        self.pumped += 1

    def read_state(self, timeout_sec=None):
        return self.joints.copy()

    def stream_positions(self, positions):
        self.joints = np.asarray(positions, dtype=float).copy()
        self.streamed.append(self.joints)


def _reachable_target(chain, q):
    """A pose the stub chain can actually hold, so following can converge."""
    from robot_control.ros_adapter import Pose

    pose = chain.pose(q)
    trace = np.trace(pose[:3, :3])
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    if w < 1e-8:
        return Pose(tuple(pose[:3, 3]), (0.0, 0.0, 0.0, 1.0), "world")
    return Pose(
        tuple(pose[:3, 3]),
        (
            (pose[2, 1] - pose[1, 2]) / (4 * w),
            (pose[0, 2] - pose[2, 0]) / (4 * w),
            (pose[1, 0] - pose[0, 1]) / (4 * w),
            w,
        ),
        "world",
    )


@pytest.fixture
def draggable(monkeypatch):
    from robot_control import ros_adapter

    chain = _stub_chain([0.2] * 7)
    arm = DraggableArm(target=_reachable_target(chain, np.full(7, 0.05)))
    arm.load = chain.gravity_torque(np.zeros(7))
    monkeypatch.setattr(ros_adapter, "RosAdapter", lambda *a, **k: arm)
    monkeypatch.setattr("robot_control.cli._gravity_chain", lambda *a: chain)
    return arm


def test_pose_follow_dry_run_streams_nothing(draggable, capsys):
    assert main(["pose", "follow", *RIGHT_ARM, "--seconds", "0.1"]) == 0

    assert draggable.streamed == []
    assert "DRY RUN" in capsys.readouterr().out


def test_pose_follow_streams_towards_the_dragged_marker(draggable, capsys):
    assert (
        main(["pose", "follow", *RIGHT_ARM, "--execute", "--seconds", "0.4"]) == 0
    )

    assert draggable.streamed, "nothing was streamed"
    # Each sample must move the arm towards the marker, not away from it.
    assert np.abs(draggable.joints).sum() > 0.0
    assert "followed" in capsys.readouterr().out


def test_pose_follow_holds_still_when_no_marker_has_been_dragged(capsys, monkeypatch):
    """No target is not a reason to command zero; it is a reason to command
    nothing, so the controller keeps holding where the arm already is."""
    from robot_control import ros_adapter

    chain = _stub_chain([0.2] * 7)
    arm = DraggableArm(target=None)
    monkeypatch.setattr(ros_adapter, "RosAdapter", lambda *a, **k: arm)
    monkeypatch.setattr("robot_control.cli._gravity_chain", lambda *a: chain)

    assert main(["pose", "follow", *RIGHT_ARM, "--execute", "--seconds", "0.2"]) == 0

    assert arm.streamed == []


def test_pose_follow_releases_torque_it_was_holding(draggable, capsys):
    assert (
        main(
            ["pose", "follow", *RIGHT_ARM, "--execute", "--seconds", "0.2",
             "--gravity", "1.0"]
        )
        == 0
    )

    np.testing.assert_allclose(draggable.published[-1], np.zeros(7))


def test_pose_follow_refuses_a_group_with_no_planning_group(no_ros, capsys):
    code = main(["pose", "follow", "--group", "tesollo_curl", "--seconds", "1"])

    assert code == 2
    assert "no planning group" in capsys.readouterr().out


def test_pose_follow_refuses_an_out_of_range_gravity_scale(no_ros, capsys):
    assert main(["pose", "follow", *RIGHT_ARM, "--gravity", "9"]) == 2
    assert "outside 0 to 1.5" in capsys.readouterr().out


class DroopingDraggableArm(DraggableArm):
    """A dragging stub that also droops, which fake hardware never does.

    This is the case that mattered: with droop larger than one period's velocity
    budget, a command rate-limited from the measured pose can never advance, and
    the arm sits still while the loop reports thousands of samples sent.
    """

    #: Radians the joint sits behind its command in order to hold position.
    DROOP = 0.03

    def stream_positions(self, positions):
        command = np.asarray(positions, dtype=float)
        self.streamed.append(command.copy())
        # Move towards the command, stopping the droop short of it.
        gap = command - self.joints
        self.joints = self.joints + np.sign(gap) * np.maximum(
            np.abs(gap) - self.DROOP, 0.0
        )


def test_pose_follow_advances_a_drooping_arm(monkeypatch, capsys):
    """The regression that a perfect-tracking stub cannot catch."""
    from robot_control import ros_adapter

    chain = _stub_chain([0.2] * 7)
    arm = DroopingDraggableArm(target=_reachable_target(chain, np.full(7, 0.15)))
    arm.load = chain.gravity_torque(np.zeros(7))
    monkeypatch.setattr(ros_adapter, "RosAdapter", lambda *a, **k: arm)
    monkeypatch.setattr("robot_control.cli._gravity_chain", lambda *a: chain)

    assert main(["pose", "follow", *RIGHT_ARM, "--execute", "--seconds", "1.0"]) == 0

    assert arm.streamed, "nothing was streamed"
    # The arm has to have actually moved, not merely been commanded at.
    assert np.abs(arm.joints).max() > DroopingDraggableArm.DROOP, (
        f"a drooping arm did not advance: {arm.joints}"
    )


def test_pose_follow_reports_how_far_it_trailed_the_marker(draggable, capsys):
    """Clamp counts say the command moved; only the lag says the arm did."""
    assert main(["pose", "follow", *RIGHT_ARM, "--execute", "--seconds", "0.3"]) == 0

    output = capsys.readouterr().out
    assert "trailed the marker by" in output
    assert "mm on average" in output


def test_pose_gravity_accepts_one_scale_per_joint(stiff, capsys):
    """The measured optima differ per joint, so one number is a compromise."""
    assert (
        main(
            ["pose", "gravity", *RIGHT_ARM, "--execute", "--hold-sec", "0.02",
             "--scale", "1.23,1.14,1.31,1.1,1.0,1.0,1.0"]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "1.23" in output and "1.14" in output


def test_pose_gravity_rejects_a_scale_count_that_is_not_one_or_per_joint(stiff, capsys):
    assert main(["pose", "gravity", *RIGHT_ARM, "--scale", "1.0,1.1"]) == 2
    assert "one value or one per joint" in capsys.readouterr().out


def test_pose_gravity_sweeps_a_single_joint_holding_the_others(stiff, capsys):
    """Refining one joint at a time is what per-joint tuning actually is."""
    assert (
        main(
            ["pose", "gravity", *RIGHT_ARM, "--execute", "--hold-sec", "0.02",
             "--scale", "1.1", "--sweep", "0.9,1.0,1.1", "--sweep-joint", "r_aj_7"]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "best measured scale for r_aj_7" in output
    # The suggested next command carries every joint's scale, not just this one.
    assert "--scale 1.1,1.1,1.1,1.1,1.1,1.1," in output


def test_pose_gravity_rejects_sweeping_a_joint_outside_the_group(stiff, capsys):
    code = main(
        ["pose", "gravity", *RIGHT_ARM, "--sweep", "1.0", "--sweep-joint", "l_aj_1"]
    )

    assert code == 2
    assert "not a joint of" in capsys.readouterr().out


def test_pose_gravity_requires_a_scale_or_a_sweep(stiff, capsys):
    assert main(["pose", "gravity", *RIGHT_ARM]) == 2
    assert "--scale, --sweep, or both" in capsys.readouterr().out


def test_pose_gravity_writes_what_it_measured(stiff, tmp_path, capsys):
    from robot_control.artifacts import read_sweep
    from robot_control.profile import load_builtin_profile

    path = tmp_path / "sweep.json"
    code = main(
        ["pose", "gravity", *RIGHT_ARM, "--execute", "--hold-sec", "0.02",
         "--sweep", "0,0.5,1.0", "--sweep-joint", "r_aj_7", "--output", str(path)]
    )

    assert code == 0
    assert str(path) in capsys.readouterr().out
    sweep = read_sweep(path, load_builtin_profile("openarm_tesollo"))
    assert sweep.rounds == 3
    assert sweep.group == "openarm_right_arm"
    assert sweep.sweep_joint == "r_aj_7"
    # The scale that varied is the one the sweep was told to vary, and the
    # errors are the ones the run printed, not a recomputation.
    np.testing.assert_allclose(sweep.scales[:, 6], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(sweep.errors[-1], np.zeros(7), atol=1e-9)


def test_pose_gravity_writes_a_single_scale_too(stiff, tmp_path):
    """One pose at one scale is a round; several files make the fit."""
    from robot_control.artifacts import read_sweep
    from robot_control.profile import load_builtin_profile

    path = tmp_path / "one.json"
    code = main(
        ["pose", "gravity", *RIGHT_ARM, "--execute", "--hold-sec", "0.02",
         "--scale", "1.0", "--output", str(path)]
    )

    assert code == 0
    assert read_sweep(path, load_builtin_profile("openarm_tesollo")).rounds == 1


def test_pose_gravity_output_needs_execute(stiff, capsys):
    """A dry run measures nothing, so there is nothing honest to write."""
    code = main(["pose", "gravity", *RIGHT_ARM, "--scale", "1.0", "--output", "x.json"])

    assert code == 2
    assert "--execute" in capsys.readouterr().out


def test_pose_follow_accepts_one_gravity_scale_per_joint(draggable, capsys):
    assert (
        main(
            ["pose", "follow", *RIGHT_ARM, "--execute", "--seconds", "0.2",
             "--gravity", "1.23,1.14,1.31,1.1,1.0,1.0,1.0"]
        )
        == 0
    )

    np.testing.assert_allclose(draggable.published[-1], np.zeros(7))


def test_pose_follow_without_gravity_publishes_no_torque(draggable, capsys):
    assert main(["pose", "follow", *RIGHT_ARM, "--execute", "--seconds", "0.2"]) == 0

    assert draggable.published == [], "torque was published with --gravity absent"
    assert "gravity off" in capsys.readouterr().out
