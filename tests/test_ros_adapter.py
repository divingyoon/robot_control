"""Adapter tests run without ROS by injecting a recording backend."""

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from robot_control.profile import load_profile
from robot_control.ros_adapter import (
    AdapterUnavailable,
    IkFailed,
    Pose,
    RosAdapter,
    gripper_failure,
    quaternion_from_rpy,
    rpy_from_quaternion,
    trajectory_failure,
)
from robot_control.safety import SafetyError

MOVEIT_SUCCESS = 1
NEUTRAL = (0.0, 0.0, 0.0, 1.0)


@pytest.fixture
def profile(tmp_path):
    """Two arm joints with opposing signs, a hand joint, and an idle group."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("control_joint_order: [a1, a2, h1, x1]\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    path = tmp_path / "profile.yaml"
    path.write_text(
        f"""
name: fixture
components: [openarm]
asset: {{id: asset, manifest: manifest.yaml, manifest_sha256: {digest}}}
joints:
  - {{canonical: a1, source: arm_1, sign: 1, unit: rad, lower: -1, upper: 1, velocity: 1, effort: 1}}
  - {{canonical: a2, source: arm_2, sign: -1, unit: rad, lower: -1, upper: 1, velocity: 1, effort: 1}}
  - {{canonical: h1, source: hand_1, sign: 1, unit: rad, lower: -1, upper: 1, velocity: 1, effort: 1}}
  - {{canonical: x1, source: idle_1, sign: 1, unit: rad, lower: -1, upper: 1, velocity: 1, effort: 1}}
groups:
  arm:
    joints: [a1, a2]
    controller: arm_controller
    moveit_group: arm_group
    tip_link: arm_tip
  hand:
    joints: [h1]
    controller: hand_controller
    action: parallel_gripper_command
    state_topic: /hand_ns/joint_states
  idle:
    joints: [x1]
ros:
  jazzy: {{command_topic: /cmd, state_topic: /state, controller: c, command_rate_hz: 100}}
"""
    )
    return load_profile(path)


class RecordingBackend:
    """Stands in for rclpy, recording what the adapter would put on the wire."""

    def __init__(self, states=None, ik=None, fk=None, marker=None):
        self.states = dict(states or {"arm_1": 0.1, "arm_2": -0.2, "hand_1": 0.3})
        self._ik = ik if ik is not None else (MOVEIT_SUCCESS, {})
        self._fk = fk if fk is not None else (MOVEIT_SUCCESS, Pose((1.0, 2.0, 3.0), NEUTRAL))
        self._marker = marker
        self.ik_requests = []
        self.fk_requests = []
        self.marker_requests = []
        self.trajectories = []
        self.streamed = []
        self.gripper_goals = []
        self.closed = False
        #: What this backend will deliver on the next pump, as
        #: (stamp_ns, {source: position}). A test sets it to stand in for
        #: whatever arrives on /joint_states.
        self.incoming: list = []
        self.recording = False
        self.clock_reads = 0
        self._recorded: list = []

    def joint_states(self, timeout_sec):
        return dict(self.states)

    def publish_trajectory_point(self, controller, joint_names, positions, horizon_sec):
        self.streamed.append((controller, list(joint_names), list(positions), horizon_sec))

    # --- recording -------------------------------------------------------
    def start_recording(self):
        self.recording = True
        self._recorded = []

    def pump(self, timeout_sec=0.0):
        # A real subscription delivers while the node spins, so the fake does
        # its delivering here too: a recorder that forgets to pump gets nothing.
        if self.recording:
            self._recorded.extend(self.incoming)
            self.incoming = []

    def stop_recording(self):
        self.recording = False
        return list(self._recorded)

    def now_ns(self):
        self.clock_reads += 1
        return 1_000_000_000 + self.clock_reads

    def compute_fk(self, link, seed, frame_id, timeout_sec):
        self.fk_requests.append((link, dict(seed), frame_id))
        return self._fk

    def marker_pose(self, name, timeout_sec):
        self.marker_requests.append(name)
        return self._marker

    def compute_ik(self, group, link, pose, seed, timeout_sec):
        self.ik_requests.append((group, link, pose, dict(seed)))
        return self._ik

    def follow_joint_trajectory(self, controller, joint_names, points, period_sec):
        self.trajectories.append(
            (controller, tuple(joint_names), [list(p) for p in points], period_sec)
        )

    def gripper_command(self, controller, joint, position):
        self.gripper_goals.append((controller, joint, position))

    def close(self):
        self.closed = True


def _adapter(profile, group="arm", execute=True, backend=None):
    return RosAdapter(
        profile, group, execute=execute, backend=backend or RecordingBackend()
    )


def test_adapter_reads_without_execute_but_refuses_to_send(profile):
    """Observing the robot is not publishing to it."""
    backend = RecordingBackend(ik=(MOVEIT_SUCCESS, {"arm_1": 0.4, "arm_2": -0.5}))
    adapter = RosAdapter(profile, "arm", execute=False, backend=backend)

    np.testing.assert_allclose(adapter.read_state(), [0.1, 0.2])
    adapter.read_pose()
    adapter.solve_ik(Pose((0.1, 0.2, 0.3), NEUTRAL), seed=np.array([0.0, 0.0]))

    with pytest.raises(SafetyError, match="--execute"):
        adapter.send_trajectory([np.array([0.1, 0.2])], period_sec=0.5)
    assert backend.trajectories == []


def test_marker_pose_reads_the_goal_the_operator_dragged(profile):
    """The RViz goal marker is a pose source like any other.

    MoveIt names the interactive marker after the end effector's parent link,
    so the adapter asks for the marker belonging to this group's tip rather
    than taking whichever marker the panel happens to be showing.
    """
    dragged = Pose((0.4, 0.1, 0.2), NEUTRAL, "world")
    backend = RecordingBackend(marker=dragged)

    pose = RosAdapter(profile, "arm", execute=False, backend=backend).read_marker_pose()

    assert pose == dragged
    assert backend.marker_requests == ["EE:goal_arm_tip"]


def test_marker_pose_says_which_marker_is_missing(profile):
    """A missing marker is a live RViz showing a different planning group.

    The panel only publishes a marker for the group it is set to, so the fix
    is a change in RViz, and the error has to say so.
    """
    backend = RecordingBackend(marker=None)

    with pytest.raises(AdapterUnavailable, match="EE:goal_arm_tip"):
        RosAdapter(profile, "arm", execute=False, backend=backend).read_marker_pose()


def test_marker_pose_refuses_a_group_with_no_planning_group(profile):
    """No tip link means no end-effector marker to read."""
    with pytest.raises(ValueError, match="no planning group"):
        RosAdapter(profile, "hand", execute=False, backend=RecordingBackend()).read_marker_pose()


def test_gripper_send_also_requires_execute(profile):
    backend = RecordingBackend()

    with pytest.raises(SafetyError, match="--execute"):
        RosAdapter(profile, "hand", execute=False, backend=backend).send_gripper(0.02)
    assert backend.gripper_goals == []


def test_adapter_reports_unavailable_when_rclpy_is_missing(profile, monkeypatch):
    # A None entry makes `import rclpy` fail exactly as an absent install does.
    monkeypatch.setitem(sys.modules, "rclpy", None)

    with pytest.raises(AdapterUnavailable, match="rclpy"):
        RosAdapter(profile, "arm", execute=True)


def test_the_backend_listens_where_the_group_publishes_its_state(profile, monkeypatch):
    """A hand on its own controller_manager publishes its own /joint_states.

    Subscribing to the default topic for such a group is not a partial read —
    the hand's joints never appear on it at all, so `pose joints` fails on a
    timeout that reads as "bringup is not running" while the bringup runs.
    """
    subscribed = []

    class _Backend:
        def __init__(self, node_name, joint_topic):
            subscribed.append((node_name, joint_topic))

        def close(self):
            pass

    monkeypatch.setattr("robot_control.ros_adapter._RclpyBackend", _Backend)

    RosAdapter(profile, "arm")
    RosAdapter(profile, "hand")

    assert [topic for _, topic in subscribed] == [
        "/joint_states",
        "/hand_ns/joint_states",
    ]


def test_adapter_rejects_a_group_without_a_controller(profile):
    with pytest.raises(ValueError, match="idle.*no controller"):
        _adapter(profile, group="idle")


def test_adapter_rejects_an_unknown_group(profile):
    with pytest.raises(ValueError, match="unknown group"):
        _adapter(profile, group="third_arm")


def test_read_state_converts_source_names_to_canonical_order_and_sign(profile):
    backend = RecordingBackend(states={"arm_1": 0.1, "arm_2": -0.2, "hand_1": 0.3})

    state = _adapter(profile, backend=backend).read_state()

    # a2 has sign -1, so a source of -0.2 is canonical +0.2.
    np.testing.assert_allclose(state, [0.1, 0.2])


def test_read_state_reports_a_group_that_the_robot_does_not_publish(profile):
    backend = RecordingBackend(states={"hand_1": 0.3})

    with pytest.raises(AdapterUnavailable, match="arm_1"):
        _adapter(profile, backend=backend).read_state()


def test_solve_ik_seeds_with_source_names_and_returns_canonical_order(profile):
    backend = RecordingBackend(ik=(MOVEIT_SUCCESS, {"arm_1": 0.4, "arm_2": -0.5}))
    adapter = _adapter(profile, backend=backend)

    solution = adapter.solve_ik(Pose((0.1, 0.2, 0.3), NEUTRAL), seed=np.array([0.1, 0.2]))

    np.testing.assert_allclose(solution, [0.4, 0.5])
    group, link, _pose, seed = backend.ik_requests[0]
    assert (group, link) == ("arm_group", "arm_tip")
    assert seed == {"arm_1": 0.1, "arm_2": -0.2}


def test_solve_ik_raises_on_a_service_failure_code(profile):
    backend = RecordingBackend(ik=(-31, {}))

    with pytest.raises(IkFailed, match="-31"):
        _adapter(profile, backend=backend).solve_ik(
            Pose((0.1, 0.2, 0.3), NEUTRAL), seed=np.array([0.0, 0.0])
        )


def test_solve_ik_ignores_joints_outside_the_group(profile):
    """MoveIt answers with the whole robot state, not just the planning group."""
    backend = RecordingBackend(
        ik=(MOVEIT_SUCCESS, {"arm_1": 0.4, "arm_2": -0.5, "hand_1": 0.9, "idle_1": 0.0})
    )

    solution = _adapter(profile, backend=backend).solve_ik(
        Pose((0.1, 0.2, 0.3), NEUTRAL), seed=np.array([0.0, 0.0])
    )

    np.testing.assert_allclose(solution, [0.4, 0.5])


def test_ee_operations_require_a_planning_group(profile):
    adapter = _adapter(profile, group="hand")

    with pytest.raises(ValueError, match="hand.*joint values"):
        adapter.solve_ik(Pose((0.0, 0.0, 0.0), NEUTRAL), seed=np.array([0.0]))
    with pytest.raises(ValueError, match="hand.*joint values"):
        adapter.read_pose()


def test_read_pose_seeds_fk_with_the_whole_robot(profile):
    """The tip's transform depends on the whole chain, not just the group."""
    backend = RecordingBackend(fk=(MOVEIT_SUCCESS, Pose((0.5, 0.0, 0.2), NEUTRAL)))

    pose = _adapter(profile, backend=backend).read_pose()

    assert pose.position == (0.5, 0.0, 0.2)
    link, seed, _frame = backend.fk_requests[0]
    assert link == "arm_tip"
    assert set(seed) == {"arm_1", "arm_2", "hand_1"}


def test_send_trajectory_targets_the_controller_declared_by_the_group(profile):
    backend = RecordingBackend()

    _adapter(profile, backend=backend).send_trajectory(
        [np.array([0.1, 0.2]), np.array([0.3, 0.4])], period_sec=0.5
    )

    controller, names, points, period = backend.trajectories[0]
    assert controller == "arm_controller"
    assert names == ("arm_1", "arm_2")
    # The sign flip must survive all the way to the wire.
    assert points == [[0.1, -0.2], [0.3, -0.4]]
    assert period == 0.5


def test_send_trajectory_refuses_a_gripper_group(profile):
    adapter = _adapter(profile, group="hand")

    with pytest.raises(ValueError, match="gripper_command"):
        adapter.send_trajectory([np.array([0.02])], period_sec=0.5)


def test_send_gripper_uses_the_gripper_action_and_source_sign(profile):
    backend = RecordingBackend()

    _adapter(profile, group="hand", backend=backend).send_gripper(0.02)

    # ParallelGripperCommand carries a JointState, so the goal names its joint.
    assert backend.gripper_goals == [("hand_controller", "hand_1", 0.02)]


def test_send_gripper_refuses_a_trajectory_group(profile):
    with pytest.raises(ValueError, match="follow_joint_trajectory"):
        _adapter(profile).send_gripper(0.02)


def test_closing_the_adapter_closes_the_backend(profile):
    backend = RecordingBackend()

    with _adapter(profile, backend=backend) as adapter:
        adapter.read_state()
    assert backend.closed


@pytest.mark.parametrize(
    "rpy",
    [(0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0), (0.1, -0.2, 0.3), (0.0, 0.0, math.pi)],
)
def test_rpy_and_quaternion_round_trip(rpy):
    restored = rpy_from_quaternion(quaternion_from_rpy(*rpy))

    # Compare through the quaternion, since Euler angles are not unique.
    np.testing.assert_allclose(
        quaternion_from_rpy(*restored), quaternion_from_rpy(*rpy), atol=1e-12
    )


def test_quaternion_from_rpy_is_normalized_and_ordered_xyzw():
    x, y, z, w = quaternion_from_rpy(0.0, 0.0, math.pi / 2)

    np.testing.assert_allclose([x, y, z], [0.0, 0.0, math.sin(math.pi / 4)], atol=1e-12)
    np.testing.assert_allclose(w, math.cos(math.pi / 4), atol=1e-12)


def test_pose_translated_offsets_position_and_keeps_orientation():
    pose = Pose((1.0, 2.0, 3.0), quaternion_from_rpy(0.0, 0.0, 0.5), frame_id="world")

    moved = pose.translated((0.0, 0.0, 0.03))

    np.testing.assert_allclose(moved.position, (1.0, 2.0, 3.03))
    assert moved.orientation == pose.orientation
    assert moved.frame_id == "world"


@dataclass
class _TrajectoryResult:
    error_code: int
    error_string: str = ""


@dataclass
class _GripperResult:
    reached_goal: bool
    stalled: bool = False


def test_action_results_are_judged_by_their_own_fields():
    """The two actions share no result field, not even an error code.

    GripperCommand.Result has position, effort, stalled, and reached_goal;
    FollowJointTrajectory.Result has error_code and error_string. Judging both
    by error_code raises AttributeError on every gripper command.
    """
    assert trajectory_failure(_TrajectoryResult(error_code=0)) is None
    assert "-1" in trajectory_failure(_TrajectoryResult(-1, "path tolerance"))

    assert gripper_failure(_GripperResult(reached_goal=True)) is None
    # A parallel gripper reports closing on an object by stalling.
    assert gripper_failure(_GripperResult(reached_goal=False, stalled=True)) is None
    assert gripper_failure(_GripperResult(reached_goal=False)) is not None


def test_recording_keeps_every_message_with_its_own_stamp(profile):
    """A stream, not a latest value. read_state discards on purpose; this cannot."""
    backend = RecordingBackend()
    adapter = _adapter(profile, execute=False, backend=backend)
    adapter.start_recording()
    backend.incoming = [
        (100, {"arm_1": 0.1, "arm_2": -0.2, "hand_1": 0.0}),
        (200, {"arm_1": 0.3, "arm_2": -0.4, "hand_1": 0.0}),
    ]
    adapter.pump()
    backend.incoming = [(300, {"arm_1": 0.5, "arm_2": -0.6, "hand_1": 0.0})]
    adapter.pump()

    recording = adapter.stop_recording()

    np.testing.assert_array_equal(recording.timestamps_ns, [100, 200, 300])
    # Canonical, so a2's opposing sign is applied the same way read_state does.
    np.testing.assert_allclose(recording.values, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    assert recording.joint_names == ("a1", "a2")


def test_a_recorder_that_never_pumps_gets_nothing(profile):
    """Subscriptions deliver while the node spins, so the loop must pump."""
    backend = RecordingBackend()
    adapter = _adapter(profile, execute=False, backend=backend)
    adapter.start_recording()
    backend.incoming = [(100, {"arm_1": 0.1, "arm_2": -0.2, "hand_1": 0.0})]

    with pytest.raises(AdapterUnavailable, match="no /joint_states"):
        adapter.stop_recording()


def test_a_message_missing_the_group_is_counted_not_fatal(profile):
    """During bringup a message may not cover the group yet; that is a gap."""
    backend = RecordingBackend()
    adapter = _adapter(profile, execute=False, backend=backend)
    adapter.start_recording()
    backend.incoming = [
        (100, {"arm_1": 0.1}),  # arm_2 absent
        (200, {"arm_1": 0.3, "arm_2": -0.4}),
    ]
    adapter.pump()

    recording = adapter.stop_recording()

    np.testing.assert_array_equal(recording.timestamps_ns, [200])
    assert recording.incomplete == 1


def test_recording_does_not_disturb_read_state(profile):
    """read_state nulls its cache and waits; recording must not depend on that."""
    backend = RecordingBackend()
    adapter = _adapter(profile, execute=False, backend=backend)
    adapter.start_recording()
    backend.incoming = [(100, {"arm_1": 0.1, "arm_2": -0.2, "hand_1": 0.0})]
    adapter.pump()

    np.testing.assert_allclose(adapter.read_state(), [0.1, 0.2])
    assert len(adapter.stop_recording()) == 1


def test_the_clock_a_command_is_stamped_with_is_the_node_s(profile):
    """Commands and measurements have to land on one epoch or normalize is junk.

    The measurement's stamp comes from the publisher's clock and the command's
    from ours, so the only thing this side can guarantee is that ours is the
    node clock — never wall time, which would be a different epoch under
    simulation time and a different one again after a reboot.
    """
    backend = RecordingBackend()
    adapter = _adapter(profile, execute=False, backend=backend)

    first, second = adapter.now_ns(), adapter.now_ns()

    assert backend.clock_reads == 2
    assert second > first


def test_stopping_without_starting_is_refused(profile):
    adapter = _adapter(profile, execute=False)

    with pytest.raises(AdapterUnavailable, match="not recording"):
        adapter.stop_recording()


def test_core_package_imports_without_rclpy():
    """The core CLI must fail cleanly without ROS, not crash on import.

    Checked in a subprocess rather than by inspecting this process's
    sys.modules: on a machine with Jazzy sourced, any earlier test that builds
    a real adapter imports rclpy for the whole session, and an in-process check
    would then report a failure that says nothing about the import contract.
    """
    probe = (
        "import sys\n"
        "sys.modules['rclpy'] = None\n"  # importable name, unusable module
        "import robot_control.cli, robot_control.ros_adapter as adapter\n"
        "print(adapter.__file__)\n"
    )
    root = Path(__file__).parents[1]
    # The subprocess gets no conftest, so hand it the same source path a bare
    # checkout relies on. Without an installed package it would otherwise fail
    # on `import robot_control` and report that as an rclpy contract violation.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=root,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).name == "ros_adapter.py"


def test_a_streamed_point_is_due_immediately(profile):
    """Why the horizon is zero, and not the stream's period.

    `joint_trajectory_controller` configured with `interpolation_method: none`
    returns the state a trajectory was installed with for any sample taken
    before that trajectory's first point comes due — see `Trajectory::sample`
    in ros2_controllers. A servo stream republishes every period, and each
    message resets the trajectory, so a point one period ahead puts every
    sample inside that window: the reference never advances and the arm never
    moves, with nothing logged and the goal reported reached.

    Measured on the OpenArm: reference span 0.00001 rad against a commanded
    0.06, while the same arm tracked an action goal on the same controller.
    """
    backend = RecordingBackend()
    adapter = _adapter(profile, backend=backend)

    adapter.stream_positions([0.4, -0.5])

    controller, _, _, horizon = backend.streamed[-1]
    assert controller == "arm_controller"
    assert horizon == 0.0
