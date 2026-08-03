"""Optional ROS execution edge for canonical commands.

``import robot_control`` must not require ``rclpy``, so every ROS import lives
inside :class:`_RclpyBackend` and surfaces as :class:`AdapterUnavailable`
rather than as an import error. The adapter itself is pure Python and takes an
injected backend, which is how it is tested without a running ROS graph.

Nothing here decides whether a motion is safe. The adapter converts between the
canonical boundary and ROS names, and puts on the wire exactly what it is
given; :class:`~robot_control.safety.CommandGate` is what authorizes it.

Reading state, forward kinematics, and inverse kinematics only observe the
robot, so they need no ``execute``. Sending a goal does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from .interface import CanonicalInterface, InterfaceError
from .profile import FOLLOW_JOINT_TRAJECTORY, PARALLEL_GRIPPER_COMMAND, RobotProfile
from .safety import SafetyError
from .track import Recording

# moveit_msgs/MoveItErrorCodes.SUCCESS
MOVEIT_SUCCESS = 1
DEFAULT_TIMEOUT_SEC = 10.0

#: Where joint positions come from for a group that does not say otherwise.
#: A group under its own controller_manager overrides it with `state_topic`.
JOINT_STATES_TOPIC = "/joint_states"

# MoveIt's RobotInteraction names each end-effector marker after the parent
# link of the SRDF end effector, which is why tip_link has to be that same link.
MARKER_PREFIX = "EE:goal_"
# The interactive marker server lives inside the RViz MotionPlanning display.
MARKER_SERVICE = (
    "/rviz_moveit_motion_planning_display"
    "/robot_interaction_interactive_marker_topic/get_interactive_markers"
)
# Published by the RViz display while a drag is in progress. Useless for reading
# a pose once, which is what the service is for, and exactly right for following.
MARKER_FEEDBACK_TOPIC = (
    "/rviz_moveit_motion_planning_display"
    "/robot_interaction_interactive_marker_topic/feedback"
)


class AdapterUnavailable(RuntimeError):
    """ROS is not installed, not running, or not answering."""


class IkFailed(RuntimeError):
    """The IK service answered, and reported that the pose is not reachable."""


@dataclass(frozen=True)
class Pose:
    """An end-effector pose. Orientation is a quaternion in x, y, z, w order."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    frame_id: str = "world"

    def translated(self, offset: Sequence[float]) -> Pose:
        """Return this pose shifted by *offset*, keeping its orientation."""
        if len(offset) != 3:
            raise ValueError("a translation offset needs exactly three values")
        moved = tuple(float(a + b) for a, b in zip(self.position, offset))
        return Pose(moved, self.orientation, self.frame_id)

    def rotated_to(self, orientation: Sequence[float]) -> Pose:
        if len(orientation) != 4:
            raise ValueError("an orientation needs exactly four values")
        return Pose(self.position, tuple(float(v) for v in orientation), self.frame_id)

    @property
    def rpy(self) -> tuple[float, float, float]:
        return rpy_from_quaternion(self.orientation)


def quaternion_from_rpy(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """Convert fixed-axis roll/pitch/yaw to a unit quaternion in x, y, z, w order."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rpy_from_quaternion(
    orientation: Sequence[float],
) -> tuple[float, float, float]:
    """Convert a quaternion in x, y, z, w order to fixed-axis roll/pitch/yaw."""
    x, y, z, w = (float(v) for v in orientation)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    # asin saturates rather than raising when the quaternion is slightly denormal.
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (roll, pitch, yaw)


def trajectory_failure(result: Any) -> str | None:
    """Describe why a FollowJointTrajectory result failed, or return None."""
    if result.error_code == 0:
        return None
    return f"error code {result.error_code}: {result.error_string}"


def gripper_failure(result: Any) -> str | None:
    """Describe why a ParallelGripperCommand result failed, or return None.

    The result carries no error code at all, only state, stalled, and
    reached_goal. Stalling is how a parallel gripper reports that it closed on
    something, so it counts as success.
    """
    if result.reached_goal or result.stalled:
        return None
    return "the gripper neither reached its commanded position nor stalled"


def make_backend(
    node_name: str = "robot_control_pose", joint_topic: str = JOINT_STATES_TOPIC
) -> Any:
    """Build one ROS backend, shareable by the adapters reading *joint_topic*.

    Reading every group costs one node and one rclpy context rather than one
    per group, and each adapter created with it must not close it. Sharing
    stops at the topic: one subscription cannot serve two, so groups that
    publish elsewhere need a backend of their own.
    """
    return _RclpyBackend(node_name, joint_topic)


class RosAdapter:
    """Convert canonical commands to controller traffic for one actuator group."""

    def __init__(
        self,
        profile: RobotProfile,
        group_name: str,
        execute: bool = False,
        backend: Any | None = None,
        node_name: str = "robot_control_pose",
    ):
        if group_name not in profile.groups:
            raise ValueError(
                f"unknown group {group_name!r}; known groups are "
                f"{sorted(profile.groups)}"
            )
        group = profile.groups[group_name]
        if group.controller is None:
            raise ValueError(
                f"group {group_name!r} declares no controller, so it cannot be "
                "commanded; add one to the profile"
            )
        self.profile = profile
        self.group = group
        # Reading state, FK, and IK observe the robot and are always allowed;
        # only putting a goal on the wire needs --execute.
        self.execute = execute
        self.interface = CanonicalInterface(profile)
        self._backend = (
            backend
            if backend is not None
            else _RclpyBackend(node_name, group.state_topic or JOINT_STATES_TOPIC)
        )
        self._recording = False

    def __enter__(self) -> RosAdapter:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()

    def close(self) -> None:
        self._backend.close()

    @property
    def state_topic(self) -> str:
        """Where this group's joint positions are published."""
        return self.group.state_topic or JOINT_STATES_TOPIC

    def read_source_state(
        self, timeout_sec: float = DEFAULT_TIMEOUT_SEC
    ) -> dict[str, float]:
        """Return the latest joint state, keyed by source joint name."""
        return self._backend.joint_states(timeout_sec)

    def read_state(self, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> np.ndarray:
        """Return this group's canonical joint values."""
        source = self.read_source_state(timeout_sec)
        try:
            return self.interface.group_state_to_canonical(self.group.name, source)
        except InterfaceError as error:
            # The graph is up but not publishing this group; that is an
            # environment problem, not a bad request.
            raise AdapterUnavailable(
                f"{self.state_topic} does not cover group "
                f"{self.group.name!r}: {error}"
            ) from error

    def read_pose(self, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> Pose:
        """Return the group's end-effector pose from ``/compute_fk``."""
        tip = self._require_planning_group()
        seed = self.read_source_state(timeout_sec)
        code, pose = self._backend.compute_fk(tip, seed, "world", timeout_sec)
        if code != MOVEIT_SUCCESS or pose is None:
            raise IkFailed(
                f"forward kinematics for {tip} failed with MoveIt error code {code}"
            )
        return pose

    def read_robot_description(self, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> str:
        """Return the URDF the running stack is actually using.

        Read from the graph rather than rendered locally, so gravity is computed
        for the robot that is running: a locally rendered URDF could differ from
        the one the controllers were configured against.
        """
        return self._backend.robot_description(timeout_sec)

    def read_tracking_error(
        self, timeout_sec: float = DEFAULT_TIMEOUT_SEC
    ) -> np.ndarray:
        """Return the group's per-joint position error, in canonical order.

        The trajectory controller publishes the difference between what it asked
        for and what it reads back, which is the droop directly. Deriving it from
        the tool centre point instead would fold in IK and FK.
        """
        # __init__ already refused a group with no controller.
        controller = self.group.controller
        errors = self._backend.tracking_error(controller, timeout_sec)
        try:
            return self.interface.group_state_to_canonical(self.group.name, errors)
        except InterfaceError as error:
            raise AdapterUnavailable(
                f"{controller} does not report every joint of "
                f"{self.group.name!r}: {error}"
            ) from error

    def send_effort(self, effort: Sequence[float]) -> None:
        """Publish authorized feedforward torque to the group's effort controller.

        The values must already have passed a gate: this is a force, and the
        adapter is not where forces are judged.
        """
        self._require_execute()
        if self.group.effort_controller is None:
            raise ValueError(
                f"group {self.group.name!r} declares no effort_controller, so "
                "torque cannot be published for it"
            )
        source = self.interface.group_command_to_source(self.group.name, effort)
        self._backend.publish_effort(
            self.group.effort_controller, list(source.values())
        )

    def watch_marker(self) -> None:
        """Start listening to the marker's drag stream.

        The feedback topic carries a pose only while a drag is in progress, so a
        servo loop subscribes once and reads whatever has arrived since, rather
        than asking for a pose it might be between.
        """
        tip = self._require_planning_group()
        self._backend.watch_marker(f"{MARKER_PREFIX}{tip}")

    def latest_marker_target(self) -> Pose | None:
        """Return the newest dragged pose, or None if none has arrived yet."""
        return self._backend.latest_marker()

    def stream_positions(self, positions: Sequence[float]) -> None:
        """Send one servo sample to the group's trajectory controller.

        Published on the controller's topic interface rather than as an action
        goal: a goal per sample would spend the whole period on the accept and
        result handshake. The controller replaces its active trajectory with
        each message, and the point is due on arrival — see
        ``STREAM_HORIZON_SEC`` for why it cannot be a period ahead.
        """
        self._require_execute()
        self._require_action(FOLLOW_JOINT_TRAJECTORY)
        source = self.interface.group_command_to_source(self.group.name, positions)
        self._backend.publish_trajectory_point(
            self.group.controller,
            list(source),
            list(source.values()),
            STREAM_HORIZON_SEC,
        )

    def pump(self, timeout_sec: float = 0.0) -> None:
        """Let subscriptions deliver. A servo loop must call this every cycle."""
        self._backend.pump(timeout_sec)

    def now_ns(self) -> int:
        """The node's clock, which is what a published command is stamped with.

        Not wall time and not a monotonic counter. ``normalize_track`` overlaps a
        command stream with a measured one by comparing their stamps, and the
        measured stamps come from the publisher's ROS clock — so ours has to be
        the same kind of clock or the two are on different epochs. Under
        ``use_sim_time`` this follows the simulation, which is the only reading
        that stays comparable there.
        """
        return self._backend.now_ns()

    def start_recording(self) -> None:
        """Begin keeping every joint state message, with its own stamp.

        Separate from :meth:`read_state`, which discards what arrived before it
        was called so a caller never reads a pose from before the motion it just
        commanded. That is right for reading a pose and fatal for recording a
        stream, so the two do not share a path.
        """
        self._backend.start_recording()
        self._recording = True

    def stop_recording(self) -> Recording:
        """Return what arrived, in this group's canonical order.

        A message that does not cover the whole group is counted rather than
        raised on: during bringup, or if a controller drops out mid-run, some
        messages legitimately do not carry every joint, and losing the rest of
        the run over it would be worse than recording the gap.
        """
        if not self._recording:
            raise AdapterUnavailable(
                "not recording; call start_recording() before stop_recording()"
            )
        self._recording = False
        samples = self._backend.stop_recording()
        stamps: list[int] = []
        rows: list[np.ndarray] = []
        incomplete = 0
        for stamp_ns, source in samples:
            try:
                rows.append(
                    self.interface.group_state_to_canonical(self.group.name, source)
                )
            except InterfaceError:
                incomplete += 1
                continue
            stamps.append(int(stamp_ns))
        if not rows:
            raise AdapterUnavailable(
                f"no {self.state_topic} covering group {self.group.name!r} was "
                f"recorded ({incomplete} message(s) arrived without it); is the "
                "bringup running, and is the loop calling pump()?"
            )
        return Recording(
            np.asarray(stamps, dtype=np.int64),
            np.vstack(rows),
            tuple(self.group.joints),
            incomplete=incomplete,
        )

    def read_marker_pose(self, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> Pose:
        """Return the pose of the RViz goal marker for this group's tip link.

        RViz's own Plan & Execute cannot drive this robot, so the marker is a
        way to *choose* a pose rather than to reach one. Reading it here lets
        an operator aim by dragging and commit through the same gate as every
        other command.
        """
        tip = self._require_planning_group()
        name = f"{MARKER_PREFIX}{tip}"
        pose = self._backend.marker_pose(name, timeout_sec)
        if pose is None:
            raise AdapterUnavailable(
                f"RViz is running but holds no marker named {name!r}; set the "
                f"MotionPlanning panel's planning group to "
                f"{self.group.moveit_group!r} so it publishes one"
            )
        return pose

    def solve_ik(
        self,
        pose: Pose,
        seed: Sequence[float],
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> np.ndarray:
        """Solve IK for *pose*, seeded from canonical *seed*, in canonical order."""
        tip = self._require_planning_group()
        source_seed = self.interface.group_command_to_source(self.group.name, seed)
        code, solution = self._backend.compute_ik(
            self.group.moveit_group, tip, pose, source_seed, timeout_sec
        )
        if code != MOVEIT_SUCCESS:
            raise IkFailed(
                f"no IK solution for {tip} in group {self.group.moveit_group!r} "
                f"(MoveIt error code {code})"
            )
        try:
            # MoveIt answers with the whole robot state, so the group is
            # extracted rather than assumed to be the entire answer.
            return self.interface.group_state_to_canonical(self.group.name, solution)
        except InterfaceError as error:
            raise IkFailed(f"IK solution does not cover the group: {error}") from error

    def send_trajectory(
        self, points: Sequence[Sequence[float]], period_sec: float
    ) -> None:
        """Send an authorized canonical trajectory to the group's controller."""
        self._require_execute()
        self._require_action(FOLLOW_JOINT_TRAJECTORY)
        if not points:
            raise ValueError("a trajectory needs at least one waypoint")
        source = [
            list(
                self.interface.group_command_to_source(self.group.name, point).values()
            )
            for point in points
        ]
        self._backend.follow_joint_trajectory(
            self.group.controller,
            self.interface.group_source_names(self.group.name),
            source,
            period_sec,
        )

    def send_gripper(self, position: float) -> None:
        """Send an authorized canonical position to the group's gripper action."""
        self._require_execute()
        self._require_action(PARALLEL_GRIPPER_COMMAND)
        source = self.interface.group_command_to_source(self.group.name, [position])
        joint, value = next(iter(source.items()))
        self._backend.gripper_command(self.group.controller, joint, value)

    def _require_execute(self) -> None:
        if not self.execute:
            raise SafetyError("publishing requires explicit --execute")

    def _require_planning_group(self) -> str:
        if self.group.moveit_group is None or self.group.tip_link is None:
            raise ValueError(
                f"group {self.group.name!r} has no planning group, so it has no "
                "end-effector pose; set it with direct joint values instead"
            )
        return self.group.tip_link

    def _require_action(self, action: str) -> None:
        if self.group.action != action:
            raise ValueError(
                f"group {self.group.name!r} is driven by {self.group.action}, "
                f"not {action}"
            )


#: Every ROS interface this backend needs, in one place because this is the
#: surface that differs between distributions. `ParallelGripperCommand` arrived
#: in control_msgs after Humble, so on an older distribution this list is what
#: fails and this list is what a port edits — not nine hundred lines of code.
REQUIRED_INTERFACES = (
    "builtin_interfaces/msg/Duration",
    "control_msgs/action/FollowJointTrajectory",
    "control_msgs/action/ParallelGripperCommand",
    "control_msgs/msg/JointTrajectoryControllerState",
    "geometry_msgs/msg/Pose",
    "moveit_msgs/srv/GetPositionIK",
    "sensor_msgs/msg/JointState",
    "trajectory_msgs/msg/JointTrajectory",
    "visualization_msgs/srv/GetInteractiveMarkers",
)

#: The trajectory controller's state topic, relative to the controller. Renamed
#: from `state` after Humble, and it is the only thing `pose gravity` measures —
#: point it at the wrong name and every gravity scale is unmeasurable rather
#: than wrong.
CONTROLLER_STATE_TOPIC = "controller_state"

#: Where a servo stream goes, and where feedforward torque goes.
CONTROLLER_STREAM_TOPIC = "joint_trajectory"
EFFORT_COMMAND_TOPIC = "commands"

#: How far ahead a streamed point is placed. Zero, so it is due the moment it
#: arrives, and deliberately not the stream's period.
#:
#: ``Trajectory::sample`` in ros2_controllers returns the state the trajectory
#: was installed with — not an interpolation toward the point — for any sample
#: taken before the first point comes due, when the controller runs
#: ``interpolation_method: none``. A servo stream reinstalls the trajectory
#: every period, so a point one period ahead puts every sample inside that
#: window: the reference never advances, and neither does the arm. Nothing is
#: logged and the controller reports its goals reached, so the only visible
#: symptom is a robot that does not move.
#:
#: Measured on the OpenArm: a commanded 0.06 rad excitation moved the reference
#: by 0.00001 rad, while the same joint on the same controller tracked an
#: action goal normally. At zero the first sample is already at the point, so
#: the controller adopts it at once — under either interpolation method.
#:
#: This is a horizon, not a rate limit. What bounds the step between
#: consecutive samples is the safety gate, which authorizes the whole track
#: against each joint's velocity limit before any of it is published.
STREAM_HORIZON_SEC = 0.0


class _RclpyBackend:
    """The real ROS backend. Every rclpy import is confined to this class."""

    def __init__(self, node_name: str, joint_topic: str = JOINT_STATES_TOPIC):
        try:
            import rclpy
            from rclpy.action import ActionClient
            from rclpy.qos import qos_profile_sensor_data
            from builtin_interfaces.msg import Duration
            from control_msgs.action import FollowJointTrajectory
            from control_msgs.action import ParallelGripperCommand
            from geometry_msgs.msg import Pose as PoseMsg
            from geometry_msgs.msg import PoseStamped, Quaternion, Point
            from control_msgs.msg import JointTrajectoryControllerState
            from moveit_msgs.srv import GetPositionFK, GetPositionIK
            from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64MultiArray, Header, String
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
            from visualization_msgs.msg import InteractiveMarkerFeedback
            from visualization_msgs.srv import GetInteractiveMarkers
        except ImportError as error:
            # Naming what was missing, because these are two different problems
            # with two different fixes: no rclpy means nothing is sourced, while
            # one absent interface on an otherwise working install means this
            # distribution does not have it. Saying "source a workspace" to
            # someone whose workspace is sourced sends them to the wrong place.
            missing = getattr(error, "name", None) or "a ROS interface"
            raise AdapterUnavailable(
                f"the ROS adapter could not import {missing}. If nothing is "
                "sourced, run `source ros_ws/install/setup.bash`. If it is, "
                "this ROS distribution does not ship that interface — the "
                "adapter's REQUIRED_INTERFACES lists everything it needs, and "
                f"{error}"
            ) from error

        self._rclpy = rclpy
        self._ActionClient = ActionClient
        self._sensor_qos = qos_profile_sensor_data
        self._Duration = Duration
        self._FollowJointTrajectory = FollowJointTrajectory
        self._ParallelGripperCommand = ParallelGripperCommand
        self._PoseMsg, self._PoseStamped = PoseMsg, PoseStamped
        self._Quaternion, self._Point = Quaternion, Point
        self._GetPositionFK, self._GetPositionIK = GetPositionFK, GetPositionIK
        self._GetInteractiveMarkers = GetInteractiveMarkers
        self._MarkerFeedback = InteractiveMarkerFeedback
        self._ControllerState = JointTrajectoryControllerState
        self._Float64MultiArray = Float64MultiArray
        self._String = String
        # robot_state_publisher latches the description, so a subscriber that
        # arrives after it was published still receives it.
        self._latching_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._JointState, self._Header = JointState, Header
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint

        # Only shut down the context this backend started, so an adapter used
        # from inside an existing rclpy application does not tear it down.
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._joint_topic = joint_topic
        self._joint_subscription = None
        self._latest: dict[str, float] | None = None
        # None until start_recording: the same callback serves read_state, and
        # a list here is what tells it to keep rather than only cache.
        self._recorded: list[tuple[int, dict[str, float]]] | None = None
        self._fk_client = None
        self._ik_client = None
        self._marker_client = None
        self._effort_publishers: dict[str, Any] = {}
        self._stream_publishers: dict[str, Any] = {}
        self._marker_subscription = None
        self._marker_name: str | None = None
        self._marker_target: Pose | None = None

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_context:
            self._rclpy.shutdown()

    def joint_states(self, timeout_sec: float) -> dict[str, float]:
        if self._joint_subscription is None:
            self._joint_subscription = self._node.create_subscription(
                self._JointState, self._joint_topic, self._record, self._sensor_qos
            )
        # Discard anything received earlier, so a caller never reads a pose
        # from before the motion it just commanded.
        self._latest = None
        deadline = time.monotonic() + timeout_sec
        while self._latest is None and time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
        if self._latest is None:
            raise AdapterUnavailable(
                f"no {self._joint_topic} within {timeout_sec} s; is the robot "
                "bringup running? (ros_ws/pose_bringup.sh)"
            )
        return dict(self._latest)

    def _record(self, message: Any) -> None:
        self._latest = dict(zip(message.name, message.position))
        if self._recorded is not None:
            # header.stamp, not the time this callback ran: it is when the
            # hardware read was taken, which is the quantity the controller
            # delay is measured against. The callback time carries the queueing
            # this recording exists to characterise.
            stamp = message.header.stamp
            self._recorded.append(
                (
                    int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
                    dict(zip(message.name, message.position)),
                )
            )

    def now_ns(self) -> int:
        """The node clock, so a command's stamp is comparable to header.stamp."""
        return int(self._node.get_clock().now().nanoseconds)

    def start_recording(self) -> None:
        if self._joint_subscription is None:
            self._joint_subscription = self._node.create_subscription(
                self._JointState, self._joint_topic, self._record, self._sensor_qos
            )
        self._recorded = []

    def stop_recording(self) -> list[tuple[int, dict[str, float]]]:
        recorded, self._recorded = self._recorded, None
        return recorded or []

    def compute_fk(
        self,
        link: str,
        seed: Mapping[str, float],
        frame_id: str,
        timeout_sec: float,
    ) -> tuple[int, Pose | None]:
        if self._fk_client is None:
            self._fk_client = self._node.create_client(
                self._GetPositionFK, "/compute_fk"
            )
        request = self._GetPositionFK.Request()
        request.header.frame_id = frame_id
        request.fk_link_names = [link]
        request.robot_state.joint_state = self._joint_state(seed, frame_id)
        request.robot_state.is_diff = True
        response = self._call(self._fk_client, request, timeout_sec, "/compute_fk")
        if not response.pose_stamped:
            return (response.error_code.val, None)
        stamped = response.pose_stamped[0]
        position, orientation = stamped.pose.position, stamped.pose.orientation
        return (
            response.error_code.val,
            Pose(
                (position.x, position.y, position.z),
                (orientation.x, orientation.y, orientation.z, orientation.w),
                stamped.header.frame_id or frame_id,
            ),
        )

    def compute_ik(
        self,
        group: str,
        link: str,
        pose: Pose,
        seed: Mapping[str, float],
        timeout_sec: float,
    ) -> tuple[int, dict[str, float]]:
        if self._ik_client is None:
            self._ik_client = self._node.create_client(
                self._GetPositionIK, "/compute_ik"
            )
        request = self._GetPositionIK.Request()
        request.ik_request.group_name = group
        request.ik_request.ik_link_name = link
        # The seed covers only the planning group, so it is a diff over the
        # planning scene's current state rather than a complete robot state.
        request.ik_request.robot_state.joint_state = self._joint_state(
            seed, pose.frame_id
        )
        request.ik_request.robot_state.is_diff = True
        request.ik_request.pose_stamped = self._PoseStamped(
            header=self._Header(frame_id=pose.frame_id),
            pose=self._PoseMsg(
                position=self._Point(
                    x=pose.position[0], y=pose.position[1], z=pose.position[2]
                ),
                orientation=self._Quaternion(
                    x=pose.orientation[0],
                    y=pose.orientation[1],
                    z=pose.orientation[2],
                    w=pose.orientation[3],
                ),
            ),
        )
        request.ik_request.timeout = self._duration(min(timeout_sec, 5.0))
        request.ik_request.avoid_collisions = True
        response = self._call(self._ik_client, request, timeout_sec, "/compute_ik")
        solution = dict(
            zip(response.solution.joint_state.name, response.solution.joint_state.position)
        )
        return (response.error_code.val, solution)

    def robot_description(self, timeout_sec: float) -> str:
        # Published transient-local, so a late subscriber still receives it.
        latest: list[str] = []
        subscription = self._node.create_subscription(
            self._String,
            "/robot_description",
            lambda message: latest.append(message.data),
            self._latching_qos,
        )
        try:
            deadline = time.monotonic() + timeout_sec
            while not latest and time.monotonic() < deadline:
                self._rclpy.spin_once(self._node, timeout_sec=0.05)
        finally:
            self._node.destroy_subscription(subscription)
        if not latest:
            raise AdapterUnavailable(
                f"no /robot_description within {timeout_sec} s; is "
                "robot_state_publisher running?"
            )
        return latest[-1]

    def tracking_error(self, controller: str, timeout_sec: float) -> dict[str, float]:
        topic = f"/{controller}/{CONTROLLER_STATE_TOPIC}"
        latest: list[dict[str, float]] = []
        subscription = self._node.create_subscription(
            self._ControllerState,
            topic,
            lambda message: latest.append(
                dict(zip(message.joint_names, message.error.positions))
            ),
            10,
        )
        try:
            deadline = time.monotonic() + timeout_sec
            while not latest and time.monotonic() < deadline:
                self._rclpy.spin_once(self._node, timeout_sec=0.05)
        finally:
            self._node.destroy_subscription(subscription)
        if not latest:
            raise AdapterUnavailable(
                f"no {topic} within {timeout_sec} s; is {controller} active?"
            )
        return latest[-1]

    def publish_effort(self, controller: str, values: Sequence[float]) -> None:
        # One publisher per controller, kept for the process's life: gravity
        # compensation republishes at the controller rate, and a fresh publisher
        # each time would drop commands to discovery every cycle.
        publisher = self._effort_publishers.get(controller)
        if publisher is None:
            publisher = self._node.create_publisher(
                self._Float64MultiArray, f"/{controller}/{EFFORT_COMMAND_TOPIC}", 10
            )
            self._effort_publishers[controller] = publisher
            # A brand new publisher has no matched subscriber yet, and a command
            # sent into that gap is simply lost.
            deadline = time.monotonic() + DEFAULT_TIMEOUT_SEC
            while (
                publisher.get_subscription_count() == 0
                and time.monotonic() < deadline
            ):
                self._rclpy.spin_once(self._node, timeout_sec=0.05)
            if publisher.get_subscription_count() == 0:
                raise AdapterUnavailable(
                    f"nothing is subscribed to /{controller}/commands; load the "
                    "effort controllers with ros_ws/load_effort_controllers.sh"
                )
        publisher.publish(
            self._Float64MultiArray(data=[float(value) for value in values])
        )

    def watch_marker(self, name: str) -> None:
        self._marker_name = name
        if self._marker_subscription is not None:
            return
        self._marker_subscription = self._node.create_subscription(
            self._MarkerFeedback,
            MARKER_FEEDBACK_TOPIC,
            self._record_marker,
            10,
        )

    def _record_marker(self, message: Any) -> None:
        if message.marker_name != self._marker_name:
            return
        position, orientation = message.pose.position, message.pose.orientation
        self._marker_target = Pose(
            (position.x, position.y, position.z),
            (orientation.x, orientation.y, orientation.z, orientation.w),
            message.header.frame_id,
        )

    def latest_marker(self) -> Pose | None:
        return self._marker_target

    def pump(self, timeout_sec: float) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def publish_trajectory_point(
        self,
        controller: str,
        joint_names: Sequence[str],
        positions: Sequence[float],
        horizon_sec: float,
    ) -> None:
        topic = f"/{controller}/{CONTROLLER_STREAM_TOPIC}"
        publisher = self._stream_publishers.get(topic)
        if publisher is None:
            publisher = self._node.create_publisher(self._JointTrajectory, topic, 10)
            self._stream_publishers[topic] = publisher
            deadline = time.monotonic() + DEFAULT_TIMEOUT_SEC
            while (
                publisher.get_subscription_count() == 0
                and time.monotonic() < deadline
            ):
                self._rclpy.spin_once(self._node, timeout_sec=0.05)
            if publisher.get_subscription_count() == 0:
                raise AdapterUnavailable(
                    f"nothing is subscribed to {topic}; is {controller} active?"
                )
        trajectory = self._JointTrajectory()
        trajectory.joint_names = list(joint_names)
        point = self._JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.time_from_start = self._duration(horizon_sec)
        trajectory.points.append(point)
        publisher.publish(trajectory)

    def marker_pose(self, name: str, timeout_sec: float) -> Pose | None:
        if self._marker_client is None:
            self._marker_client = self._node.create_client(
                self._GetInteractiveMarkers, MARKER_SERVICE
            )
        if not self._marker_client.wait_for_service(timeout_sec=timeout_sec):
            raise AdapterUnavailable(
                f"{MARKER_SERVICE} is not available; the marker lives in RViz, "
                "so start the bringup with RViz (ros_ws/pose_bringup.sh)"
            )
        # Asking the server beats listening for feedback: feedback is published
        # only while a drag is in progress, so a poll would have to be racing
        # the operator's mouse. The server holds the pose after the drag ends.
        response = self._resolve(
            self._marker_client.call_async(self._GetInteractiveMarkers.Request()),
            timeout_sec,
            MARKER_SERVICE,
        )
        for marker in response.markers:
            if marker.name != name:
                continue
            position, orientation = marker.pose.position, marker.pose.orientation
            return Pose(
                (position.x, position.y, position.z),
                (orientation.x, orientation.y, orientation.z, orientation.w),
                marker.header.frame_id,
            )
        return None

    def follow_joint_trajectory(
        self,
        controller: str,
        joint_names: Sequence[str],
        points: Sequence[Sequence[float]],
        period_sec: float,
    ) -> None:
        goal = self._FollowJointTrajectory.Goal()
        goal.trajectory = self._JointTrajectory()
        goal.trajectory.joint_names = list(joint_names)
        for index, point in enumerate(points):
            waypoint = self._JointTrajectoryPoint()
            waypoint.positions = [float(value) for value in point]
            # The first waypoint sits one period ahead: a point at t=0 asks the
            # controller to already be there.
            waypoint.time_from_start = self._duration((index + 1) * period_sec)
            goal.trajectory.points.append(waypoint)
        self._send_goal(
            self._FollowJointTrajectory,
            f"/{controller}/follow_joint_trajectory",
            goal,
            result_timeout_sec=(len(points) + 1) * period_sec + DEFAULT_TIMEOUT_SEC,
            failure=trajectory_failure,
        )

    def gripper_command(self, controller: str, joint: str, position: float) -> None:
        # ParallelGripperCommand carries a JointState, not a bare position, so
        # the goal must name the joint it is driving.
        goal = self._ParallelGripperCommand.Goal()
        goal.command = self._joint_state({joint: position}, "")
        self._send_goal(
            self._ParallelGripperCommand,
            f"/{controller}/gripper_cmd",
            goal,
            result_timeout_sec=DEFAULT_TIMEOUT_SEC,
            failure=gripper_failure,
        )

    def _send_goal(
        self,
        action_type: Any,
        action_name: str,
        goal: Any,
        result_timeout_sec: float,
        failure: Any,
    ) -> None:
        client = self._ActionClient(self._node, action_type, action_name)
        try:
            if not client.wait_for_server(timeout_sec=DEFAULT_TIMEOUT_SEC):
                raise AdapterUnavailable(
                    f"no action server at {action_name}; is the controller "
                    "spawned and active?"
                )
            handle = self._resolve(
                client.send_goal_async(goal), DEFAULT_TIMEOUT_SEC, action_name
            )
            if not handle.accepted:
                raise AdapterUnavailable(f"{action_name} rejected the goal")
            result = self._resolve(
                handle.get_result_async(), result_timeout_sec, action_name
            )
            reason = failure(result.result)
            if reason is not None:
                raise AdapterUnavailable(f"{action_name} failed: {reason}")
        finally:
            client.destroy()

    def _call(
        self, client: Any, request: Any, timeout_sec: float, name: str
    ) -> Any:
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise AdapterUnavailable(
                f"{name} is not available; is move_group running?"
            )
        return self._resolve(client.call_async(request), timeout_sec, name)

    def _resolve(self, future: Any, timeout_sec: float, name: str) -> Any:
        # A synchronous call() from a node nobody spins deadlocks, because the
        # response can only arrive while the executor is running.
        self._rclpy.spin_until_future_complete(
            self._node, future, timeout_sec=timeout_sec
        )
        if not future.done():
            raise AdapterUnavailable(f"{name} did not answer within {timeout_sec} s")
        return future.result()

    def _joint_state(self, values: Mapping[str, float], frame_id: str) -> Any:
        return self._JointState(
            header=self._Header(frame_id=frame_id),
            name=list(values),
            position=[float(value) for value in values.values()],
        )

    def _duration(self, seconds: float) -> Any:
        whole = int(seconds)
        return self._Duration(
            sec=whole, nanosec=int(round((seconds - whole) * 1_000_000_000))
        )
