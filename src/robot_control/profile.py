from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


#: The distributions a profile may declare an endpoint for. A spelling list for
#: the validator, not a branch in the code: nothing here behaves differently per
#: distro, and `tests/test_distro_neutrality.py` keeps it that way.
KNOWN_DISTROS = ("humble", "jazzy")


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Joint:
    canonical: str
    source: str
    sign: int
    unit: str
    lower: float
    upper: float
    velocity: float
    effort: float


# How a controller accepts a goal. The action cannot be inferred from the
# controller name: the OpenArm grippers run
# parallel_gripper_action_controller, whose server takes a
# control_msgs/ParallelGripperCommand carrying a sensor_msgs/JointState, and
# shares neither its goal nor its result fields with a trajectory controller.
FOLLOW_JOINT_TRAJECTORY = "follow_joint_trajectory"
PARALLEL_GRIPPER_COMMAND = "parallel_gripper_command"
_ACTIONS = (FOLLOW_JOINT_TRAJECTORY, PARALLEL_GRIPPER_COMMAND)


@dataclass(frozen=True)
class Group:
    name: str
    joints: tuple[str, ...]
    controller: str | None = None
    moveit_group: str | None = None
    action: str | None = None
    tip_link: str | None = None
    # A second controller claiming the same joints' effort interfaces, used to
    # publish gravity feedforward without the trajectory controller giving up
    # position. Named rather than derived: nothing in the trajectory
    # controller's name says what an effort controller beside it is called.
    effort_controller: str | None = None
    # What hdgp's training env calls this same set of joints, in its
    # `actuators={...}` dict. Named rather than derived for the same reason
    # `moveit_group` is: the two namespaces were written independently, and
    # hdgp's `get_actuator_params` answers an unrecognised name with the env's
    # own default instead of an error. A guessed name trains silently wrong.
    hdgp_group: str | None = None
    # Where this group's chain ends in the *asset* URDF, which names joints
    # canonically (r_aj_1...) and has no `tip_link` frame. Named rather than
    # derived: the asset generator and the bringup description chose their tool
    # frames independently.
    asset_tip_link: str | None = None
    # Where this group's joints are published, when that is not the robot-wide
    # `/joint_states`. A hand driven by its own controller_manager runs under a
    # namespace of its own and publishes there instead — not as well as, so a
    # reader on the default topic never sees the group at all.
    state_topic: str | None = None

    @property
    def executable(self) -> bool:
        return self.controller is not None

    @property
    def compensable(self) -> bool:
        """Whether feedforward torque can be published for this group."""
        return self.effort_controller is not None


@dataclass(frozen=True)
class RosEndpoint:
    command_topic: str
    state_topic: str
    controller: str
    command_rate_hz: float


@dataclass(frozen=True)
class RobotProfile:
    name: str
    components: tuple[str, ...]
    asset_id: str
    manifest_path: Path
    manifest_sha256: str
    joints: tuple[Joint, ...]
    groups: dict[str, Group]
    ros: dict[str, RosEndpoint]
    # The canonical-named URDF of the asset itself, whose masses include the
    # mounted hand — the gravity model the bringup description cannot provide.
    asset_urdf_path: Path | None = None

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(j.canonical for j in self.joints)

    def endpoint(self) -> RosEndpoint:
        """The ROS endpoint for the distribution this branch declares.

        Callers ask for "the endpoint" rather than for a named one, so the same
        source file works on either branch and the answer comes from
        ``.rosdistro`` — the one place that legitimately differs.
        """
        from .layout import declared_distro

        distro = declared_distro()
        if distro not in self.ros:
            raise ProfileError(
                f"this branch declares {distro!r}, which profile {self.name!r} "
                f"has no ROS endpoint for; it has {sorted(self.ros)}"
            )
        return self.ros[distro]

    def executable_groups(self) -> dict[str, Group]:
        """Return only the groups a controller can actually be commanded on."""
        return {
            name: group for name, group in self.groups.items() if group.executable
        }


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfileError(f"{label} must be an object")
    return value


def _resolve_manifest(profile_path: Path, value: str) -> Path:
    configured = Path(value)
    if configured.is_absolute():
        return configured

    direct = (profile_path.parent / configured).resolve()
    if direct.is_file():
        return direct

    parts = configured.parts
    if "hdgp" not in parts:
        return direct
    hdgp_relative = Path(*parts[parts.index("hdgp") + 1 :])

    explicit_root = os.environ.get("HDGP_ROOT")
    if explicit_root:
        return (Path(explicit_root).expanduser() / hdgp_relative).resolve()

    for ancestor in profile_path.parents:
        candidate = ancestor / "hdgp" / hdgp_relative
        if candidate.is_file():
            return candidate.resolve()
    return direct


def _group(name: str, body: dict[str, Any]) -> Group:
    controller = body.get("controller")
    moveit_group = body.get("moveit_group")
    action = body.get("action")
    tip_link = body.get("tip_link")
    effort_controller = body.get("effort_controller")
    hdgp_group = body.get("hdgp_group")
    asset_tip_link = body.get("asset_tip_link")
    state_topic = body.get("state_topic")
    if moveit_group is None and tip_link is not None:
        # The tip link is only ever used as the IK frame of a planning group.
        raise ProfileError(f"group {name} declares a tip_link without a moveit_group")
    if controller is None:
        # A planning group or an action without a controller names no endpoint,
        # so it would silently never execute.
        if moveit_group is not None:
            raise ProfileError(f"group {name} declares a moveit_group without a controller")
        if action is not None:
            raise ProfileError(f"group {name} declares an action without a controller")
        if effort_controller is not None:
            raise ProfileError(
                f"group {name} declares an effort_controller without a controller"
            )
    else:
        action = FOLLOW_JOINT_TRAJECTORY if action is None else str(action)
        if action not in _ACTIONS:
            raise ProfileError(f"group {name} declares an unsupported action: {action}")
    return Group(
        name=name,
        joints=tuple(body["joints"]),
        controller=None if controller is None else str(controller),
        moveit_group=None if moveit_group is None else str(moveit_group),
        action=action,
        tip_link=None if tip_link is None else str(tip_link),
        effort_controller=(
            None if effort_controller is None else str(effort_controller)
        ),
        hdgp_group=None if hdgp_group is None else str(hdgp_group),
        asset_tip_link=None if asset_tip_link is None else str(asset_tip_link),
        state_topic=None if state_topic is None else str(state_topic),
    )


def load_profile(path: str | Path) -> RobotProfile:
    path = Path(path).resolve()
    raw = _mapping(yaml.safe_load(path.read_text()), "profile")
    asset = _mapping(raw.get("asset"), "asset")
    manifest = _resolve_manifest(path, str(asset["manifest"]))
    if not manifest.is_file():
        raise ProfileError(f"asset manifest not found: {manifest}")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    expected = str(asset.get("manifest_sha256", ""))
    if digest != expected:
        raise ProfileError(f"manifest hash mismatch: expected {expected}, got {digest}")

    asset_urdf: Path | None = None
    if asset.get("urdf") is not None:
        # Resolved the same way as the manifest: both live in the asset's own
        # directory, wherever the hdgp checkout put it.
        asset_urdf = _resolve_manifest(path, str(asset["urdf"]))
        if not asset_urdf.is_file():
            raise ProfileError(f"asset urdf not found: {asset_urdf}")

    joints = tuple(Joint(**item) for item in raw.get("joints", []))
    names = tuple(j.canonical for j in joints)
    if not joints or len(set(names)) != len(names):
        raise ProfileError("joints must be non-empty and canonical names unique")
    for joint in joints:
        if joint.sign not in (-1, 1) or joint.unit != "rad":
            raise ProfileError(f"invalid normalization for joint {joint.canonical}")
        if not joint.lower < joint.upper or joint.velocity <= 0 or joint.effort <= 0:
            raise ProfileError(f"invalid safety limits for joint {joint.canonical}")

    manifest_raw = _mapping(yaml.safe_load(manifest.read_text()), "manifest")
    manifest_joints = set(manifest_raw.get("control_joint_order", []))
    missing_manifest = set(names) - manifest_joints
    if missing_manifest:
        raise ProfileError(f"joints absent from asset manifest: {sorted(missing_manifest)}")

    groups = {
        name: _group(name, _mapping(body, f"group {name}"))
        for name, body in _mapping(raw.get("groups"), "groups").items()
    }
    counts = {name: 0 for name in names}
    unknown: set[str] = set()
    for group in groups.values():
        for name in group.joints:
            if name in counts:
                counts[name] += 1
            else:
                unknown.add(name)
    if unknown or any(count != 1 for count in counts.values()):
        raise ProfileError(
            "every canonical joint must belong to exactly one actuator group"
            f"; unknown={sorted(unknown)}, counts={counts}"
        )

    ros = {
        distro: RosEndpoint(**_mapping(body, f"ros.{distro}"))
        for distro, body in _mapping(raw.get("ros"), "ros").items()
    }
    for distro, endpoint in ros.items():
        if distro not in KNOWN_DISTROS or endpoint.command_rate_hz <= 0:
            raise ProfileError(f"invalid ROS endpoint: {distro}")
    return RobotProfile(
        name=str(raw["name"]),
        components=tuple(raw["components"]),
        asset_id=str(asset["id"]),
        manifest_path=manifest,
        manifest_sha256=digest,
        joints=joints,
        groups=groups,
        ros=ros,
        asset_urdf_path=asset_urdf,
    )


def load_builtin_profile(name: str) -> RobotProfile:
    resource = files("robot_control").joinpath("profiles", f"{name}.yaml")
    if not resource.is_file():
        raise ProfileError(f"unknown profile: {name}")
    return load_profile(Path(str(resource)))
