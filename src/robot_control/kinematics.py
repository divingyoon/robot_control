"""Forward kinematics, Jacobian, and gravity torque for a serial chain.

Numpy only, and deliberately so. Gravity compensation needs a torque at the
controller rate, and the marker servo loop needs a Jacobian at the same rate;
both are cheap to compute directly and expensive to obtain through a ROS
service. Keeping the maths here also means it can be tested against closed
forms offline, without a robot or a running graph.

`pinocchio` is not packaged for this platform and `kdl_parser_py` is absent, so
PyKDL would need the URDF walked by hand anyway and would still not supply a
Jacobian reusable for inverse kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import xml.etree.ElementTree as ElementTree

import numpy as np

#: Standard gravity, along -z of the chain's base frame.
GRAVITY = np.array([0.0, 0.0, -9.80665])
#: Damping for the least-squares inverse used by :meth:`Chain.delta_q`. Small
#: enough not to blunt ordinary motion, large enough to bound a step taken at a
#: singularity.
DEFAULT_DAMPING = 0.05


class KinematicsError(ValueError):
    """The chain does not describe what was asked of it."""


@dataclass(frozen=True)
class Revolute:
    """One actuated joint, and the fixed transform from the previous one.

    ``origin`` and ``rotation`` place this joint in its parent's frame; any
    fixed joints between the two are folded into them, because a fixed joint is
    a transform rather than a degree of freedom.
    """

    name: str
    axis: np.ndarray
    origin: np.ndarray
    rotation: np.ndarray
    child: str


@dataclass(frozen=True)
class Link:
    """A link's mass and centre of mass, in the link's own frame."""

    name: str
    mass: float
    com: np.ndarray


class Chain:
    """A serial chain of revolute joints, with a fixed transform to the tip."""

    def __init__(
        self,
        joints: Sequence[Revolute],
        links: Sequence[Link],
        tip: np.ndarray | None = None,
    ):
        if len(links) != len(joints):
            raise KinematicsError(
                f"each joint needs the link it drives: {len(joints)} joints, "
                f"{len(links)} links"
            )
        self.joints = tuple(joints)
        self.links = tuple(links)
        # Transform from the last joint's frame to the tip frame, identity when
        # the tip is the last driven link itself.
        self.tip = np.eye(4) if tip is None else np.asarray(tip, dtype=float)

    def __len__(self) -> int:
        return len(self.joints)

    def frames(self, q: Sequence[float]) -> list[np.ndarray]:
        """Return the world transform of each joint frame, after its rotation."""
        q = self._check(q)
        frames = []
        current = np.eye(4)
        for joint, angle in zip(self.joints, q):
            step = np.eye(4)
            step[:3, :3] = joint.rotation @ _rotation(joint.axis, angle)
            step[:3, 3] = joint.origin
            current = current @ step
            frames.append(current)
        return frames

    def pose(self, q: Sequence[float]) -> np.ndarray:
        """Return the 4x4 world transform of the tip."""
        return self.frames(q)[-1] @ self.tip

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """Return the 6xN geometric Jacobian of the tip, in the base frame.

        Rows 0-2 are linear, rows 3-5 angular. For a revolute joint the linear
        column is ``z x (p_tip - p_joint)`` and the angular column is ``z``.
        """
        frames = self.frames(q)
        tip = frames[-1] @ self.tip
        jacobian = np.zeros((6, len(self.joints)))
        for column, (joint, frame) in enumerate(zip(self.joints, frames)):
            axis = frame[:3, :3] @ joint.axis
            jacobian[:3, column] = np.cross(axis, tip[:3, 3] - frame[:3, 3])
            jacobian[3:, column] = axis
        return jacobian

    def delta_q(
        self,
        q: Sequence[float],
        twist: Sequence[float],
        damping: float = DEFAULT_DAMPING,
    ) -> np.ndarray:
        """Return the joint step that moves the tip by *twist*.

        Damped least squares rather than a pseudo-inverse. At a singularity the
        Jacobian loses rank and a pseudo-inverse asks for an unbounded joint
        velocity to achieve a motion the arm cannot make; the damping trades a
        little accuracy for a step that stays finite. Servoing runs wherever the
        operator drags, singularities included.
        """
        twist = np.asarray(twist, dtype=float)
        if twist.shape != (6,):
            raise KinematicsError(
                f"a twist needs six values, three linear and three angular, "
                f"got {twist.size}"
            )
        jacobian = self.jacobian(q)
        square = jacobian @ jacobian.T + (damping**2) * np.eye(6)
        return jacobian.T @ np.linalg.solve(square, twist)

    def gravity_torque(self, q: Sequence[float]) -> np.ndarray:
        """Return the joint torque that holds the chain against gravity.

        Only masses and centres of mass enter: gravity torque is the sum over
        links of the load's weight projected onto the velocity each joint would
        give that link's centre of mass. Inertia tensors describe acceleration,
        which a static hold does not involve.
        """
        frames = self.frames(q)
        torque = np.zeros(len(self.joints))
        centres = [
            frame[:3, :3] @ link.com + frame[:3, 3]
            for frame, link in zip(frames, self.links)
        ]
        for index, (joint, frame) in enumerate(zip(self.joints, frames)):
            axis = frame[:3, :3] @ joint.axis
            # Only links at or beyond this joint load it.
            for link, centre in zip(self.links[index:], centres[index:]):
                lever = np.cross(axis, centre - frame[:3, 3])
                torque[index] -= link.mass * float(GRAVITY @ lever)
        return torque

    def _check(self, q: Sequence[float]) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.joints),):
            raise KinematicsError(
                f"this chain has {len(self.joints)} joints, got {q.size} values"
            )
        return q


def twist_between(current: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Return the six-vector taking pose *current* to pose *goal*.

    Linear part first, then the rotation as an axis-angle vector, which is what
    a Jacobian's angular rows are expressed in.
    """
    current = np.asarray(current, dtype=float)
    goal = np.asarray(goal, dtype=float)
    twist = np.zeros(6)
    twist[:3] = goal[:3, 3] - current[:3, 3]
    twist[3:] = _log_rotation(goal[:3, :3] @ current[:3, :3].T)
    return twist


def _log_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle vector of a rotation matrix."""
    # Clipped because a matrix assembled from measured joint values is only
    # orthonormal to floating-point precision, and acos would raise just outside.
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3)
    if abs(angle - np.pi) < 1e-6:
        # At half a turn the skew part vanishes; recover the axis from the
        # symmetric part instead, where it survives.
        axis = np.sqrt(np.maximum(np.diag(rotation) + 1.0, 0.0) / 2.0)
        return angle * axis / np.linalg.norm(axis)
    skew = (rotation - rotation.T) / (2.0 * np.sin(angle))
    return angle * np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def chain_from_urdf(urdf: str, joint_names: Iterable[str], tip_link: str) -> Chain:
    """Build a chain for *joint_names*, ending at *tip_link*.

    The named joints must form a path in the URDF. Fixed joints along the way,
    including the usual fixed hop from the last driven link out to a tool centre
    point, are folded into the neighbouring transforms.
    """
    root = ElementTree.fromstring(urdf)
    # Direct children only. A <ros2_control> block carries <joint> elements
    # under the *same names* as the kinematic joints, with no parent or child,
    # so a recursive search lets those shadow the real chain.
    joints = {element.get("name"): element for element in root.findall("joint")}
    links = {element.get("name"): element for element in root.findall("link")}
    wanted = list(joint_names)
    missing = [name for name in wanted if name not in joints]
    if missing:
        raise KinematicsError(f"URDF has no joint named {missing[0]!r}")
    if tip_link not in links:
        raise KinematicsError(f"URDF has no link named {tip_link!r}")

    # Walk down from the tip so the path is unambiguous: a link has one parent
    # joint, while a joint may have many children.
    parent_of: dict[str, ElementTree.Element] = {}
    for element in joints.values():
        child = element.find("child")
        if child is not None:
            parent_of[child.get("link")] = element

    path: list[ElementTree.Element] = []
    link = tip_link
    while link in parent_of:
        element = parent_of[link]
        path.append(element)
        link = element.find("parent").get("link")
    path.reverse()

    on_path = [element.get("name") for element in path]
    unreachable = [name for name in wanted if name not in on_path]
    if unreachable:
        raise KinematicsError(
            f"joint {unreachable[0]!r} is not between the URDF root and "
            f"{tip_link!r}"
        )

    # Fixed joints by parent link, so a driven link's rigidly attached
    # descendants can be lumped into it. Without this the mass beyond the last
    # joint — the hand and the tool centre point, several hundred grams on the
    # OpenArm — loads no joint at all, and gravity torque comes out low.
    fixed_children: dict[str, list[ElementTree.Element]] = {}
    for element in joints.values():
        if element.get("type") != "fixed":
            continue
        parent = element.find("parent")
        if parent is not None:
            fixed_children.setdefault(parent.get("link"), []).append(element)

    chain_joints: list[Revolute] = []
    chain_links: list[Link] = []
    pending = np.eye(4)
    for element in path:
        origin, rotation = _origin(element)
        if element.get("name") not in wanted:
            # A fixed or unwanted joint contributes a transform only.
            pending = pending @ _homogeneous(rotation, origin)
            continue
        combined = pending @ _homogeneous(rotation, origin)
        child = element.find("child").get("link")
        chain_joints.append(
            Revolute(
                name=element.get("name"),
                axis=_axis(element),
                origin=combined[:3, 3],
                rotation=combined[:3, :3],
                child=child,
            )
        )
        chain_links.append(_lumped(child, links, fixed_children))
        pending = np.eye(4)

    return Chain(chain_joints, chain_links, tip=pending)


def _lumped(
    name: str,
    links: dict[str, ElementTree.Element],
    fixed_children: dict[str, list[ElementTree.Element]],
    transform: np.ndarray | None = None,
) -> Link:
    """Return *name*'s mass and centre of mass with fixed descendants folded in.

    Anything bolted on through fixed joints moves as one body with this link, so
    for a static hold it is one mass at the combined centre. Descendants behind a
    movable joint are left out: their contribution depends on that joint's own
    value, which this chain does not carry.
    """
    transform = np.eye(4) if transform is None else transform
    element = links.get(name)
    own = _link(element) if element is not None else Link(name, 0.0, np.zeros(3))
    mass = own.mass
    moment = own.mass * (transform[:3, :3] @ own.com + transform[:3, 3])
    for joint in fixed_children.get(name, ()):
        origin, rotation = _origin(joint)
        child = _lumped(
            joint.find("child").get("link"),
            links,
            fixed_children,
            transform @ _homogeneous(rotation, origin),
        )
        mass += child.mass
        # The recursive call already returned its centre in this link's frame.
        moment += child.mass * child.com
    return Link(name, mass, moment / mass if mass > 0.0 else np.zeros(3))


def with_payload(chain: Chain, mass: float, com: Sequence[float]) -> Chain:
    """Return *chain* with an extra load folded into its last link.

    `_lumped` deliberately leaves out anything behind a movable joint: where it
    sits depends on that joint's value, which an arm chain does not carry. For a
    multi-fingered hand that exclusion is most of the hand — the Tesollo DG-5F
    keeps 0.85 kg of adapter/base/palm and drops 0.835 kg of fingers, so the
    wrist gets under a third of the gravity torque it actually carries (2026-08-31
    measurement; the 2026-07-29 calibration recorded the same shortfall as
    "j6 about 3.4x under").

    The caller knows the finger angles, so the caller can compute that missing
    mass and its centre and pass them here. *com* is in the last link's frame,
    the same frame `Link.com` uses.

    The chain is not modified; a new one is returned.
    """
    if mass < 0.0:
        raise KinematicsError(f"payload mass cannot be negative, got {mass}")
    com = np.asarray(com, dtype=float)
    if com.shape != (3,):
        raise KinematicsError(f"a payload centre needs three values, got {com.size}")
    if mass == 0.0:
        return chain
    last = chain.links[-1]
    total = last.mass + mass
    merged = (last.mass * last.com + mass * com) / total
    links = list(chain.links[:-1]) + [Link(last.name, total, merged)]
    return Chain(chain.joints, links, chain.tip)


def _link(element: ElementTree.Element) -> Link:
    inertial = element.find("inertial")
    if inertial is None:
        return Link(element.get("name"), 0.0, np.zeros(3))
    mass = inertial.find("mass")
    origin, _rotation_matrix = _origin(inertial)
    return Link(
        element.get("name"),
        0.0 if mass is None else float(mass.get("value", 0.0)),
        origin,
    )


def _origin(element: ElementTree.Element) -> tuple[np.ndarray, np.ndarray]:
    origin = element.find("origin")
    if origin is None:
        return np.zeros(3), np.eye(3)
    xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
    return xyz, _rpy_matrix(*rpy)


def _axis(element: ElementTree.Element) -> np.ndarray:
    axis = element.find("axis")
    if axis is None:
        return np.array([1.0, 0.0, 0.0])
    values = np.array([float(v) for v in axis.get("xyz", "1 0 0").split()])
    norm = np.linalg.norm(values)
    if norm == 0.0:
        raise KinematicsError(f"joint {element.get('name')!r} has a zero axis")
    return values / norm


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy: fixed-axis roll about x, then pitch about y, then yaw about z."""
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def _rotation_x(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])


def _rotation_y(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])


def _rotation_z(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about an arbitrary unit axis."""
    axis = np.asarray(axis, dtype=float)
    cos, sin = np.cos(angle), np.sin(angle)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + sin * cross + (1.0 - cos) * (cross @ cross)
