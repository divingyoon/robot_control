"""`with_payload` — the hand mass a chain's own model leaves out.

A chain built from URDF drops every link behind a movable joint, so a
multi-fingered hand contributes only its palm. These tests pin the arithmetic
that lets a caller add the rest back, and the gravity torque it changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_control.kinematics import (
    Chain,
    KinematicsError,
    Link,
    Revolute,
    with_payload,
)


def _one_joint_chain(mass: float = 1.0, com=(0.0, 0.0, 0.1)) -> Chain:
    """A single revolute joint about x, with one link at *com*."""
    joint = Revolute(
        name="j1",
        axis=np.array([1.0, 0.0, 0.0]),
        origin=np.zeros(3),
        rotation=np.eye(3),
        child="l1",
    )
    return Chain([joint], [Link("l1", mass, np.asarray(com, dtype=float))])


def test_payload_adds_mass_and_moves_centre():
    chain = _one_joint_chain(mass=1.0, com=(0.0, 0.0, 0.10))

    loaded = with_payload(chain, 1.0, (0.0, 0.0, 0.30))

    assert loaded.links[-1].mass == pytest.approx(2.0)
    # Equal masses put the combined centre exactly between the two.
    assert loaded.links[-1].com == pytest.approx([0.0, 0.0, 0.20])


def test_payload_leaves_the_original_chain_alone():
    chain = _one_joint_chain(mass=1.0, com=(0.0, 0.0, 0.10))

    with_payload(chain, 5.0, (0.0, 0.0, 0.40))

    assert chain.links[-1].mass == pytest.approx(1.0)
    assert chain.links[-1].com == pytest.approx([0.0, 0.0, 0.10])


def test_zero_payload_returns_the_same_chain():
    chain = _one_joint_chain()

    assert with_payload(chain, 0.0, (0.0, 0.0, 1.0)) is chain


def test_negative_mass_is_refused():
    """A negative load would silently cancel real mass — refuse it loudly."""
    with pytest.raises(KinematicsError, match="negative"):
        with_payload(_one_joint_chain(), -0.5, (0.0, 0.0, 0.1))


def test_centre_needs_three_values():
    with pytest.raises(KinematicsError, match="three values"):
        with_payload(_one_joint_chain(), 1.0, (0.0, 0.1))


def test_payload_raises_the_gravity_torque_it_should():
    """The whole point: a heavier, farther load asks the joint for more torque."""
    chain = _one_joint_chain(mass=1.0, com=(0.0, 0.0, 0.10))
    # Rotate the link out horizontally so gravity has a lever to work on.
    q = [np.pi / 2]

    bare = float(chain.gravity_torque(q)[0])
    loaded = float(with_payload(chain, 0.835, (0.0, 0.0, 0.22)).gravity_torque(q)[0])

    assert abs(loaded) > abs(bare)
    # Torque is linear in mass*lever, so the ratio is exactly the moment ratio.
    expected = (1.0 * 0.10 + 0.835 * 0.22) / (1.0 * 0.10)
    assert loaded / bare == pytest.approx(expected, rel=1e-9)


def test_payload_on_a_massless_link_takes_the_payload_centre():
    chain = _one_joint_chain(mass=0.0, com=(0.0, 0.0, 0.0))

    loaded = with_payload(chain, 2.0, (0.01, 0.0, 0.25))

    assert loaded.links[-1].mass == pytest.approx(2.0)
    assert loaded.links[-1].com == pytest.approx([0.01, 0.0, 0.25])
