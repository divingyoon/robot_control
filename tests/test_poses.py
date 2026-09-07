"""The taught-pose store: a YAML file per profile, one entry per name.

Read and written whole, so what is on disk is always a complete file rather
than a partial one a crash left behind. Every operation returns a new mapping;
nothing here edits the one it was given.
"""

import numpy as np
import pytest
import yaml

from robot_control.poses import (
    PoseStoreError,
    TaughtPose,
    default_poses_path,
    load_poses,
    pose_values,
    save_poses,
    with_pose,
)


def _pose(name="a", groups=("openarm_right_arm",), values=(0.1, 0.2), gravity=None):
    return TaughtPose(
        name=name,
        groups=groups,
        joints={"r_aj_1": values[0], "r_aj_2": values[1]},
        gravity={"r_aj_1": 1.0, "r_aj_2": 1.0} if gravity is None else gravity,
        saved_at="2026-09-03T00:00:00",
    )


def test_missing_file_is_an_empty_store(tmp_path):
    assert load_poses(tmp_path / "none.yaml", profile="openarm_tesollo") == {}


def test_round_trip_preserves_every_field(tmp_path):
    path = tmp_path / "poses.yaml"
    poses = with_pose({}, _pose())
    save_poses(path, poses, profile="openarm_tesollo")

    loaded = load_poses(path, profile="openarm_tesollo")

    assert loaded == poses
    assert loaded["a"].gravity == {"r_aj_1": 1.0, "r_aj_2": 1.0}


def test_with_pose_returns_a_new_mapping_and_replaces_by_name():
    first = with_pose({}, _pose(values=(0.1, 0.2)))
    second = with_pose(first, _pose(values=(0.3, 0.4)))

    assert first["a"].joints["r_aj_1"] == 0.1
    assert second["a"].joints["r_aj_1"] == 0.3
    assert first is not second


def test_store_refuses_another_profiles_file(tmp_path):
    path = tmp_path / "poses.yaml"
    save_poses(path, with_pose({}, _pose()), profile="openarm_tesollo")
    with pytest.raises(PoseStoreError, match="profile"):
        load_poses(path, profile="something_else")


def test_store_refuses_a_malformed_file(tmp_path):
    path = tmp_path / "poses.yaml"
    path.write_text(yaml.safe_dump({"schema": 1, "profile": "p", "poses": [1, 2]}))
    with pytest.raises(PoseStoreError):
        load_poses(path, profile="p")
    path.write_text("- just: a list\n")
    with pytest.raises(PoseStoreError):
        load_poses(path, profile="p")


def test_pose_values_follow_the_groups_joint_order():
    class Group:
        name = "openarm_right_arm"
        joints = ("r_aj_2", "r_aj_1")

    assert np.allclose(pose_values(_pose(values=(0.1, 0.2)), Group()), [0.2, 0.1])


def test_pose_values_refuse_a_pose_taught_on_another_group():
    class Group:
        name = "openarm_left_arm"
        joints = ("r_aj_1", "r_aj_2")

    with pytest.raises(PoseStoreError, match="openarm_right_arm"):
        pose_values(_pose(), Group())


def test_a_pose_can_span_an_arm_and_its_gripper():
    pose = TaughtPose(
        name="grip", groups=("openarm_left_arm", "openarm_left_gripper"),
        joints={"l_aj_1": 0.1, "l_hj_gripper_1": 0.02}, gravity={"l_aj_1": 1.0},
        saved_at="",
    )

    class Gripper:
        name = "openarm_left_gripper"
        joints = ("l_hj_gripper_1",)

    assert np.allclose(pose_values(pose, Gripper()), [0.02])


def test_pose_values_refuse_a_pose_missing_a_joint():
    class Group:
        name = "openarm_right_arm"
        joints = ("r_aj_1", "r_aj_2", "r_aj_3")

    with pytest.raises(PoseStoreError, match="r_aj_3"):
        pose_values(_pose(), Group())


def test_taught_pose_refuses_a_nonfinite_value():
    with pytest.raises(PoseStoreError):
        _pose(values=(float("nan"), 0.0))


def test_default_path_is_per_profile_under_the_checkout():
    path = default_poses_path("openarm_tesollo")
    assert path.name == "openarm_tesollo.yaml"
    assert path.parent.name == "poses"
