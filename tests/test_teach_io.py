"""Recording files and their merging across joint-state topics."""

import numpy as np
import pytest

from robot_control.teach_io import merge_recordings
from robot_control.track import Recording, TrackError


def test_merge_puts_the_other_topic_on_the_primary_clock():
    arm = Recording(np.array([0, 10, 20, 30], dtype=np.int64), np.zeros((4, 1)), ("a",))
    # The hand publishes on its own clock, offset and at a different rate.
    hand = Recording(np.array([5, 25], dtype=np.int64), np.array([[0.0], [1.0]]), ("h",))

    merged = merge_recordings(arm, [hand])

    assert merged.joint_names == ("a", "h")
    assert np.array_equal(merged.timestamps_ns, arm.timestamps_ns)
    assert np.allclose(merged.values[:, 1], [0.0, 0.25, 0.75, 1.0])


def test_merge_refuses_a_topic_with_one_sample():
    arm = Recording(np.array([0, 10], dtype=np.int64), np.zeros((2, 1)), ("a",))
    hand = Recording(np.array([5], dtype=np.int64), np.array([[0.0]]), ("h",))
    with pytest.raises(TrackError):
        merge_recordings(arm, [hand])
