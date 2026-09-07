"""`--payload` parsing — the hand mass the gravity chain leaves out.

Bad input here would silently change how much torque the arm feeds forward, so
each malformed shape gets refused by name rather than coerced.
"""

from __future__ import annotations

import pytest

from robot_control.cli import _parse_payload


def test_none_stays_none():
    """No flag means no payload — not a zero one."""
    assert _parse_payload(None) is None


def test_parses_mass_and_centre():
    mass, centre = _parse_payload("0.8350,-0.00450,-0.01723,0.22147")

    assert mass == pytest.approx(0.835)
    assert centre == pytest.approx([-0.0045, -0.01723, 0.22147])


def test_tolerates_spaces():
    mass, centre = _parse_payload(" 1.0 , 0.0 , 0.0 , 0.2 ")

    assert mass == pytest.approx(1.0)
    assert centre == pytest.approx([0.0, 0.0, 0.2])


@pytest.mark.parametrize("text", ["1.0,0.0,0.2", "1.0,0.0,0.0,0.2,0.3", "1.0"])
def test_wrong_count_is_refused(text):
    with pytest.raises(SystemExit, match="four values"):
        _parse_payload(text)


def test_non_numeric_is_refused():
    with pytest.raises(SystemExit, match="must be numbers"):
        _parse_payload("heavy,0.0,0.0,0.2")
