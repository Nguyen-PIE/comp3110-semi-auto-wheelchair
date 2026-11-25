"""assist/proxemics.py: the zones the proxemic reward and the eval share."""

import math

import numpy as np
import pytest

from assist import proxemics as px

R = 0.36   # a person's radius (0.72 m shoulders)


def ahead(gap_m):
    """Chair-frame x of a person whose edge is gap_m in front of the chair's outline."""
    return px.HALF_LENGTH_M + R + gap_m


def test_touching_the_outline_is_zero():
    assert px.zone_gap(ahead(0.0), 0.0, R, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert px.zone_gap(ahead(-0.1), 0.0, R, 0.0) == 0.0


def test_at_rest_the_gap_is_the_real_distance_in_front_beside_and_behind():
    assert px.zone_gap(ahead(1.0), 0.0, R, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert px.zone_gap(0.0, px.HALF_WIDTH_M + R + 1.0, R, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert px.zone_gap(-ahead(1.0), 0.0, R, 0.0) == pytest.approx(1.0, abs=1e-6)


def test_the_front_stretches_with_speed_but_not_the_sides_or_back():
    assert px.zone_gap(ahead(1.0), 0.0, R, 1.0) == pytest.approx(0.5, abs=1e-6)     # twice as deep
    assert px.zone_gap(ahead(1.0), 0.0, R, 0.5) == pytest.approx(1 / 1.5, abs=1e-6)
    beside = px.HALF_WIDTH_M + R + 1.0
    assert px.zone_gap(0.0, beside, R, 1.0) == pytest.approx(1.0, abs=1e-6)
    assert px.zone_gap(-ahead(1.0), 0.0, R, 1.0) == pytest.approx(1.0, abs=1e-6)


def test_zones():
    assert [px.zone_of(g) for g in (0.0, 0.44, 0.45, 1.19, 1.2, 3.59, 3.6, math.inf)] == [
        "intimate", "intimate", "personal", "personal", "social", "social", "public", "public"]
    assert list(px.zone_index([0.0, 0.45, 1.2, 3.6, np.inf])) == [0, 1, 2, 3, 3]


def test_penalty_is_zero_when_stopped_or_far_and_largest_at_the_outline():
    assert px.penalty(0.0, 0.0) == 0.0
    assert px.penalty(1.2, 1.0) == 0.0
    assert px.penalty(math.inf, 1.0) == 0.0
    assert px.penalty(0.0, 1.0) == pytest.approx(0.27)
    assert px.penalty(0.45, 1.0) == pytest.approx(0.02)        # continuous at the intimate edge
    assert px.penalty(0.0, 0.5) == pytest.approx(0.135)        # scaled by speed


def test_people_behind_the_chair_are_left_out_of_the_reward():
    behind = (-(px.HALF_LENGTH_M + 0.1), 0.0, R)               # chair at the origin facing +x
    assert px.nearest_gap([behind], 0.0, 0.0, 0.0, 1.0) == math.inf
    assert px.nearest_gap([behind], 0.0, 0.0, 0.0, 1.0, skip_behind=False) == 0.0


def test_heading_turns_the_frame():
    person = (0.0, ahead(1.0), R)                              # straight ahead of a chair facing +y
    assert px.nearest_gap([person], 0.0, 0.0, math.pi / 2, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert px.nearest_gap([person], 0.0, 0.0, math.pi / 2, 1.0) == pytest.approx(0.5, abs=1e-6)
