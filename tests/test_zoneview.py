"""webots/zoneview.py: the zone rings drawn round the chair (no Webots needed)."""

import pytest

from assist import proxemics as px
from webots import zoneview as zv


def test_outline_reaches_the_zone_edge_in_front_beside_and_behind():
    pts = zv.outline(0.45, 1.0)
    assert max(x for x, _ in pts) == pytest.approx(px.HALF_LENGTH_M + 0.45)
    assert min(x for x, _ in pts) == pytest.approx(-(px.HALF_LENGTH_M + 0.45))
    assert max(y for _, y in pts) == pytest.approx(px.HALF_WIDTH_M + 0.45)


def test_only_the_front_stretches_with_speed():
    pts = zv.outline(1.2, 2.0)
    assert max(x for x, _ in pts) == pytest.approx(px.HALF_LENGTH_M + 2.4)
    assert min(x for x, _ in pts) == pytest.approx(-(px.HALF_LENGTH_M + 1.2))


def test_the_two_halves_meet_beside_the_chair():
    front, back = zv.half(1.2, 2.0, True), zv.half(1.2, 2.0, False)
    assert front[-1] == pytest.approx(back[0])
    assert back[-1] == pytest.approx(front[0])


def test_a_ring_matches_the_zone_measure():
    """A person whose edge just touches a drawn ring is on that zone's edge in assist/proxemics.py."""
    for k in range(zv.LEVELS):
        share = k / (zv.LEVELS - 1)
        x, y = max(zv.half(0.45, 1 + px.MAX_STRETCH * share, True))     # the ring's front point
        assert float(px.zone_gap(x + 0.36, y, 0.36, share)) == pytest.approx(0.45, abs=1e-3)


def test_the_strips_face_up():
    outer, inner = zv.half(1.2, 1.0, True), zv.half(0.45, 1.0, True)
    quad = [outer[3], outer[4], inner[4], inner[3]]
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(quad, quad[1:] + quad[:1])) / 2
    assert area > 0                       # anticlockwise seen from above


def test_one_front_per_speed_no_outlines_and_only_fields_webots_knows():
    node = zv.zones_node()
    assert node.count("DEF ZONES_F") == zv.LEVELS
    assert node.startswith(f"DEF ZONES Pose {{ translation 0 0 {zv.HIDDEN_Z:g}")
    assert "IndexedLineSet" not in node   # no black edge round the rings
    assert "solid" not in node            # not a field of Webots' IndexedFaceSet
