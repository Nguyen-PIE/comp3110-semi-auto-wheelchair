"""Checks for the signs (webots/signs.py), reading them (webots/vision.py) and finding the way without
the floor plan (webots/explore.py). No Webots needed: a sign is drawn into a picture by hand, as the
camera would see it, and a stub stands in for the chair's camera.

    python tests/test_signs.py          (or: python -m pytest tests, if pytest is installed)
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.floorplan import FREE, WALL, load_floorplan  # noqa: E402
from webots import explore, signs, vision  # noqa: E402

HOSPITAL = load_floorplan(str(ROOT / json.loads((ROOT / "webots/worlds/fp_hospital.json").read_text())["map"]))
CORRIDOR = load_floorplan(str(ROOT / json.loads((ROOT / "webots/worlds/corridor.json").read_text())["map"]))
_layouts = {}


def layout(name="fp_hospital"):
    if name not in _layouts:
        _layouts[name] = signs.load(HOSPITAL if name == "fp_hospital" else CORRIDOR, name)
    return _layouts[name]


# ------------------------------------------------------------------ the signs
def test_every_hospital_place_gets_a_plate_and_a_two_sided_blade():
    lay = layout()
    names = [p.name for p in signs._hospital()]
    for name in names:
        kinds = [f.kind for f in lay.faces if f.place == name]
        assert kinds.count("plate") == 1 and kinds.count("blade") == 2, f"{name}: {kinds}"
        assert name in lay.stops


def test_tags_are_unique_and_in_their_kinds_block():
    for name in ("fp_hospital", "corridor"):
        tags = [f.tag for f in layout(name).faces]
        assert len(tags) == len(set(tags))
        for f in layout(name).faces:
            k = signs.KINDS[f.kind]
            assert k["first"] <= f.tag < k["first"] + k["count"], (f.kind, f.tag)


def test_plates_hang_on_a_wall_facing_open_floor():
    geo = signs.Geometry(HOSPITAL)
    for f in layout().faces:
        if f.kind != "plate":
            continue
        behind = geo.march(f.x, f.y, -math.cos(f.yaw), -math.sin(f.yaw), 0.3)
        assert behind is not None and behind <= 0.2, f"{f.place}: no wall right behind its plate"
        front = (f.x + math.cos(f.yaw) * 0.6, f.y + math.sin(f.yaw) * 0.6)
        assert HOSPITAL.clearance_at(*front) > 0.1, f"{f.place}: its plate faces a wall"


def test_stops_are_on_open_floor_near_their_signs():
    lay = layout()
    for name, stop in lay.stops.items():
        assert HOSPITAL.clearance_at(*stop) >= 0.3, f"{name}: stop {stop} is too close to a wall"
        plate = next(f for f in lay.faces if f.place == name and f.kind == "plate")
        assert math.dist(stop, (plate.x, plate.y)) < 2.5


def test_direction_signs_point_the_right_way():
    """Walking the way a direction sign points for a department gets nearer to it."""
    lay = layout()
    faces = [f for f in lay.faces if f.kind == "direction"]
    assert len(faces) >= 8
    good = total = 0
    wings = signs.wing_of("fp_hospital")
    for f in faces:
        for group, arrow in f.rows:                  # a department, or a wing (its departments' places)
            target = min((lay.stops[p.name] for p in signs._hospital()
                          if (p.group == group or wings.get(p.group) == group) and p.name in lay.stops),
                         key=lambda s: math.dist(s, (f.x, f.y)))
            heading = f.yaw + math.pi
            way = heading + {"ahead": 0.0, "left": math.pi / 2, "right": -math.pi / 2}[arrow]
            route = HOSPITAL.route((f.x, f.y), target)
            first = next((p for p in route if math.dist(p, (f.x, f.y)) > 1.0), route[-1]) if route else target
            off = abs(math.atan2(math.sin(math.atan2(first[1] - f.y, first[0] - f.x) - way),
                                 math.cos(math.atan2(first[1] - f.y, first[0] - f.x) - way)))
            total += 1
            good += off < math.radians(80)
    assert good / total >= 0.85, f"only {good} of {total} arrows point along the route"


def test_pictures_have_power_of_two_sides_and_the_tag_reads_back():
    face = next(f for f in layout().faces if f.place == "CT Scan" and f.kind == "plate")
    img = signs.render(face)
    h, w = img.shape[:2]
    assert w & (w - 1) == 0 and h & (h - 1) == 0, (w, h)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(signs.ARUCO))
    _, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert ids is not None and int(ids.ravel()[0]) == face.tag


def test_the_house_has_room_signs_and_no_directory():
    plan = load_floorplan(str(ROOT / json.loads((ROOT / "webots/worlds/fp_house.json").read_text())["map"]))
    lay = signs.load(plan, "fp_house")
    assert {"Bedroom", "Kitchen", "Living Room", "Dining Room", "Study"} <= set(lay.places())
    assert not [f for f in lay.faces if f.kind == "directory"]


def test_every_building_and_its_variants_have_the_same_names():
    for building, variant in (("fp_office", "fp_office_v2"), ("fp_hospital", "fp_hospital_v3"), ("corridor", "corridor_v3"),
                              ("fp_warehouse2", "fp_warehouse2_v2")):
        names = []
        for world in (building, variant):
            plan = load_floorplan(str(ROOT / json.loads((ROOT / f"webots/worlds/{world}.json").read_text())["map"]))
            names.append(set(signs.load(plan, world).places()))
        assert names[0] == names[1], (building, names[0] ^ names[1])
    assert {"Boardroom", "Meeting Room 1", "Lunch Room", "Main Reception"} <= names_of("fp_office")


def names_of(world):
    plan = load_floorplan(str(ROOT / json.loads((ROOT / f"webots/worlds/{world}.json").read_text())["map"]))
    return set(signs.load(plan, world).places())


def test_departments_with_several_rooms_have_a_sign_at_their_entrance():
    lay = layout()
    for dept in ("Ward 4A", "Ward 4B", "Medical Imaging", "Outpatient Clinics", "Allied Health"):
        assert [f for f in lay.faces if f.place == dept and f.kind == "plate"], dept
        assert dept in lay.stops


def test_directories_list_every_department_with_its_wing():
    lay = layout()
    boards = [f for f in lay.faces if f.kind == "directory"]
    assert len(boards) >= 3
    assert min(math.dist((f.x, f.y), lay.stops["Main Reception"]) for f in boards) < 4.0, "one at the main reception"
    rows = dict(boards[0].rows)
    assert rows["Ward 4A"] == "West Wing" and rows["Ward 4B"] == "East Wing" and rows["Medical Imaging"] == "Central Wing"
    assert list(rows)[0] == "Emergency" and "Staff Room" not in rows
    assert {f.here for f in boards} <= {"West Wing", "Central Wing", "East Wing"}


def test_direction_signs_name_other_wings_first():
    lay = layout()
    wings = {"West Wing", "Central Wing", "East Wing"}
    for f in [f for f in lay.faces if f.kind == "direction"]:
        kinds = [g in wings for g, _ in f.rows]
        assert kinds == sorted(kinds, reverse=True), f.rows      # wings, then departments


# ------------------------------------------------------------------ reading them
class Camera:
    def __init__(self, w=960, h=540, fov=1.2):
        self.w, self.h, self.fov = w, h, fov

    def enable(self, period):
        pass

    def getWidth(self):
        return self.w

    def getHeight(self):
        return self.h

    def getFov(self):
        return self.fov


class Scene:
    step_ms = 100
    camera = Camera()
    plan = HOSPITAL

    def hide_from_cameras_again(self):   # (nothing drawn for the rider here)
        pass


def reader(**kw):
    return vision.SignReader(Scene(), layout(), read=kw.pop("read", "tags"), **kw)


def picture_of(face, pose, r):
    """The camera's picture of one sign face, drawn by projecting its corners (no Webots)."""
    from webots.sign_detector import corners_of, project
    tex = signs.render(face)
    pts = project(corners_of(face), pose, r.K)
    dst = np.array([[u, v] for u, v, _ in pts], np.float32)
    th, tw = tex.shape[:2]
    src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    img = np.full((r.size[1], r.size[0], 3), 190, np.uint8)
    warped = cv2.warpPerspective(tex, M, r.size, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), M, r.size) > 0
    img[mask] = warped[mask]
    return img


def standing(face, d, side_deg=0.0):
    a = face.yaw + math.radians(side_deg)
    x, y = face.x + math.cos(a) * d, face.y + math.sin(a) * d
    return (x, y, 0.0, math.atan2(face.y - y, face.x - x))


def test_a_sign_is_put_where_it_is():
    r = reader()
    for f in [f for f in layout().faces if f.place in ("CT Scan", "Pharmacy")][:4]:
        far = 8.0 if f.kind == "blade" else 6.0          # plate tags are smaller: they are found less far away
        for d, side in ((3.0, 0), (5.0, 30), (far, -40)):
            pose = standing(f, d, side)
            img = picture_of(f, pose, r)
            found = dict(r.detect(img))
            assert f.tag in found, f"tag {f.tag} not found at {d} m"
            x, y, z, yaw, dist = r.locate(f.kind, found[f.tag], pose)
            assert math.dist((x, y), (f.x, f.y)) < 0.15 + 0.03 * d, (d, side, (x, y), (f.x, f.y))
            assert abs(z - f.z) < 0.15
            assert abs(math.atan2(math.sin(yaw - f.yaw), math.cos(yaw - f.yaw))) < math.radians(8)


def test_the_words_are_read_off_a_sign():
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        print("      (RapidOCR is not installed: skipped)")
        return
    r = reader(read="text", wait_for_ocr=True)
    for name in ("CT Scan", "Nuclear Medicine", "4A Room 3", "Nurses' Station 4A", "Staff Room 4B"):
        f = next(f for f in layout().faces if f.place == name and f.kind == "blade")
        img = picture_of(f, standing(f, 3.0, 15), r)
        tag, corners = next((t, c) for t, c in r.detect(img) if t == f.tag)
        _, _, (_, words, score) = r._work((tag, f.kind, r.rectify(img, f.kind, corners)))
        assert vision.normalise(words) == vision.normalise(name), (words, name)


def test_a_direction_sign_reads_back_with_its_arrows():
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return
    r = reader(read="text", wait_for_ocr=True)
    f = max((f for f in layout().faces if f.kind == "direction"), key=lambda f: len(f.rows))
    img = picture_of(f, standing(f, 4.0, 0), r)
    tag, corners = next((t, c) for t, c in r.detect(img) if t == f.tag)
    _, _, (_, rows) = r._work((tag, f.kind, r.rectify(img, f.kind, corners)))
    assert [(vision.normalise(t), a) for t, a in rows] == [(vision.normalise(t), a) for t, a in f.rows], (rows, f.rows)


def test_an_unsure_line_is_read_again_and_the_surest_reading_kept():
    import numpy as np
    r = vision.SignReader.__new__(vision.SignReader)
    sizes = []

    def fake_ocr(img):                                # a garbled first reading, better ones bigger
        sizes.append(img.shape[1])
        if len(sizes) == 1:
            return [([[10, 10], [110, 10], [110, 40], [10, 40]], "Bu!/, 158/v", 0.67)]
        return [([[0, 0], [img.shape[1], 0], [img.shape[1], 20], [0, 20]],
                 "West Wing" if len(sizes) == 2 else "Wost", 0.94 if len(sizes) == 2 else 0.7)]
    r._ocr = fake_ocr
    crop = np.zeros((60, 200, 3), np.uint8)
    assert r.ocr_lines(crop) == [("West Wing", 0.94, (10, 10, 101, 31))]
    assert len(sizes) == 1 + len(vision.AGAIN_SCALES)
    sizes.clear()
    assert r.ocr_lines(crop, again=False)[0][0] == "Bu!/, 158/v" and len(sizes) == 1


def test_a_directory_reads_back():
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return
    r = reader(read="text", wait_for_ocr=True)
    f = next(f for f in layout().faces if f.kind == "directory")
    img = picture_of(f, standing(f, 1.8, 10), r)
    tag, corners = next((t, c) for t, c in r.detect(img) if t == f.tag)
    x, y, z, yaw, _ = r.locate(f.kind, corners, standing(f, 1.8, 10))
    assert math.dist((x, y), (f.x, f.y)) < 0.15 and abs(z - f.z) < 0.15
    here, cells = r.read_directory(r.rectify(img, f.kind, corners))
    got = [(vision.normalise(n), vision.normalise(w)) for _, n, w, _ in sorted(cells)]
    want = [(vision.normalise(n), vision.normalise(w)) for n, w in
            sorted(f.rows, key=lambda row: divmod(f.rows.index(row), signs.DIRECTORY_ROWS)[::-1])]
    assert here == f.here and sorted(got) == sorted(want), (here, set(got) ^ set(want))


def test_ocr_slips_are_tidied():
    assert vision.correct(vision.tidy("4ARo0m6")) == "4A Room 6"
    assert vision.correct(vision.tidy("CTScan")) == "CT Scan"
    assert vision.correct(vision.tidy("Tollets")) == "Toilets"
    assert vision.correct(vision.tidy("Walting Area")) == "Waiting Area"
    assert vision.wing_name("Eastwing") == "East Wing" and vision.wing_name("Central") == "Central Wing"
    assert vision.correct(vision.tidy("In ing Room")) == "Dining Room"          # one word read in two pieces
    assert vision.correct(vision.tidy("Phar macy")) == "Pharmacy"
    assert vision.correct(vision.tidy("Main Reception")) == "Main Reception"    # two words stay two
    assert vision.correct(vision.tidy("Goods In")) == "Goods In"


def test_one_doors_signs_agree_on_one_name():
    r = reader()
    for tag, kind, name, votes, x in ((0, "plate", "Dining Room", 2, 0.0), (60, "blade", "Dining", 1, 0.4),
                                      (1, "plate", "Kitchen", 2, 3.0)):
        s = vision.Seen(tag, kind, xs=[x], ys=[0.0], zs=[1.5], yaws=[0.0], weights=[1.0])
        s.name, s.votes = name, {name: votes}
        r.seen[tag] = s
    assert sorted(r.places()) == ["Dining Room", "Kitchen"]


def test_signs_of_different_doors_are_apart():
    for name in ("fp_hospital", "corridor"):
        faces = [f for f in layout(name).faces if f.kind in ("plate", "blade")]
        for i, a in enumerate(faces):
            for b in faces[i + 1:]:
                if a.place != b.place:
                    assert math.dist((a.x, a.y), (b.x, b.y)) >= signs.MIN_SIGN_GAP_M - 0.05, (a.place, b.place)
        for f in faces:                              # and each blade is next to its own plate
            if f.kind == "blade":
                assert min(math.dist((f.x, f.y), (p.x, p.y)) for p in faces
                           if p.kind == "plate" and p.place == f.place) < vision.SAME_DOOR_M


def test_a_directory_is_not_hidden_by_a_wall_beside_it():
    geo = signs.Geometry(HOSPITAL)
    for f in layout().faces:
        if f.kind != "directory":
            continue
        nx, ny, half = math.cos(f.yaw), math.sin(f.yaw), signs.KINDS["directory"]["w"] / 2
        middle = geo.march(f.x + nx * 0.3, f.y + ny * 0.3, -nx, -ny, 1.0)
        for t in np.arange(-half - 0.25, half + 0.25, geo.res):
            x, y = f.x - ny * t + nx * 0.3, f.y + nx * t + ny * 0.3
            depth = geo.march(x, y, -nx, -ny, 1.0)
            assert depth is None or depth > middle - 0.04, (f.tag, t)


def test_part_of_a_name_is_not_taken_for_the_name():
    r = reader()                                     # without the building's names: what a name needs
    assert r.part_of_a_name("4B Room") and r.part_of_a_name("4BR") and r.part_of_a_name("Nurses'")
    assert r.part_of_a_name("Aisle") and r.part_of_a_name("Meeting Room")
    assert not r.part_of_a_name("4B Room 4") and not r.part_of_a_name("Training Room") and not r.part_of_a_name("CT Scan")
    r = reader(names=["Discharge Lounge"])
    assert r.part_of_a_name("Lounge")


def test_arrows_are_told_apart():
    for way in ("left", "right", "ahead"):
        patch = np.full((40, 40), 40, np.uint8)
        cv2.fillPoly(patch, [np.array(signs.arrow_points(way, 20, 20, 26), np.int32)], 255)
        assert vision.arrow_direction(patch) == way


# ------------------------------------------------------------------ finding the way
def scan(plan, x, y, heading, n=360, reach=4.0):
    """A lidar scan from the floor plan, in Webots' order (from straight behind, clockwise), with the
    chair's lidar (360 rays, 4 m)."""
    out = []
    for i in range(n):
        a = heading + math.pi - 2 * math.pi * i / n
        d = 0.0
        while d < reach and plan.clearance_at(x + math.cos(a) * d, y + math.sin(a) * d) > 0.02:
            d += 0.05
        out.append(d if d < reach else math.inf)
    return np.array(out, np.float32)


def test_the_chair_maps_what_its_lidar_sees_and_finds_frontiers():
    m = explore.ChairMap(CORRIDOR)
    for x in (-6.0, -3.0):
        m.update(x, 0.0, 0.0, scan(CORRIDOR, x, 0.0, 0.0), 4.0)
    plan = m.plan((-6.0, 0.0), 0.0)
    truth_walls = CORRIDOR.cells == WALL
    solid = m.logodds > explore.SOLID
    assert (solid & ~truth_walls).sum() < 0.02 * solid.sum() + 50, "walls put where there are none"
    assert m.known_open()[CORRIDOR.cells == FREE].mean() > 0.3
    assert plan.route((-6.0, 0.0), (-3.0, 0.5)) is not None
    fronts = m.frontiers(plan)
    assert fronts, "a corridor seen from one end has unexplored space left"


def test_unknown_space_counts_as_open_for_routes():
    m = explore.ChairMap(CORRIDOR)
    plan = m.plan((0.0, 0.0), 0.0)
    assert plan.route((-8.0, 0.0), (8.0, 0.0)) is not None


def test_names_and_departments_match_as_a_visitor_would():
    assert explore.same_name("CT Scan", "ct scan") and explore.same_name("4A Room 6", "4ARoom 6")
    assert explore.same_name("Pathology Colection", "Pathology Collection")
    assert not explore.same_name("4B Room 1", "4B Room 5"), "a different room number is a different room"
    assert not explore.same_name("4A Room 1", "4B Room 1") and not explore.same_name("Ward 4A", "Ward 4B")
    assert not explore.same_name("Meeting Room 4", "Meeting Room 5") and not explore.same_name("Aisle 3", "Aisle 13")
    assert not explore.same_name("Imaging Reception", "Main Reception"), "word by word, not the whole name"
    assert explore.same_name("Staff Toom", "Staff Room") and not explore.same_name("Eating Area", "Waiting Area")
    assert "medical imaging" in explore.departments_for("CT Scan")
    assert "ward 4a" in explore.departments_for("4A Room 3")
    assert "ward 4a" in explore.departments_for("Treatment Room", "Ward 4A"), "the directory's department counts"


def test_boxed_in_the_chair_backs_out_the_way_it_came():
    """A big room and a small pocket joined by a gap too narrow for the route grid: the chair drove
    in, its map closed the gap behind it, and its way out is back through the gap into the room."""
    from envs.floorplan import build_plan
    from webots.open_world import POCKET_M2, ROUTE_CELL_M, Session
    res = 0.05
    wall = np.zeros((200, 400), bool)                    # 10 m x 20 m, centred on (0, 0)
    wall[:4, :] = wall[-4:, :] = wall[:, :4] = wall[:, -4:] = True
    wall[66:70, 330:] = wall[130:134, 330:] = True       # a 3 m x 3 m pocket at x = 6.7 m to 9.8 m
    wall[66:134, 330:334] = True
    wall[96:104, 330:334] = False                        # its doorway: a 0.4 m gap at y = 0
    pocket = (8.0, 0.0)
    plan = build_plan("test", res, wall, ~wall, (0.0, 0.0), keep_xy=pocket)
    assert float((~plan.blocked).sum()) * ROUTE_CELL_M ** 2 < POCKET_M2, "the pocket should look closed off"
    s = Session.__new__(Session)
    s.trail = [(x, 0.0) for x in np.arange(-4.0, 8.1, 0.3)]
    out = s._way_out(plan, pocket)
    assert out is not None and out[0] < 6.0, out


def test_a_sign_spotted_but_not_read_is_looked_at_and_rooms_of_the_department_pull():
    m = explore.ChairMap(CORRIDOR)
    plan = m.plan((0.0, 0.0), 0.0)
    fronts = [(-6.0, 0.0, 6), (6.0, 0.0, 6)]
    assert explore.Explorer().choose(plan, fronts, (0.0, 0.0), looks=[(5.0, 0.0)]) == (5.0, 0.0)
    west = explore.Explorer().choose(plan, fronts, (0.0, 0.0), target="4A Room 2", department="Ward 4A",
                                     near=[(-7.0, 0.0)])
    assert west[0] < 0, "rooms of the same ward read to the west: look there"


def test_signs_pointing_the_way_win_the_choice_of_frontier():
    m = explore.ChairMap(CORRIDOR)
    plan = m.plan((0.0, 0.0), 0.0)
    fronts = [(-8.0, 0.0, 6), (8.0, 0.0, 6)]
    e = explore.Explorer()
    west = e.choose(plan, fronts, (0.0, 0.0), hints=[("Medical Imaging", 0.0, 0.0, math.pi)], target="CT Scan")
    east = explore.Explorer().choose(plan, fronts, (0.0, 0.0), hints=[("Medical Imaging", 0.0, 0.0, 0.0)], target="CT Scan")
    assert west[0] < 0 < east[0]


def test_once_a_directory_is_read_the_chair_does_not_go_up_to_another():
    """As a visitor notes the wing once: with a directory read, another one spotted on the way is not
    worth a look (before, it is), and the reader does not read it either."""
    from webots.open_world import Session
    boards = [f for f in layout().faces if f.kind == "directory"]
    assert len(boards) >= 2

    def spotted(face, read):
        s = vision.Seen(face.tag, "directory")
        for _ in range(3):
            s.xs.append(face.x), s.ys.append(face.y), s.zs.append(face.z)
            s.yaws.append(face.yaw), s.weights.append(1.0)
        s.times = 3
        if read:
            s.attempts, s.cells = 5, {(0, 0): {("Emergency", "East Wing"): vision.AGREE}}
        return s

    reader = vision.SignReader.__new__(vision.SignReader)
    reader.seen = {boards[1].tag: spotted(boards[1], read=False)}
    s = Session.__new__(Session)
    s.reader, s.mode, s.room_signs, s.department, s.looked, s.look_tries = reader, "looking", False, None, {}, {}
    s.scene = type("Scene", (), {"sup": type("Sup", (), {"getTime": staticmethod(lambda: 0.0)})()})()
    plan = explore.ChairMap(HOSPITAL).plan((boards[1].x, boards[1].y), 0.0)
    assert [tag for _, tag, _ in s._unread(plan)] == [boards[1].tag], "no directory read yet: worth a look"
    reader.seen[boards[0].tag] = spotted(boards[0], read=True)
    assert reader.have_directory()
    assert s._unread(plan) == [], "a directory read already: the other one is passed by"


def test_a_frontier_given_up_on_is_tried_again_later():
    m = explore.ChairMap(CORRIDOR)
    plan = m.plan((0.0, 0.0), 0.0)
    e = explore.Explorer()
    e.give_up((8.0, 0.0), now=10.0)
    e.now = 20.0
    assert e.choose(plan, [(8.0, 0.0, 6)], (0.0, 0.0)) is None
    e.now = 10.0 + explore.TRY_AGAIN_S + 1
    assert e.choose(plan, [(8.0, 0.0, 6)], (0.0, 0.0)) is not None


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL ", name, "-", repr(exc))
    sys.exit(1 if failed else 0)
