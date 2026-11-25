"""Reading signs with the chair's camera: where each sign is, which way it faces, and what it says.

The chair's forward camera (960 x 540, at a seated rider's eye height) is read a few times a second.
In each picture:
  1. Sign tags are found (OpenCV's ArUco detector). A tag's four corners give the sign's position and
     the way it faces (solvePnP), from the chair's pose, so every sign goes on the chair's map. The
     tag's number block gives the kind of sign (signs.py: plate, blade or direction) and so its size.
  2. The sign is straightened out (a homography from the tag's corners to the sign's known layout)
     and its words are read with OCR (RapidOCR, a small PaddleOCR model run with onnxruntime). On a
     direction sign each row's arrow is read from its shape: where the arrow's weight lies.
  3. A sign counts as read once two readings agree (or one is very sure). Its position is the
     average of every sighting.

read="tags" skips the OCR and takes each name from the building's list of signs (as a QR code would
carry it): quicker, and what the open-world mode falls back to if RapidOCR is not installed. The
evaluation scripts (sign_eval.py) measure both against the true answer.
"""

from __future__ import annotations

import math
import os
import queue
import re
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from webots import signs

CAMERA_X, CAMERA_Z = 0.1, 1.15      # the camera on the chair (protos/Wheelchair.proto): ahead of its centre, height
CAMERA_PITCH = 0.15                 # ... and tilted up this much (radians), to see signs hung high up close by


def camera_to_chair(v):
    """A vector from OpenCV's camera frame (x right, y down, z ahead) to the chair's (x ahead, y left,
    z up), through the camera's upward tilt."""
    c, s = math.cos(CAMERA_PITCH), math.sin(CAMERA_PITCH)
    return np.array([v[2] * c + v[1] * s, -v[0], v[2] * s - v[1] * c])
EVERY_S = 0.2                       # how often the camera is read
AGAIN_BELOW = 0.8                   # a line of words read less surely than this is read again on its own ...
AGAIN_SCALES = (1.5, 2.0)           # ... this much bigger, and the surest reading kept
READ_MIN_TAG_PX = 22                # read words only off signs whose tag is at least this big in the picture
READ_MIN_TAG_PX_DIRECTORY = 40      # ... a directory's small print: from about 2.5 m
MIN_TAG_PX = 10                     # smaller tags are too far away for a trustworthy position
TRUST_M = 14.0                      # ... and so are signs further than this
OCR_WAITING = 2                     # signs waiting to be read at once (OCR takes ~0.1 s a sign, in its own thread)
OCR_THREADS = max(1, (os.cpu_count() or 2) // 2)   # OCR uses half the cores at most, so Webots keeps the rest
RECTIFY_PX_PER_M = 600
AGREE = 2                           # readings that must agree
SAME_DOOR_M = 0.65                  # signs this close are one door's (signs.py: a door's are within 0.5 m,
                                    # the next door's 0.8 m away at least)
SURE = 0.985                        # ... or one reading this sure
# Words found on signs in hospitals, offices and warehouses: a reading close to one of them is taken as
# it (OCR's lexicon, as text readers use a dictionary). General words, not this building's names.
SIGN_WORDS = """Room Rooms Ward Toilet Toilets Accessible Waiting Area Reception Main Cafe Lift Lifts Stairs Exit
Entrance Emergency Department Pharmacy Pathology Collection Imaging Medical Radiology X-Ray Ultrasound MRI CT Scan
Nuclear Medicine Outpatient Outpatients Clinic Clinics Consult Treatment Physiotherapy Occupational Therapy Allied
Health Discharge Lounge Staff Only Clean Dirty Utility Linen Store Medication Nurses' Station Office Meeting Kitchen
Print Storeroom Quiet Comms Loading Dock Docks Goods In Dispatch Forklift Charging First Aid Amenities Aisle Aisles
Open Plan Theatre Theatres Maternity Children's Surgery Day Cardiology Oncology Dental Chapel Shop Information
West East North South Central Zone Directory Boardroom Training Conference Records Breakout Hot Desks Lunch
Manager's Plant Cleaner's Fire Bedroom Study Living Dining Bathroom Laundry""".split()
WORDS = {w.lower() for w in SIGN_WORDS}
STAFF_ONLY = "staff only"
WING_WORDS = ["West", "East", "North", "South", "Central"]   # wings are named this way: "East Wing"
UNFINISHED = [r"\d+[a-z]? ?room", r"\d+[a-z]?[a-z]", r"(consult|consult room|meeting room|loading dock|aisle|ward)"]
                                    # names with their number or letter missing: "4B Room", "4BR", "Aisle"
KIND_OF = {}
for _kind, _k in signs.KINDS.items():
    for _t in range(_k["first"], _k["first"] + _k["count"]):
        KIND_OF[_t] = _kind


def normalise(text: str) -> str:
    """For comparing names: lower case, letters and digits only, single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def correct(text: str) -> str:
    """Each word that is nearly a common sign word becomes that word ("Tollets" -> "Toilets"), and a
    word read in two pieces is put back together when that is much nearer one ("In ing" -> "Dining")."""
    import difflib
    lower = {w.lower(): w for w in SIGN_WORDS}

    def nearest(word):
        match = difflib.get_close_matches(word.lower(), list(lower), n=1, cutoff=0.75)
        if not match:
            return None, 0.0
        return lower[match[0]], difflib.SequenceMatcher(None, word.lower(), match[0]).ratio()

    out, before, before_score = [], None, 0.0
    for word in text.split():
        if any(c.isdigit() for c in word):
            out.append(word)
            before = None
            continue
        alone, score = nearest(word) if len(word) > 2 else (None, 0.0)
        if before is not None:
            joined, joined_score = nearest(before + word)
            if joined and joined_score > max(score, before_score) + 0.05:
                out[-1], before = joined, None
                continue
        out.append(alone or word)
        before, before_score = word, score
    return " ".join(out)


def wing_name(text: str) -> str:
    """A wing as read (small print: "Eastwing", "East Ty", "Central"): the nearest "<West|East|...> Wing"."""
    import difflib
    t = normalise(text).replace(" ", "").replace("wing", "")
    if not t:
        return ""
    best = max(WING_WORDS, key=lambda w: difflib.SequenceMatcher(None, t, w.lower()).ratio())
    if difflib.SequenceMatcher(None, t, best.lower()).ratio() >= 0.5:
        return f"{best} Wing"
    return correct(tidy(text))


def tidy(text: str) -> str:
    """Clean up what OCR read off a sign: the model runs words together and mixes up O and 0, l and 1.
    "4ARo0m6" -> "4A Room 6", "CTScan" -> "CT Scan"."""
    words = []
    for word in text.split():
        letters = sum(c.isalpha() for c in word)
        digits = sum(c.isdigit() for c in word)
        if letters >= 2 and digits:            # a word with a stray digit in the middle of letters
            word = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "o", word)
            word = re.sub(r"(?<=[A-Za-z])1(?=[A-Za-z])", "l", word)
        words.append(word)
    text = " ".join(words)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)               # aB -> a B
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)          # ARoom -> A Room
    text = re.sub(r"(?<=[A-Za-z]{2})(?=\d)", " ", text)             # Room6 -> Room 6 (but not 4A)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Seen:
    """One sign face as the chair knows it."""
    tag: int
    kind: str
    xs: list = field(default_factory=list)       # sightings of its centre
    ys: list = field(default_factory=list)
    zs: list = field(default_factory=list)
    yaws: list = field(default_factory=list)
    weights: list = field(default_factory=list)  # closer sightings count more: 1 / distance squared
    votes: dict = field(default_factory=dict)    # reading -> number of times (plates and blades)
    best: float = 0.0                            # the surest single reading's score
    rows: dict = field(default_factory=dict)     # direction signs: (words, arrow) -> times read
    cells: dict = field(default_factory=dict)    # directories: (column, row) -> {(name, wing): times read}
    here_votes: dict = field(default_factory=dict)   # directories: "You are here" wing -> times read
    attempts: int = 0                            # times its words were read
    name: str | None = None                      # once read
    arrows: list = field(default_factory=list)   # direction signs, once read: [(group, arrow)]
    last_seen: float = 0.0
    times: int = 0
    _at: tuple = field(default=(-1, 0.0, 0.0, 0.0), repr=False)   # (sightings, x, y, yaw) last worked out

    def _where(self) -> tuple:
        """Where the face is, from its 12 closest sightings, worked out again only when there are new
        ones: asked for many times a step, it was the open world's biggest cost after a long trip."""
        if self._at[0] != len(self.weights):
            order = np.argsort(self.weights)[-12:]
            w = np.array(self.weights)[order]
            a = np.array(self.yaws)[order]
            self._at = (len(self.weights), float(np.average(np.array(self.xs)[order], weights=w)),
                        float(np.average(np.array(self.ys)[order], weights=w)),
                        float(math.atan2((np.sin(a) * w).sum(), (np.cos(a) * w).sum())))
        return self._at

    @property
    def x(self):
        return self._where()[1]

    @property
    def y(self):
        return self._where()[2]

    @property
    def yaw(self):
        return self._where()[3]

    @property
    def listing(self) -> list[tuple[str, str]]:
        """A directory's entries read so far, in its order: [(name, wing)], each agreed by AGREE readings."""
        out = []
        for key in sorted(self.cells):
            (name, wing), n = max(self.cells[key].items(), key=lambda kv: kv[1])
            if n >= AGREE:
                out.append((name, wing))
        return out

    @property
    def here(self) -> str:
        return max(self.here_votes, key=self.here_votes.get) if self.here_votes else ""

    @property
    def read(self) -> bool:
        if self.kind == "direction":                 # rows are read one by one: read it a few times
            return self.attempts >= 4 or (len(self.arrows) >= 2 and self.attempts >= 2)
        if self.kind == "directory":                 # every entry seen agreed, or tried enough
            return self.attempts >= 5 or (self.attempts >= 2 and bool(self.cells) and
                                          len(self.listing) == len(self.cells))
        return self.name is not None


class SignReader:
    """Reads the chair camera's pictures into a map of signs. Call `look()` after every step."""

    def __init__(self, scene, layout: signs.Layout | None = None, read: str = "text", every_s: float = EVERY_S,
                 wait_for_ocr: bool = False, names=(), ocr_s: float = 0.0):
        """wait_for_ocr: read words before returning from look() (evaluation) instead of in the background.
        ocr_s (with wait_for_ocr): as if each reading took this long in the background, in simulated time,
        one at a time: the result counts only from then, and while OCR_WAITING are waiting no more are
        started, as run.py's reading thread does. So an evaluation run faster than real time reads signs
        as late as run.py does.
        A reading that is clearly only part of a name ("4B Room" for "4B Room 4": OCR often drops a
        last letter) is not taken as the sign's name; `names`, if given, also rules out any reading
        that is part of one of them."""
        self.names = {normalise(n).replace(" ", "") for n in names}
        self.scene, self.layout = scene, layout
        self.truth = layout.by_tag() if layout is not None else {}
        self.camera = scene.camera
        period = max(scene.step_ms, int(round(every_s * 1000 / scene.step_ms)) * scene.step_ms)
        self.camera.enable(period)
        scene.hide_from_cameras_again()            # what is drawn for the rider is not read off the scene
        self.period_s = period / 1000
        w, h = self.camera.getWidth(), self.camera.getHeight()
        f = (w / 2) / math.tan(self.camera.getFov() / 2)
        self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
        self.size = (w, h)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.errorCorrectionRate = 0.2        # 250 small codes are close to each other: correct few bits, so no wrong numbers
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(signs.ARUCO), params)
        self.read_mode = read
        self.ocr = None
        if read == "text":
            try:
                from rapidocr_onnxruntime import RapidOCR
                self.ocr = RapidOCR(intra_op_num_threads=OCR_THREADS, inter_op_num_threads=1)
            except ImportError:
                print("vision: RapidOCR is not installed (pip install rapidocr-onnxruntime), so sign names "
                      "come from their tags instead of their words", flush=True)
                self.read_mode = "tags"
        self.seen: dict[int, Seen] = {}
        self.wait_for_ocr = wait_for_ocr
        self.jobs, self.results, self.pending = queue.Queue(), queue.Queue(), set()
        if self.ocr is not None and not wait_for_ocr:
            threading.Thread(target=self._worker, daemon=True).start()
        self.last_look = -math.inf
        self.last_picture = None
        self.last_detections = []
        self.reads = 0
        self.ocr_s, self.ocr_free_at, self.delayed = ocr_s, -math.inf, []   # ocr_s: [(when done, result)]

    # ------------------------------------------------------------ the picture
    def picture(self) -> np.ndarray | None:
        data = self.camera.getImage()
        w, h = self.size
        if not data or len(data) < w * h * 4:
            return None
        return np.frombuffer(data, np.uint8).reshape(h, w, 4)[:, :, :3].copy()

    def chair_pose(self):
        node = self.scene.chair_node
        x, y, z = node.getPosition()
        m = node.getOrientation()
        return x, y, z, math.atan2(m[3], m[0])

    def look(self, force: bool = False) -> list[int]:
        """Read the newest picture if one has come since the last look. Returns the tags seen in it."""
        now = self.scene.sup.getTime()
        if not force and now - self.last_look < self.period_s - 1e-6:
            return []
        self.last_look = now
        img = self.picture()
        if img is None:
            return []
        self.last_picture = img
        return self.process(img, self.chair_pose(), now)

    def detect(self, img):
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is None:
            return []
        return [(int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners) if int(i) in KIND_OF]

    def process(self, img, pose, now) -> list[int]:
        self._collect(now)
        found = []
        detections = self.detect(img)
        self.last_detections = detections
        # the biggest unread signs are read first
        detections.sort(key=lambda d: -cv2.contourArea(d[1].astype(np.float32)))
        for tag, c in detections:
            kind = KIND_OF[tag]
            if _side_px(c) < MIN_TAG_PX:
                continue
            place = self.locate(kind, c, pose)
            if place is None:
                continue
            x, y, z, yaw, d = place
            s = self.seen.setdefault(tag, Seen(tag, kind))
            if len(s.xs) >= 3 and math.dist((x, y), (s.x, s.y)) > max(1.5, 0.15 * d):
                continue                                 # far from where it has been seen before: a bad fit
            s.xs.append(x), s.ys.append(y), s.zs.append(z), s.yaws.append(yaw), s.weights.append(1.0 / max(d, 0.5) ** 2)
            s.last_seen, s.times = now, s.times + 1
            found.append(tag)
            if s.read or tag in self.pending:
                continue
            if kind == "directory" and self.have_directory():
                continue                                 # one directory read is enough: they all list the same
            if self.read_mode == "tags":
                self._from_tag(s)
            elif _side_px(c) >= (READ_MIN_TAG_PX_DIRECTORY if kind == "directory" else READ_MIN_TAG_PX) \
                    and len(self.pending) < OCR_WAITING and self.whole(kind, c):
                job = (tag, kind, self.rectify(img, kind, c))
                if self.wait_for_ocr and self.ocr_s > 0:      # done now, but known only once a thread would have it
                    self.ocr_free_at = max(now, self.ocr_free_at) + self.ocr_s
                    self.pending.add(tag)
                    self.delayed.append((self.ocr_free_at, self._work(job)))
                elif self.wait_for_ocr:
                    self._apply(*self._work(job))
                else:
                    self.pending.add(tag)
                    self.jobs.put(job)
        return found

    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                self.results.put(self._work(job))
            except Exception as e:                       # one bad picture must not stop the reading
                self.results.put((job[0], job[1], None))
                print(f"vision: could not read tag {job[0]}: {e}", flush=True)

    def _work(self, job):
        tag, kind, crop = job
        self.reads += 1
        if kind == "directory":
            return tag, kind, ("directory",) + self.read_directory(crop)
        lines = self.ocr_lines(crop)
        if kind == "direction":
            return tag, kind, ("rows", self.read_rows(crop, lines))
        import difflib
        kept = [(t, sc) for t, sc, _ in lines        # not the STAFF ONLY line (0.8: "Staff Room" and "Station 4A" stay)
                if difflib.SequenceMatcher(None, normalise(t).replace(" ", ""), "staffonly").ratio() < 0.8]
        words = correct(tidy(" ".join(t for t, _ in kept)))
        return tag, kind, ("words", words, min((sc for _, sc in kept), default=0.0))

    def _collect(self, now: float = math.inf) -> None:
        while self.delayed and self.delayed[0][0] <= now + 1e-9:
            result = self.delayed.pop(0)[1]
            self.pending.discard(result[0])
            self._apply(*result)
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                return
            self.pending.discard(result[0])
            self._apply(*result)

    def _apply(self, tag, kind, result) -> None:
        s = self.seen.get(tag)
        if s is None or result is None:
            return
        s.attempts += 1
        if result[0] == "directory":
            _, here, cells = result
            if here:
                s.here_votes[here] = s.here_votes.get(here, 0) + 1
            for key, name, wing, score in cells:
                if score < 0.6 or len(normalise(name).replace(" ", "")) < 3 or self.part_of_a_name(name):
                    continue
                votes = s.cells.setdefault(key, {})
                votes[(name, wing)] = votes.get((name, wing), 0) + (AGREE if score >= SURE else 1)
            return
        if result[0] == "rows":
            for row in result[1]:
                s.rows[row] = s.rows.get(row, 0) + 1
            s.arrows = [row for row, n in s.rows.items() if n >= AGREE]
            return
        _, words, score = result
        if not words or score < 0.75 or len(normalise(words).replace(" ", "")) < 3 or self.part_of_a_name(words):
            return
        s.votes[words] = s.votes.get(words, 0) + 1
        s.best = max(s.best, score)
        if s.votes[words] >= AGREE or (score >= SURE and len(normalise(words).replace(" ", "")) >= 5):
            s.name = words                           # one sure reading will do, but not of a short word ("Bed")

    def part_of_a_name(self, words: str) -> bool:
        n = normalise(words)
        if words.rstrip().endswith("'") or any(re.fullmatch(p, n) for p in UNFINISHED):
            return True
        w = n.replace(" ", "")
        if " " not in n and n not in WORDS and any(x.startswith(n) and len(x) >= len(n) + 2 for x in WORDS):
            return True                              # the start of a sign word: "Bed" of "Bedroom"
        return bool(self.names) and w not in self.names and any(w in name for name in self.names)

    # ------------------------------------------------------------ where
    def locate(self, kind: str, corners, pose):
        """The sign face's centre (x, y, z) and the way it faces (yaw), in the world."""
        k = signs.KINDS[kind]
        s = k["tag"] / 2
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        # A small square seen at an angle fits two poses (tilted towards or away from the camera). Signs
        # hang straight up, so the pose whose face is upright (normal level with the floor) is the one.
        n_fit, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, corners.astype(np.float64), self.K, None,
                                                     flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not n_fit:
            return None
        fits = [(cv2.Rodrigues(r)[0], t) for r, t in zip(rvecs, tvecs)]
        R, tvec = min(fits, key=lambda f: abs(camera_to_chair(f[0][:, 2])[2]))   # the most level normal
        left, top, side = signs.tag_square(kind)
        # the face's centre from the tag's: right by (w/2 - tag centre), down by (h/2 - tag centre)
        to_centre = k["w"] / 2 - (left + side / 2)
        down = k["h"] / 2 - (top + side / 2)
        centre = tvec.ravel() + R[:, 0] * to_centre - R[:, 1] * down
        normal = R[:, 2]                             # out of the sign, towards the camera
        c = camera_to_chair(centre) + np.array([CAMERA_X, 0.0, CAMERA_Z])
        n = camera_to_chair(normal)
        x0, y0, z0, h = pose
        ch, sh = math.cos(h), math.sin(h)
        wx, wy = x0 + ch * c[0] - sh * c[1], y0 + sh * c[0] + ch * c[1]
        nx, ny = ch * n[0] - sh * n[1], sh * n[0] + ch * n[1]
        d = math.hypot(c[0], c[1])
        if d > TRUST_M or c[0] < 0.2:                 # too far to trust, or behind the camera (a bad fit)
            return None
        return wx, wy, z0 + c[2], math.atan2(ny, nx), d

    # ------------------------------------------------------------ what
    def _from_tag(self, s: Seen) -> None:
        face = self.truth.get(s.tag)
        if face is None:
            return
        if face.kind == "direction":
            s.arrows = [tuple(r) for r in face.rows]
            s.attempts = 4
        elif face.kind == "directory":
            s.cells = {divmod(i, signs.DIRECTORY_ROWS): {tuple(r): AGREE} for i, r in enumerate(face.rows)}
            s.here_votes = {face.here: 1} if face.here else {}
            s.attempts = 5
        else:
            s.name = face.place

    def whole(self, kind: str, corners) -> bool:
        """Whether all of the sign face is in the picture (part of a sign reads as a different name)."""
        k = signs.KINDS[kind]
        left, top, side = signs.tag_square(kind)
        face_pts = np.array([[left, top], [left + side, top], [left + side, top + side], [left, top + side]], np.float32)
        H = cv2.getPerspectiveTransform(face_pts, corners.astype(np.float32))
        box = np.array([[[0, 0]], [[k["w"], 0]], [[k["w"], k["h"]]], [[0, k["h"]]]], np.float32)
        pts = cv2.perspectiveTransform(box, H).reshape(4, 2)
        w, h = self.size
        return bool(((pts[:, 0] > 2) & (pts[:, 0] < w - 2) & (pts[:, 1] > 2) & (pts[:, 1] < h - 2)).all())

    def rectify(self, img, kind: str, corners) -> np.ndarray:
        """The sign's words, straightened out: the text part of the face at RECTIFY_PX_PER_M."""
        left, top, side = signs.tag_square(kind)
        face_pts = np.array([[left, top], [left + side, top], [left + side, top + side], [left, top + side]], np.float64)
        H = cv2.getPerspectiveTransform(face_pts.astype(np.float32), corners.astype(np.float32))
        tx, ty, tw, th = signs.text_box(kind)
        S = RECTIFY_PX_PER_M
        T = np.array([[1 / S, 0, tx], [0, 1 / S, ty], [0, 0, 1]])
        return cv2.warpPerspective(img, H @ T, (int(tw * S), int(th * S)),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    def _ocr(self, crop, pad: int = 8):
        """OCR's [(box, text, score)], with the crop padded with its own edges first: text right at an
        edge (a sign's top row) is often misread without room around it. Boxes are in the crop's pixels."""
        padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        result, _ = self.ocr(padded)
        return [([[x - pad, y - pad] for x, y in box], text, score) for box, text, score in result or []]

    def ocr_lines(self, crop, again: bool = True) -> list[tuple[str, float, tuple]]:
        """[(text, score, (x, y, w, h))], top to bottom. A line read with little confidence is read again
        on its own at other sizes, and the surest reading kept: OCR misreads a line now and then on a few
        pixels' difference ("West Wing" as "Bu!/, 158/v" on one computer and not on another)."""
        lines = self._lines(self._ocr(crop))
        if not again:
            return lines
        return [self._again(crop, line) if line[1] < AGAIN_BELOW else line for line in lines]

    def _again(self, crop, line):
        text, score, (x, y, w, h) = line
        m = int(round(0.25 * h))
        piece = crop[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
        best = line
        for scale in AGAIN_SCALES:
            parts = self._lines(self._ocr(cv2.resize(piece, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)))
            if parts:
                words = " ".join(t for t, _, _ in sorted(parts, key=lambda p: p[2][0]))
                sure = min(sc for _, sc, _ in parts)
                if sure > best[1]:
                    best = (words, sure, (x, y, w, h))
        return best

    @staticmethod
    def _lines(result) -> list[tuple[str, float, tuple]]:
        out = []
        for box, text, score in result or []:
            pts = np.array(box, dtype=np.float32)
            x, y, w, h = cv2.boundingRect(pts)
            out.append((text.strip(), float(score), (x, y, w, h)))
        out.sort(key=lambda t: (t[2][1] + t[2][3] / 2, t[2][0]))
        # join pieces of the same line
        lines = []
        for t in out:
            if lines and abs((lines[-1][2][1] + lines[-1][2][3] / 2) - (t[2][1] + t[2][3] / 2)) < 0.5 * t[2][3]:
                a = lines[-1]
                x0, y0 = min(a[2][0], t[2][0]), min(a[2][1], t[2][1])
                x1 = max(a[2][0] + a[2][2], t[2][0] + t[2][2])
                y1 = max(a[2][1] + a[2][3], t[2][1] + t[2][3])
                first, second = (a, t) if a[2][0] <= t[2][0] else (t, a)
                lines[-1] = (first[0] + " " + second[0], min(a[1], t[1]), (x0, y0, x1 - x0, y1 - y0))
            else:
                lines.append(t)
        return lines

    def read_directory(self, crop):
        """A directory: ("You are here" wing, [((column, row), name, wing, score)]). Each piece of text
        goes where it sits on the board: the header, or a row's name (left of its column) or wing (right)."""
        S = RECTIFY_PX_PER_M
        header, row_h, cols = signs.directory_layout()
        result = self._ocr(crop)
        here, parts = "", {}
        for box, text, score in result or []:
            pts = np.array(box, dtype=np.float32)
            x, y, w, h = cv2.boundingRect(pts)
            cx, cy = (x + w / 2) / S, (y + h / 2) / S
            text = text.strip()
            if cy < header:
                m = re.search(r"here\W*(.+)$", text, re.I)
                if m and score >= 0.6:
                    here = wing_name(m.group(1))
                continue
            row = int((cy - header) // row_h)
            col = 0 if cx < signs.KINDS["directory"]["w"] / 2 else 1
            if not 0 <= row < signs.DIRECTORY_ROWS:
                continue
            x0, x1 = cols[col]
            part = "name" if (x / S) < x0 + 0.45 * (x1 - x0) else "wing"
            parts.setdefault((col, row), {}).setdefault(part, []).append((x, text, float(score)))
        cells = []
        for key, found in parts.items():
            if "name" not in found:
                continue
            name = correct(tidy(" ".join(t for _, t, _ in sorted(found["name"]))))
            wing = wing_name(" ".join(t for _, t, _ in sorted(found.get("wing", []))))
            score = min(sc for items in found.values() for _, _, sc in items)
            cells.append((key, name, wing, score))
        return here, cells

    def read_rows(self, crop, lines) -> list[tuple[str, str]]:
        """A direction sign's rows: each line of words, and the arrow in the arrow column beside it."""
        rows = []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        S = RECTIFY_PX_PER_M
        row_h = signs.text_box("direction")[3] / signs.MAX_ROWS
        ax = (0.04 + row_h * 0.35) * S                       # the arrow column (signs.render)
        half = row_h * 0.4 * S
        for text, score, (x, y, w, h) in lines:
            text = tidy(re.sub(r"^[^A-Za-z0-9]+", "", text))
            text = correct(re.sub(r"^\S\s+(?=\S{2})", "", text))   # an arrow read as a letter: "L Lifts"
            if re.search(r"w\s?[il1]n[gq]$", text, re.I):   # a wing: "Wast Wlng" is West Wing
                text = wing_name(text)
            if score < 0.6 or len(text) < 2:
                continue
            cy = y + h / 2
            patch = gray[int(max(0, cy - half)):int(cy + half), int(max(0, ax - half)):int(ax + half)]
            arrow = arrow_direction(patch)
            if arrow:
                rows.append((text, arrow))
        return rows

    # ------------------------------------------------------------ what the chair knows
    def have_directory(self) -> bool:
        """Whether a whole directory has been read (not just a few of its entries)."""
        return any(s.kind == "directory" and s.read and s.listing for s in self.seen.values())

    def directory(self) -> dict:
        """What the directories read so far list: name -> wing ("" in a building without wings)."""
        out = {}
        for s in self.seen.values():
            if s.kind == "directory":
                for name, wing in s.listing:
                    out.setdefault(name, wing)
        return out

    def places(self) -> dict:
        """Read places: name -> (x, y) where to stop, from the best sighting of its signs. A door's plate
        and blade faces name one place: where they disagree (a blade read at an angle, "Bed" for
        "Bedroom"), the name read most often wins, the plate's counting once more."""
        read = [s for s in self.seen.values() if s.name and s.kind in ("plate", "blade")]
        plan = getattr(self, "map_plan", None)
        key = (id(plan), tuple((s.tag, len(s.weights), s.name, tuple(sorted(s.votes.items()))) for s in read))
        if key == getattr(self, "_places_for", None):   # nothing new read or mapped since last asked
            return dict(self._places)
        out = {}
        for s in read:
            tally = {}
            for o in read:
                if math.dist((s.x, s.y), (o.x, o.y)) < SAME_DOOR_M:
                    tally[o.name] = tally.get(o.name, 0) + o.votes.get(o.name, 1) + (o.kind == "plate")
            if max(tally, key=lambda n: (tally[n], len(n))) != s.name:
                continue
            better = s.name not in out or (s.kind == "plate" and out[s.name][1] != "plate")
            if better:
                out[s.name] = (self.stop_for(s), s.kind)
        self._places_for, self._places = key, {name: stop for name, (stop, _) in out.items()}
        return dict(self._places)

    def stop_for(self, s: Seen):
        """Where to stop to be at a sign's room: a metre in front of a plate; beside a blade sign,
        on its open side (the blade sticks out from the wall)."""
        if s.kind == "plate":
            return s.x + math.cos(s.yaw) * 1.0, s.y + math.sin(s.yaw) * 1.0
        rx, ry = -math.sin(s.yaw + math.pi), math.cos(s.yaw + math.pi)   # the face's right, seen from in front
        plan = getattr(self, "map_plan", None) or self.scene.plan
        options = [(s.x + rx * 0.7 + math.cos(s.yaw) * 0.3, s.y + ry * 0.7 + math.sin(s.yaw) * 0.3),
                   (s.x - rx * 0.7 + math.cos(s.yaw) * 0.3, s.y - ry * 0.7 + math.sin(s.yaw) * 0.3)]
        return max(options, key=lambda p: plan.clearance_at(*p))

    def hints(self) -> list[tuple[str, float, float, float]]:
        """What direction signs say: [(place or department, sign x, sign y, the way to go)]."""
        out = []
        for s in self.seen.values():
            for group, arrow in s.arrows:
                ahead = s.yaw + math.pi                # someone reading the sign is heading this way
                way = ahead + {"ahead": 0.0, "left": math.pi / 2, "right": -math.pi / 2}[arrow]
                out.append((group, s.x, s.y, way))
        return out


def _side_px(corners) -> float:
    return float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))


def _arrow_templates(size=24):
    out = {}
    for way in ("left", "right", "ahead"):
        t = np.zeros((size, size), np.uint8)
        pts = np.array(signs.arrow_points(way, size / 2, size / 2, size), np.int32)
        cv2.fillPoly(t, [pts], 255)
        out[way] = t > 0
    return out


ARROWS = _arrow_templates()


def arrow_direction(patch) -> str | None:
    """Left, right or ahead: the white arrow's shape, matched against the three arrows signs use."""
    if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
    _, white = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[k]
    if w < 5 or h < 5 or area < 20:
        return None
    side = max(w, h)                                   # square around the blob, so the shape keeps its proportions
    cx, cy = x + w / 2, y + h / 2
    blob = (labels == k).astype(np.uint8) * 255
    M = np.array([[24 / side, 0, 12 - cx * 24 / side], [0, 24 / side, 12 - cy * 24 / side]], np.float32)
    shape = cv2.warpAffine(blob, M, (24, 24)) > 127
    scores = {way: (shape & t).sum() / max(1, (shape | t).sum()) for way, t in ARROWS.items()}
    way = max(scores, key=scores.get)
    return way if scores[way] >= 0.45 else None
