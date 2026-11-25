"""Walk people round the chair from phones (webots/run.py --phones), seeing through their eyes.

run.py starts a small web server on the laptop (Python's own, nothing to install) and prints its
address, for example http://192.168.1.23:8765/ab12, and opens a page with a QR code for it. Up to
four phones on the same Wi-Fi open that address in their browser and turn sideways. Each gets a
random walker near the chair, with its own ring colour (players.py), and, like a phone shooter game:
    the whole screen shows what that person sees, live in 3D: a camera on the chair model is moved
    to their eyes every step (protos/Wheelchair.proto, "walker eye 1" to "walker eye 4")
    the left thumb walks: a joystick appears where it touches; up walks forward, down steps back,
    left and right step sideways
    the right thumb drags anywhere on the right half to look round (and up and down); the person
    turns with it, and the view turns at once, before the next picture arrives from the laptop: the
    page keeps the last few pictures and redraws them as seen from the new direction, so turning
    shows what was there a moment ago (or plain sky and floor colours, where nothing was seen yet)
    instead of a black edge
    a marker at the top shows where the chair is, and how far. Walking up to the chair, the view
    looks down at it by itself, so it stays in sight right up to touching it
    New person (take over someone else) and Let go (they walk on by themselves)
The last part of the address is a code made up when run.py starts, so only people who can see the
laptop's screen can join. A phone that stops sending for 1 s stands still; after 20 s its walker
is handed back to the crowd and its slot is free again.

The phones and the laptop must be able to reach each other: university Wi-Fi often stops that, so
on demo day put them all on one phone's hotspot (or the Mac's Internet Sharing) and try it first.
"""

from __future__ import annotations

import http.server
import json
import math
import secrets
import socket
import threading
import time
from urllib.parse import parse_qs, urlsplit

import numpy as np

from webots.players import SLOTS, PhoneInput

STILL_AFTER_S = 1.0     # no word from a phone for this long: its walker stands still
GONE_AFTER_S = 20.0     # ... and for this long: the walker is let go and the slot is free
CHAIR_AND_PERSON_M = 0.4 + 0.36   # the chair's radius and a person's: the gap between them is edge to edge
EYE_HEIGHT_M = 1.6      # a walker's eyes ...
EYE_FORWARD_M = 0.15    # ... just in front of their face, so their own head is not in the picture
EYE_FOV = 1.5           # radians across: the walker eye cameras' fieldOfView in protos/Wheelchair.proto
PITCH_LIMITS = (-0.5, 0.9)   # how far up and down a walker can look, radians (down is positive): the thumb
                             # goes to 0.7 down, and the page looks further down at the chair right in front
JPEG_QUALITY = 85     # with the 1056 x 480 eye cameras, about 4x the bytes of 528 x 240 at 70
PARKED = [0.0, 0.0, -5.0]   # where an unused eye camera waits: under the floor
WAIT_FOR_PICTURE_S = 5.0


def lan_address() -> str:
    """This computer's address on the local network (what the phones type in)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))    # no packet is sent: this only picks the network interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def css_colour(rgb: str) -> str:
    r, g, b = (round(float(v) * 255) for v in rgb.split())
    return f"rgb({r},{g},{b})"


def eye_rotation(yaw: float, pitch: float) -> list[float]:
    """Webots axis and angle for turning `yaw` to the left and then looking `pitch` down."""
    cz, sz, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    w, x, y, z = cz * cp, -sz * sp, cz * sp, sz * cp      # quaternion: about z, then about the new y
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return [0.0, 0.0, 1.0, 0.0]
    return [x / n, y / n, z / n, 2 * math.atan2(n, w)]


class WalkerEyes:
    """The chair model's walker-eye cameras: aim() puts each phone's camera at its walker's eyes
    before a step (in the chair's own frame, since the cameras belong to the chair), and frames()
    returns the pictures as JPEGs after it. A camera is on only while its phone walks someone."""

    def __init__(self, scene):
        self.scene, self.chair = scene, scene.chair_node
        sup = scene.sup
        self.cameras = {s.number: sup.getDevice(f"walker eye {s.number}") for s in SLOTS}
        self.fields = {s.number: (self.chair.getField(f"walkerEye{s.number}Translation"),
                                  self.chair.getField(f"walkerEye{s.number}Rotation")) for s in SLOTS}
        self.on: set[int] = set()
        self.pitch: dict[int, float] = {}       # slot -> how far down its walker looks (run.py sets it)

    def aim(self) -> None:
        """Run just before each step (bridge.py's before_moving), when the people have moved but the
        chair has not: the pictures are taken at the end of the step, so the chair is placed where
        it will be by then."""
        crowd, dt = self.scene.crowd, self.scene.step_ms / 1000
        velocity = self.chair.getVelocity()
        rot = np.array(self.chair.getOrientation()).reshape(3, 3)
        chair_yaw = math.atan2(rot[1, 0], rot[0, 0]) + velocity[5] * dt
        pos = np.array(self.chair.getPosition()) + np.array(velocity[:3]) * dt
        c, s = math.cos(chair_yaw), math.sin(chair_yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for slot, camera in self.cameras.items():
            walker = crowd.player(slot)
            where, turn = self.fields[slot]
            if walker is None:
                if slot in self.on:
                    camera.disable()
                    where.setSFVec3f(PARKED)
                    self.on.discard(slot)
                continue
            if slot not in self.on:
                camera.enable(self.scene.step_ms)
                self.on.add(slot)
                self.scene.hide_from_cameras_again()   # the sensor view, paths and markers: the rider's only
            h = walker.orientation
            eye = np.array([walker.x + EYE_FORWARD_M * math.cos(h), walker.y + EYE_FORWARD_M * math.sin(h), EYE_HEIGHT_M])
            where.setSFVec3f([float(v) for v in rot.T @ (eye - pos)])
            turn.setSFRotation(eye_rotation(h - chair_yaw, self.pitch.get(slot, 0.0)))

    def walkers(self) -> dict[int, list]:
        """Where each phone's walker is: slot -> [x, y, heading]."""
        crowd = self.scene.crowd
        return {slot: [w.x, w.y, w.orientation] for slot in self.cameras if (w := crowd.player(slot)) is not None}

    def frames(self) -> dict[int, bytes]:
        import cv2
        out = {}
        for slot in self.on:
            camera = self.cameras[slot]
            data = camera.getImage()
            w, h = camera.getWidth(), camera.getHeight()
            if not data or len(data) < w * h * 4:
                continue
            img = np.frombuffer(data, np.uint8).reshape(h, w, 4)[:, :, :3]
            ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                out[slot] = jpeg.tobytes()
        return out


class PhoneServer:
    """The web server, in its own thread. run.py calls read() every step for the thumbs and
    buttons, publish() with the chair's state, and set_frames() with what each walker sees.

    Looking round goes as turns (radians to the left since the last message), so none is lost
    when a message is late. Each phone's turns are counted three ways: received from the phone,
    applied to its walker (read() hands them to players.py), and in each picture, so the phone can
    slide a picture by what it has turned since then."""

    def __init__(self, port: int = 8765, code: str | None = None):
        self.code = code or secrets.token_hex(2)
        self.port = port
        self.lock = threading.Lock()
        self.new_frame = threading.Condition(self.lock)
        self.phones: dict[int, dict] = {}
        self.state: dict = {}
        self.state_looks: dict[int, float] = {}            # slot -> turns applied when the state was published
        self.frames: dict[int, tuple] = {}                 # slot -> (picture number, JPEG, turns, pitch)
        self.closing = False
        self.httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), _handler(self))
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.address = f"http://{lan_address()}:{port}/{self.code}"

    # -------------------------------------------------------------- called by run.py
    def read(self) -> dict[int, PhoneInput]:
        """The connected phones' thumbs and the buttons pressed since the last call."""
        now, out = time.monotonic(), {}
        with self.lock:
            for slot, p in list(self.phones.items()):
                quiet = now - p["last"]
                if quiet > GONE_AFTER_S:
                    del self.phones[slot]
                    continue
                moving = quiet <= STILL_AFTER_S
                out[slot] = PhoneInput(forward=p["forward"] if moving else 0.0, side=p["side"] if moving else 0.0,
                                       look=p["look"], pitch=p["pitch"], actions=tuple(p["actions"]))
                p["applied"] += p["look"]
                p["look"], p["pitch_applied"], p["actions"] = 0.0, p["pitch"], []
        return out

    def publish(self, state: dict) -> None:
        with self.lock:
            self.state = state
            self.state_looks = {slot: p["applied"] for slot, p in self.phones.items()}

    def set_frames(self, frames: dict[int, bytes], walkers: dict[int, list] | None = None) -> None:
        """Each walker's newest picture; `walkers` (WalkerEyes.walkers()) says where it was taken, so the
        phone keeps only pictures taken near where its walker now is."""
        if not frames:
            return
        with self.new_frame:
            for slot, jpeg in frames.items():
                p = self.phones.get(slot)
                if p is None:
                    continue
                number = self.frames.get(slot, (0,))[0] + 1
                at = (walkers or {}).get(slot, [math.nan, math.nan])[:2]
                self.frames[slot] = (number, jpeg, p["applied"], p["pitch_applied"], at)
            self.new_frame.notify_all()

    def close(self) -> None:
        with self.new_frame:
            self.closing = True
            self.new_frame.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()

    # -------------------------------------------------------------- called by the phones
    def join(self, slot: int | None = None) -> dict:
        with self.lock:
            if slot not in self.phones:
                free = [s.number for s in SLOTS if s.number not in self.phones]
                if not free:
                    return dict(full=True)
                slot = free[0]
                self.phones[slot] = dict(forward=0.0, side=0.0, look=0.0, pitch=0.0, actions=["join"],
                                         last=time.monotonic(), received=0.0, applied=0.0, pitch_applied=0.0)
            p = self.phones[slot]
            s = next(s for s in SLOTS if s.number == slot)
            return dict(slot=slot, name=s.name, colour=css_colour(s.colour), colour_name=s.colour_name,
                        look=p["received"], pitch=p["pitch"], fov=EYE_FOV)

    def tick(self, slot: int, forward: float = 0.0, side: float = 0.0, look: float = 0.0, pitch: float = 0.0,
             actions: list = ()) -> dict:
        with self.lock:
            p = self.phones.get(slot)
            if p is None:
                return dict(rejoin=True)
            p["forward"] = max(-1.0, min(1.0, float(forward)))
            p["side"] = max(-1.0, min(1.0, float(side)))
            look = float(look) if math.isfinite(float(look)) else 0.0
            p["look"] += look
            p["received"] += look
            p["pitch"] = max(PITCH_LIMITS[0], min(PITCH_LIMITS[1], float(pitch)))
            p["actions"] += [a for a in actions if a in ("let_go", "new")]
            p["last"] = time.monotonic()
            state, applied = self.state, self.state_looks.get(slot, p["applied"])
        walkers = state.get("walkers", {})
        you, chair = walkers.get(slot), state.get("chair")
        gap = None if you is None or chair is None else round(math.dist(you[:2], chair[:2]) - CHAIR_AND_PERSON_M, 2)
        return dict(you=you, chair=chair, gap=gap, status=state.get("status", ""), trip=state.get("trip"),
                    time=state.get("time"), look=applied)

    def frame(self, slot: int, after: int):
        """This slot's newest picture once it is not picture number `after` (waits for it up to
        WAIT_FOR_PICTURE_S): (number, JPEG, turns, pitch, [x, y] where taken), or None."""
        with self.new_frame:
            self.new_frame.wait_for(lambda: self.closing or self.frames.get(slot, (after,))[0] != after,
                                    timeout=WAIT_FOR_PICTURE_S)
            got = self.frames.get(slot)
        return None if self.closing or got is None or got[0] == after else got


def _handler(server: PhoneServer):
    page = PAGE.replace("__CODE__", server.code)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"           # keep each phone's connection open between pictures

        def log_message(self, *args):        # quiet: one line per request would flood the terminal
            pass

        def _send(self, body: bytes, kind: str, status: int = 200, headers: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: dict):
            self._send(json.dumps(data).encode(), "application/json")

        def _parts(self):
            parts = urlsplit(self.path).path.strip("/").split("/")
            return parts[1:] if parts and parts[0] == server.code else None

        def do_GET(self):
            try:
                self._get()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass                              # the phone went away

        def _get(self):
            parts = self._parts()
            if parts is None:
                return self._send(b"Not found: check the address on the laptop's screen.", "text/plain", 404)
            if parts == []:
                return self._send(page.encode(), "text/html; charset=utf-8")
            if len(parts) == 2 and parts[0] == "eye" and parts[1].endswith(".jpg") and parts[1][:-4].isdigit():
                after = parse_qs(urlsplit(self.path).query).get("after", ["0"])[0]
                got = server.frame(int(parts[1][:-4]), int(after) if after.isdigit() else 0)
                if got is None:
                    return self._send(b"", "image/jpeg", 204)
                number, jpeg, look, pitch, at = got
                return self._send(jpeg, "image/jpeg", headers={"X-Frame": str(number), "X-Look": repr(look),
                                                                "X-Pitch": repr(pitch), "X-At": f"{at[0]:.2f},{at[1]:.2f}"})
            if parts == ["qr"]:
                return self._send(QR_PAGE.replace("__ADDRESS__", server.address).replace("__CODE__", server.code).encode(),
                                  "text/html; charset=utf-8")
            if parts == ["qr.png"]:
                import cv2
                img = cv2.QRCodeEncoder.create().encode(server.address)
                img = cv2.resize(img, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST)
                return self._send(cv2.imencode(".png", img)[1].tobytes(), "image/png")
            return self._send(b"Not found", "text/plain", 404)

        def do_POST(self):
            parts = self._parts()
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                body = {}
            try:
                if parts == ["join"]:
                    return self._json(server.join(body.get("slot")))
                if parts == ["tick"]:
                    fields = ("forward", "side", "look", "pitch", "actions")
                    return self._json(server.tick(int(body.get("slot", -1)), **{k: body[k] for k in fields if k in body}))
            except (TypeError, ValueError):
                return self._send(b"Bad request", "text/plain", 400)
            return self._send(b"Not found", "text/plain", 404)

    return Handler


QR_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Join with your phone</title>
<style>body{margin:0;font-family:-apple-system,system-ui,sans-serif;background:#fafaf8;color:#1a1a1a;
display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh;text-align:center}
img{width:min(70vh,80vw);image-rendering:pixelated}h1{font-size:28px;margin:0 0 12px}p{font-size:20px;margin:8px}
code{font-size:22px}</style></head><body><h1>Walk a person round the wheelchair</h1>
<img src="/__CODE__/qr.png" alt="QR code"><p>Scan with your phone's camera, or open</p><code>__ADDRESS__</code>
<p>The phone must be on the same Wi-Fi as this laptop. Hold it sideways: left thumb walks, right thumb looks.</p>
</body></html>"""


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<title>Walk a person</title>
<style>
:root { --you: #8a8a8a; --edge: max(14px, env(safe-area-inset-left), env(safe-area-inset-right)); }
html, body { margin: 0; height: 100%; overflow: hidden; background: #000; color: #fff;
  font-family: -apple-system, system-ui, sans-serif; touch-action: none; -webkit-user-select: none; user-select: none;
  -webkit-touch-callout: none; }
body { position: fixed; inset: 0; }
#view { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
#touch { position: absolute; inset: 0; touch-action: none; }
.hud { position: absolute; pointer-events: none; }
#top { top: 0; left: 0; right: 0; padding: 10px var(--edge) 26px; background: linear-gradient(rgba(0,0,0,.55), rgba(0,0,0,0)); }
#who { display: inline-block; padding: 4px 11px; border-radius: 999px; background: var(--you); font-weight: 700; font-size: 14px; }
#status { margin-top: 5px; max-width: 52%; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  text-shadow: 0 1px 3px rgba(0,0,0,.9); }
#chair { top: 68px; left: 50%; transform: translateX(-50%); display: none; padding: 3px 9px; border-radius: 999px;
  background: rgba(0,0,0,.55); font-size: 13px; font-weight: 700; white-space: nowrap; }
#chair.ahead::after { content: ""; position: absolute; left: 50%; bottom: -7px; margin-left: -6px;
  border: 6px solid transparent; border-top-color: rgba(0,0,0,.55); border-bottom: 0; }
#stick { left: calc(var(--edge) + 70px); top: calc(100% - 84px); width: 120px; height: 120px; margin: -60px 0 0 -60px;
  border-radius: 50%; background: rgba(255,255,255,.14); border: 2px solid rgba(255,255,255,.5); }
#stick.held { background: rgba(255,255,255,.24); }
#knob { position: absolute; left: 50%; top: 50%; width: 52px; height: 52px; margin: -26px 0 0 -26px; border-radius: 50%;
  background: var(--you); border: 2px solid #fff; box-shadow: 0 2px 8px rgba(0,0,0,.4); }
.tip { font-size: 12px; font-weight: 600; color: rgba(255,255,255,.85); text-shadow: 0 1px 3px rgba(0,0,0,.9); transition: opacity .6s; }
#walktip { left: calc(var(--edge) + 140px); bottom: 20px; }
#looktip { right: var(--edge); bottom: 20px; text-align: right; }
#buttons { position: absolute; top: 10px; right: var(--edge); display: flex; gap: 8px; }
button { font: inherit; font-size: 14px; font-weight: 600; padding: 9px 13px; border: 0; border-radius: 10px;
  background: rgba(0,0,0,.5); color: #fff; }
button:active { background: rgba(255,255,255,.35); }
#hint { inset: 0; display: flex; align-items: center; justify-content: center; padding: 0 28%; text-align: center;
  font-size: 18px; font-weight: 600; background: rgba(0,0,0,.5); }
#turn { position: fixed; inset: 0; display: none; flex-direction: column; align-items: center; justify-content: center;
  gap: 18px; padding: 32px; text-align: center; background: #111; font-size: 20px; font-weight: 600; z-index: 5; }
#turn .phone { width: 46px; height: 80px; border: 4px solid #fff; border-radius: 10px; animation: turn 2s ease-in-out infinite; }
#turn small { font-size: 14px; font-weight: 400; color: #bbb; }
@keyframes turn { 0%, 30% { transform: rotate(0); } 60%, 100% { transform: rotate(-90deg); } }
@media (orientation: portrait) { #turn { display: flex; } }
</style></head><body>
<canvas id="view"></canvas>
<div id="touch"></div>
<div id="top" class="hud"><div id="who">Connecting...</div><div id="status"></div></div>
<div id="chair" class="hud"></div>
<div id="stick" class="hud"><div id="knob"></div></div>
<div id="walktip" class="hud tip">Left thumb: walk</div>
<div id="looktip" class="hud tip">Right thumb: drag to look round</div>
<div id="hint" class="hud">Connecting to the laptop...</div>
<div id="buttons"><button id="new">New person</button><button id="letgo">Let go</button></div>
<div id="turn"><div class="phone"></div>Turn your phone sideways<small>If it stays upright, switch off rotation lock</small></div>
<script>
const CODE = "__CODE__";
const LOOK_PER_WIDTH = 2.6;    // radians turned by a drag across the whole screen
const STICK_REACH = 50;        // px from where the left thumb lands to full speed
const DEAD = 0.12;             // the joystick does nothing this close to the middle (share of full)
const OVERSCAN = 1.08;         // the picture is drawn a little bigger than the screen
const KEEP = 12, KEEP_M = 0.8, KEEP_MS = 20000; // pictures kept for turning: so many, taken this near and this recently
const STRIPS = 48;             // the screen is redrawn in this many columns, each turned the right amount
const SKY = "#c4cad0", FLOOR = "#85827c";        // where no picture has been yet
const PITCH = [-0.5, 0.7];     // how far up and down one can look, radians (down is positive)
const EYE_H = 1.6, EYE_FWD = 0.15, CHAIR_H = 0.7;   // eyes (phones.py) and the middle of the chair, metres
let autoDown = 0;              // how far the walker looks down by itself at the chair in front (lookDown)
let slot = null, me = null, view = null, busy = false, actions = [], fov = 1.5;
let move = { f: 0, s: 0 };
let look = 0, lookSent = 0, pitch = 0;          // look: turned to the left in all, radians
let pics = [], picNumber = 0;                    // the newest last: { img, look, pitch, x, y, t }
const $ = id => document.getElementById(id);
const who = $("who"), statusLine = $("status"), hint = $("hint"), chairMark = $("chair");
const canvas = $("view"), ctx = canvas.getContext("2d"), touch = $("touch"), stick = $("stick"), knob = $("knob");
const sleep = ms => new Promise(r => setTimeout(r, ms));

function remembered() { try { return JSON.parse(sessionStorage.getItem("slot")); } catch (e) { return null; } }
function remember(s) { try { sessionStorage.setItem("slot", JSON.stringify(s)); } catch (e) {} }

async function post(path, body) {
  const r = await fetch(`/${CODE}/${path}`, { method: "POST", body: JSON.stringify(body),
                                              headers: { "Content-Type": "application/json" } });
  return r.json();
}

async function join() {
  const r = await post("join", { slot: remembered() });
  if (r.full) { who.textContent = "Four phones are walking already: try again later"; hint.textContent = ""; return; }
  slot = r.slot; me = r; remember(slot);
  look = lookSent = r.look; pitch = r.pitch; fov = r.fov;
  document.documentElement.style.setProperty("--you", r.colour);
  who.textContent = `You walk the person on the ${r.colour_name} ring`;
}

// Thumbs. The first one down on the left half is the joystick, which appears where it lands;
// any other drags to look round, as in phone shooter games.
const thumbs = new Map();
function stickHome() {
  stick.classList.remove("held"); stick.style.left = stick.style.top = ""; knob.style.transform = "";
  move = { f: 0, s: 0 };
}
touch.addEventListener("pointerdown", e => {
  e.preventDefault();
  try { touch.setPointerCapture(e.pointerId); } catch (err) {}
  const walking = [...thumbs.values()].some(t => t.walk);
  if (e.clientX < innerWidth / 2 && !walking) {
    thumbs.set(e.pointerId, { walk: true, x0: e.clientX, y0: e.clientY });
    stick.classList.add("held"); stick.style.left = e.clientX + "px"; stick.style.top = e.clientY + "px";
    $("walktip").style.opacity = 0;
  } else {
    thumbs.set(e.pointerId, { walk: false, x: e.clientX, y: e.clientY });
    $("looktip").style.opacity = 0;
  }
});
touch.addEventListener("pointermove", e => {
  const t = thumbs.get(e.pointerId);
  if (!t) return;
  if (t.walk) {
    let dx = e.clientX - t.x0, dy = e.clientY - t.y0;
    const d = Math.hypot(dx, dy), k = d > STICK_REACH ? STICK_REACH / d : 1;
    dx *= k; dy *= k;
    knob.style.transform = `translate(${dx}px, ${dy}px)`;
    const f = -dy / STICK_REACH, s = -dx / STICK_REACH;
    move = Math.hypot(f, s) < DEAD ? { f: 0, s: 0 } : { f, s };
  } else {
    const per = LOOK_PER_WIDTH / innerWidth;
    look -= (e.clientX - t.x) * per;
    pitch = Math.max(PITCH[0], Math.min(PITCH[1], pitch + (e.clientY - t.y) * per));
    t.x = e.clientX; t.y = e.clientY;
  }
});
function lift(e) {
  const t = thumbs.get(e.pointerId);
  thumbs.delete(e.pointerId);
  if (t && t.walk) stickHome();
}
touch.addEventListener("pointerup", lift);
touch.addEventListener("pointercancel", lift);
$("new").onclick = () => actions.push("new");
$("letgo").onclick = () => actions.push("let_go");

// Phones that allow it (Android) go full screen and sideways at the first tap; iPhones are turned
// by hand. The screen stays on while walking.
document.addEventListener("pointerup", () => {
  const page = document.documentElement;
  if (!document.fullscreenElement && page.requestFullscreen)
    page.requestFullscreen({ navigationUI: "hide" })
      .then(() => screen.orientation && screen.orientation.lock && screen.orientation.lock("landscape")).catch(() => {});
  if (navigator.wakeLock) navigator.wakeLock.request("screen").catch(() => {});
}, { once: true });

// Walking up to the chair, look down at it as anyone would: it is low, and the picture is wide but not
// tall, so straight ahead it would drop out of sight below the screen about a metre away. Eases in
// and out, and only while the chair is in front.
function lookDown() {
  let want = 0;
  if (view && view.you && view.chair) {
    const [x, y, h] = view.you, [cx, cy] = view.chair;
    let side = Math.atan2(cy - y, cx - x) - (h + look - view.look);
    side = Math.atan2(Math.sin(side), Math.cos(side));
    const d = Math.max(0.2, Math.hypot(cx - x, cy - y) - EYE_FWD);
    const halfTall = Math.atan(Math.tan(fov / 2) * 480 / 1056);       // the eye cameras' 1056 x 480 pictures
    if (Math.abs(side) < fov / 2 + 0.2) want = Math.max(0, Math.min(0.9, Math.atan2(EYE_H - CHAIR_H, d) - halfTall / 2));
  }
  autoDown += (want - autoDown) * 0.2;
  return autoDown;
}
const seen = () => Math.max(pitch, autoDown);    // where the walker looks: the thumb's, or down at the chair

async function tick() {
  if (slot === null || busy) return;
  busy = true;
  const sent = actions, turned = look - lookSent;
  actions = []; lookSent = look;
  lookDown();
  try {
    const r = await post("tick", { slot, forward: move.f, side: move.s, look: turned, pitch: seen(), actions: sent });
    if (r.rejoin) { slot = null; await join(); }
    else {
      view = r;
      hint.style.display = r.you ? "none" : "flex";
      hint.textContent = "Touch the left side to walk someone near the wheelchair";
      who.textContent = r.you ? `You walk the person on the ${me.colour_name} ring` : `You are ${me.colour_name}: not walking anyone`;
      statusLine.textContent = r.you ? `Gap to the chair ${Math.max(0, r.gap).toFixed(1)} m. ${r.status}` : "";
    }
  } catch (e) {
    lookSent -= turned; actions = sent.concat(actions);       // send them again next time
    statusLine.textContent = "No connection to the laptop: same Wi-Fi?";
  }
  busy = false;
}

// Pictures: the newest one each time, with how far the walker had turned when it was taken.
async function decode(blob) {
  if (window.createImageBitmap) return createImageBitmap(blob);
  const url = URL.createObjectURL(blob), img = new Image();
  img.src = url;
  await img.decode();
  URL.revokeObjectURL(url);
  return img;
}
async function pictures() {
  for (;;) {
    if (slot === null) { await sleep(300); continue; }
    try {
      const r = await fetch(`/${CODE}/eye/${slot}.jpg?after=${picNumber}`, { cache: "no-store" });
      if (r.status !== 200) { if (r.status !== 204) await sleep(1000); continue; }
      const number = +r.headers.get("X-Frame"), turned = +r.headers.get("X-Look"), down = +r.headers.get("X-Pitch");
      const [x, y] = (r.headers.get("X-At") || "NaN,NaN").split(",").map(Number);
      const img = await decode(await r.blob()), now = performance.now();
      const keep = pics.filter(q => now - q.t < KEEP_MS && Math.hypot(q.x - x, q.y - y) < KEEP_M &&
                                    Math.abs(q.look - turned) + Math.abs(q.pitch - down) > 0.05)  // the new one shows that
                       .slice(-(KEEP - 1));
      for (const q of pics) if (!keep.includes(q) && q.img.close) q.img.close();
      pics = keep.concat([{ img, look: turned, pitch: down, x, y, t: now }]);
      picNumber = number;
    } catch (e) { await sleep(1000); }
  }
}

// Draw the view for where the thumb looks now, although each picture takes a moment to come from
// the laptop: the screen is cut into columns, and each column shows the newest picture that saw
// that direction, turned by what the walker has turned since (a camera turning on the spot moves
// each column sideways and stretches it a little, so near the edges too it lines up with the next
// picture). Where no kept picture saw, sky and floor colours meet at the horizon.
function draw() {
  requestAnimationFrame(draw);
  const dpr = devicePixelRatio || 1, W = Math.round(innerWidth * dpr), H = Math.round(innerHeight * dpr);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  if (!pics.length) { ctx.fillStyle = "#1c1c1c"; ctx.fillRect(0, 0, W, H); return; }
  const iw = pics[0].img.width, ih = pics[0].img.height, k = Math.max(W / iw, H / ih) * OVERSCAN;
  const fs = iw / 2 / Math.tan(fov / 2), F = fs * k, half = fov / 2;   // focal lengths: picture and screen pixels
  const down = seen(), horizon = Math.max(0, Math.min(H, H / 2 - F * Math.tan(down)));
  ctx.fillStyle = SKY; ctx.fillRect(0, 0, W, horizon);
  ctx.fillStyle = FLOOR; ctx.fillRect(0, horizon, W, H - horizon);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";   // a smooth enlargement, not blocky
  for (let i = 0; i < STRIPS; i++) {
    const x0 = Math.round(i * W / STRIPS), x1 = Math.round((i + 1) * W / STRIPS);
    const a0 = Math.atan((x0 - W / 2) / F), a1 = Math.atan((x1 - W / 2) / F);   // to the right of straight ahead
    const use = [];                                  // the newest picture that saw all of it, and any newer that saw some
    for (let j = pics.length - 1; j >= 0; j--) {
      const turn = look - pics[j].look, b0 = a0 - turn, b1 = a1 - turn;    // the same directions in that picture
      if (b1 <= -half || b0 >= half) continue;
      use.push(j);
      if (b0 >= -half && b1 <= half) break;
    }
    for (let n = use.length - 1; n >= 0; n--) {
      const p = pics[use[n]], turn = look - p.look;
      const b0 = Math.max(a0 - turn, -half), b1 = Math.min(a1 - turn, half);
      const d0 = Math.round(W / 2 + F * Math.tan(b0 + turn)), d1 = Math.round(W / 2 + F * Math.tan(b1 + turn));
      if (d1 <= d0) continue;
      const s0 = iw / 2 + fs * Math.tan(b0), s1 = iw / 2 + fs * Math.tan(b1);
      const mid = (b0 + b1) / 2, tall = ih * k * Math.cos(mid) / Math.cos(mid + turn);
      const centre = H / 2 + F * Math.tan(p.pitch - down);
      ctx.drawImage(p.img, s0, 0, Math.max(s1 - s0, 0.5), ih, d0, centre - tall / 2, d1 - d0 + 1, tall);
    }
  }
  markChair(F / dpr);
}

// Where the chair is: above it when in sight, otherwise at the side to turn to. `focal`: the screen's
// focal length in CSS pixels.
function markChair(focal) {
  if (!view || !view.you || !view.chair) { chairMark.style.display = "none"; return; }
  const [x, y, h] = view.you, [cx, cy] = view.chair;
  let side = Math.atan2(cy - y, cx - x) - (h + look - view.look);      // to the left is positive
  side = Math.atan2(Math.sin(side), Math.cos(side));
  const at = Math.abs(side) < 1.4 ? -Math.tan(side) * focal : -Math.sign(side) * 1e4;
  const inSight = Math.abs(at) < innerWidth / 2 - 10;
  const text = `Chair ${Math.max(0, view.gap).toFixed(1)} m`;
  const label = inSight ? text : at < 0 ? `◀ ${text}` : `${text} ▶`;
  chairMark.style.display = "block";
  if (chairMark.textContent !== label) chairMark.textContent = label;
  chairMark.classList.toggle("ahead", inSight);
  const reach = innerWidth / 2 - chairMark.offsetWidth / 2 - 8;       // keep the whole label on the screen
  chairMark.style.left = `calc(50% + ${Math.max(-reach, Math.min(reach, at))}px)`;
}

join().then(() => { setInterval(tick, 50); pictures(); });
requestAnimationFrame(draw);
</script></body></html>
"""
