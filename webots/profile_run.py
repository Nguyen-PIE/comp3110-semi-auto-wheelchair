"""Where the time of each run.py step goes, on this computer: run.py as usual (any options), timing
each part of every 0.1 s step, then quitting after a number of steps and printing the time per step
for each part. Nothing in run.py changes; the parts are timed from the outside.

    python webots/profile_run.py fp_hospital --open-world            # 300 steps, the Webots window open
    python webots/profile_run.py fp_hospital --steps 600 --no-trail  # any run.py options after the world
    python webots/profile_run.py fp_hospital --open-world --camera-every 0.4   # the sign camera read less often
    python webots/profile_run.py fp_hospital --open-world --camera-every off   # ... or not at all (signs unread)
    python webots/profile_run.py fp_hospital --open-world --steps 6000 --every 500 --drive   # 10 minutes: does it slow down?

--drive (open world): the chair keeps driving round the building on its own. Whenever it stands still
it is sent on, as a rider would with 1-9 and G: to the directory first, then to each place read in
turn. Without it the chair stands still at the start until G, so a long run measures a chair that
never moves.
--override: as --drive, but the rider takes over (O) for 20 s of every 30 and drives with the arrow
keys: ahead, turning now and then, and every other time straight at the nearest person
(--at-people: every time). The lines say how many people were hit while the rider drove: with
run.py's --no-brake after the world, the chair only warns and does not slow down for them.
--picture FILE.jpg: the 3D view saved at the last step, to check what is drawn.

The Webots window closes by itself after the steps (300 by default, about half a minute): that is
the end of the measurement. The table is printed and added to data/traces/profile_run.txt.
A step that takes more than 100 ms makes the trip slower than real time. "Webots" is the simulation
and drawing, "waiting for Webots" is time Python spends on a call that must wait for Webots to finish
drawing (such as changing the blue trail).
"""

from __future__ import annotations

import collections
import functools
import math
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TIMES: dict[str, float] = collections.defaultdict(float)
CALLS: collections.Counter = collections.Counter()
IDLE = ("start", "waiting", "arrived", "done", "at a marker")    # the open world chair standing still
OVERRIDE_EVERY, OVERRIDE_FOR = 300, 200                          # --override: 20 s of every 30
SHOWN = ("open world: choosing where to go", "open world: adding the lidar to its map", "open world: its own map",
         "people in the way (rider driving)",
         "status text", "list text", "red aiming line", "sensor view", "assistant (policy and tracker)",
         "moving the people's models")


def timed(owner, name: str, label: str) -> None:
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            TIMES[label] += time.perf_counter() - started
            CALLS[label] += 1
    setattr(owner, name, wrapper)


def main(argv: list[str]) -> None:
    steps, warm_up, camera, every = 300, 20, None, 0
    at_people = "--at-people" in argv
    picture = None
    if "--picture" in argv:                               # the 3D view saved at the last step, to check what is drawn
        i = argv.index("--picture")
        picture = Path(argv[i + 1]).resolve()
        argv = argv[:i] + argv[i + 2:]
    override = "--override" in argv or at_people
    drive = "--drive" in argv or override
    argv = [a for a in argv if a not in ("--drive", "--override", "--at-people")]
    if "--steps" in argv:
        i = argv.index("--steps")
        steps = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if "--every" in argv:                                 # a line every this many steps, to see it change over time
        i = argv.index("--every")
        every = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if "--camera-every" in argv:
        i = argv.index("--camera-every")
        camera = 1e6 if argv[i + 1] == "off" else float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    from assist import driver as assist_driver
    from webots import bridge, open_world, people, run, sensorview, trackview, vision
    if camera is not None:                                # for this measurement only: how much the camera costs
        reader_init = vision.SignReader.__init__

        def reader_every(self, *args, **kwargs):
            kwargs["every_s"] = camera
            reader_init(self, *args, **kwargs)
        vision.SignReader.__init__ = reader_every
    timed(bridge.WebotsScene, "step", "the whole step (scene.step)")
    timed(bridge.WebotsScene, "_advance", "  Webots: simulating and drawing")
    timed(bridge.WebotsScene, "_extend_trail", "  blue trail (waiting for Webots)")
    timed(bridge.WebotsScene, "_draw_people", "  moving the people's models")
    timed(people.Crowd, "step", "  people's walking rules")
    timed(people.Crowd, "stream", "  people appearing and leaving")
    timed(assist_driver._AdjustedRouteDriver, "act", "assistant (policy and tracker)")
    timed(open_world.OpenWorld, "step", "open world (everything in its step)")
    timed(vision.SignReader, "look", "  open world: reading the camera picture")
    timed(open_world.Trail, "add", "  open world: blue trail")
    trails = []                                           # the open world's own trail, to count its points
    trail_init = open_world.Trail.__init__

    def trail_made(self, *args, **kwargs):
        trail_init(self, *args, **kwargs)
        trails.append(self)
    open_world.Trail.__init__ = trail_made
    worlds = []                                           # the open world, to send the chair on (--drive)
    world_init = open_world.OpenWorld.__init__

    def world_made(self, *args, **kwargs):
        world_init(self, *args, **kwargs)
        worlds.append(self)
    open_world.OpenWorld.__init__ = world_made
    timed(run.Views, "update", "camera views")
    timed(sensorview.SensorView, "update", "sensor view")
    timed(trackview.TrackView, "update", "track view")
    timed(run.Overlay, "lines", "status text")
    timed(run.Overlay, "legend", "list text")
    timed(open_world.Session, "_decide", "  open world: choosing where to go")
    timed(open_world.Session, "_scan", "  open world: adding the lidar to its map")
    timed(open_world.Session, "_use_map", "  open world: its own map")
    from webots import landmarks
    timed(landmarks.Aim, "update", "red aiming line")
    guards = []                                           # people hit while the rider drove
    try:
        from webots import guard
    except ImportError:                                   # before guard.py
        guard = None
    if guard is not None:
        timed(guard.Guard, "check", "people in the way (rider driving)")
        guard_init = guard.Guard.__init__

        def guard_made(self, *args, **kwargs):
            guard_init(self, *args, **kwargs)
            guards.append(self)
        guard.Guard.__init__ = guard_made
    count = {"steps": 0, "started": None, "block": None, "times": {}, "sent": 0}

    def send_on() -> set:
        """--drive: G (after picking the next place read, once there are some) when the chair stands still."""
        ow = worlds[-1]
        if ow.session.mode not in IDLE:
            return set()
        places = [p for p in ow.places() if p != ow.session.target]
        if places:
            ow.pick(places[count["sent"] % len(places)])
        count["sent"] += 1
        return {ord("G")}
    import random
    rng = random.Random(1)

    def ride(scene) -> set:
        """--override: O to take over and to hand back, and the arrow keys in between."""
        from controller import Keyboard
        t = (count["steps"] - warm_up) % OVERRIDE_EVERY
        if t in (0, OVERRIDE_FOR):
            return {ord("O")}
        if t > OVERRIDE_FOR:
            return set()
        if t == 1:                                        # how this override goes
            count["at_people"] = at_people or ((count["steps"] - warm_up) // OVERRIDE_EVERY) % 2 == 1
        r = scene.robot
        if count.get("at_people") and scene.crowd.everyone:
            p = min(scene.crowd.everyone, key=lambda p: math.dist((p.x, p.y), (r.x, r.y)))
            a = math.atan2(p.y - r.y, p.x - r.x) - r.orientation
            a = math.atan2(math.sin(a), math.cos(a))
            return {Keyboard.UP} | ({Keyboard.LEFT} if a > 0.2 else {Keyboard.RIGHT} if a < -0.2 else set())
        if t % 30 == 1:                                   # a new move every 3 s
            count["move"] = rng.choice([{Keyboard.UP}] * 3 + [{Keyboard.UP, Keyboard.LEFT}, {Keyboard.UP, Keyboard.RIGHT},
                                                                {Keyboard.LEFT}])
        return set(count.get("move", ()))
    original_keys = bridge.WebotsScene.keys
    out = ROOT / "data" / "traces" / "profile_run.txt"

    def keys(self):
        count["steps"] += 1
        if count["steps"] == warm_up:                     # the start (loading, the first pictures) left out
            count["started"] = count["block"] = time.perf_counter()
            TIMES.clear()
            CALLS.clear()
        done = count["steps"] - warm_up
        if every and done > 0 and done % every == 0:      # how this stretch went, by part
            now, before = time.perf_counter(), count["times"]
            parts = {k: (v - before.get(k, 0.0)) / every * 1000 for k, v in TIMES.items()}
            if trails:                                    # the open world draws its own trail, in pieces
                t = trails[-1]
                trail_ms = parts.get("  open world: blue trail", 0)
                trail = getattr(t, "drawn", None) or (len(t.pieces) - 1) * open_world.TRAIL_PIECE + getattr(t, "count", 0) if t.pieces else 0
            else:
                trail_ms = parts.get("  blue trail (waiting for Webots)", 0)
                trail = self.trail_points.getCount() if getattr(self, "trail_points", None) is not None else 0
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss     # bytes on a Mac, kB on Linux
            peak_mb = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
            line = (f"after {done:5d} steps: {(now - count['block']) / every * 1000:4.0f} ms a step | Webots "
                    f"{parts.get('  Webots: simulating and drawing', 0):3.0f} | trail {trail_ms:4.1f} ({trail} points)"
                    f" | open world "
                    f"{parts.get('open world (everything in its step)', 0) - parts.get('the whole step (scene.step)', 0):4.1f}"
                    f" | peak memory {peak_mb:.0f} MB")
            if worlds:                                    # what the chair has done, so the line shows it moved
                session = worlds[-1].session
                line += f" | {session.mode}, {len(session.reader.seen)} signs seen, {len(worlds[-1].places())} places known"
            line += f" | {len(self.crowd.everyone)} people" + (f", {guards[-1].hits} hit" if guards else "")
            line += "".join(f" | {k.strip()} {v:.1f}" for k, v in parts.items()
                            if k.strip() in SHOWN and v >= 0.05)
            print(line, flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a") as f:
                f.write(line + "\n")
            count["block"], count["times"] = now, dict(TIMES)
        held = original_keys(self)
        if picture is not None and count["steps"] == warm_up + steps - 1:
            self.sup.exportImage(str(picture), 90)
        riding = override and count["steps"] >= warm_up and (count["steps"] - warm_up) % OVERRIDE_EVERY <= OVERRIDE_FOR
        if override and count["steps"] >= warm_up:
            held = held | ride(self)
        if drive and worlds and count["steps"] % 10 == 0 and not riding:    # once a second, like a rider would
            held = held | send_on()
        return held | {ord("Q")} if count["steps"] >= warm_up + steps else held
    bridge.WebotsScene.keys = keys
    try:
        run.main(argv)
    finally:
        if count["started"] is None:
            print("Stopped before the timing started")
            return
        n = max(1, count["steps"] - warm_up)
        wall = time.perf_counter() - count["started"]
        lines = [f"{n} steps of 0.1 s in {wall:.1f} s: {wall / n * 1000:.0f} ms a step, "
                 f"{0.1 * n / wall:.2f} times real time ({' '.join(argv)}" + (", overriding" if override else ", driving" if drive else "")
                 + (f", camera every {camera:g} s" if camera is not None else "") + f", {time.strftime('%Y-%m-%d %H:%M')})"]
        lines += [f"  {seconds / n * 1000:7.1f} ms a step  {label}" for label, seconds in TIMES.items()]
        if guards and override:
            lines += [f"  people hit while the rider drove: {guards[-1].hits}"
                      + ("" if guards[-1].brake else " (--no-brake: warnings only)")]
        print("\n" + "\n".join(lines))
        out = ROOT / "data" / "traces" / "profile_run.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main(sys.argv[1:])
