"""People walked by hand from phones (webots/run.py --phones, webots/phones.py), so that someone at
the demo can test the chair themselves.

Each phone gets a slot, 1 to 4, with its own ring colour. When it joins it takes over a random
walker near the chair, and another one at the start of every trip, until it lets go. Held sideways,
the left thumb walks (forward, back and sideways) and the right thumb turns them to look round;
New person takes over someone else. A coloured ring on the floor marks who is being walked, and
F on the laptop can look through their eyes. The crowd stops steering that person, but to everyone
else, the chair included, they are an ordinary person: the chair's lidar sees them and its people
tracker follows them, so walking into its way shows how the assistant reacts.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from webots.people import PLAYER_BACK_SPEED, PLAYER_SIDE_SPEED, PLAYER_WALK_SPEED


@dataclass(frozen=True)
class Slot:
    number: int
    name: str               # as shown on the screen
    colour: str             # the ring's colour, red green blue
    colour_name: str


SLOTS = (
    Slot(1, "Phone 1", "1 0.5 0", "orange"),
    Slot(2, "Phone 2", "0.55 0.25 1", "purple"),
    Slot(3, "Phone 3", "1 0.85 0", "yellow"),
    Slot(4, "Phone 4", "0.85 0.1 0.1", "red"),
)
RING_Z = 0.02                # just above the sensor view's rings (sensorview.py)
HIDDEN = [0.0, 0.0, -1000.0]
RING_NODE = """DEF PLAYER{n}_RING Pose {{
  translation 0 0 -1000
  children [ Shape {{
    appearance PBRAppearance {{ baseColor {colour} emissiveColor {colour} emissiveIntensity 1 transparency 0.15 roughness 1 metalness 0 }}
    geometry Cylinder {{ height 0.02 radius 0.55 }} }} ]
}}"""


@dataclass
class PhoneInput:
    """One phone's thumbs: the joystick (forward and side each -1 to 1, side to the left), how far
    its walker turned to the left since the last step (look, radians), how far down they look
    (pitch, radians, for their camera in phones.py), and the buttons pressed since the last step
    ("join", "let_go", "new")."""
    forward: float = 0.0
    side: float = 0.0
    look: float = 0.0
    pitch: float = 0.0
    actions: tuple = ()


class Players:
    """Reads the phones every step, moves the walked people and draws their rings."""

    def __init__(self, scene, seed: int = 0):
        self.scene, self.rng = scene, random.Random(seed)
        sup = scene.sup
        children = sup.getRoot().getField("children")
        for s in SLOTS:
            if sup.getFromDef(f"PLAYER{s.number}_RING") is None:
                children.importMFNodeFromString(-1, RING_NODE.format(n=s.number, colour=s.colour))
        self.rings = {s.number: sup.getFromDef(f"PLAYER{s.number}_RING").getField("translation") for s in SLOTS}
        self.keeps: set[int] = set()     # slots that walk someone: a new trip gives them a new walker

    def walker(self, number: int):
        return self.scene.crowd.player(number)

    def update(self, phones: dict[int, PhoneInput]) -> list[str]:
        """Apply this step's thumbs and buttons. Returns what changed, to print."""
        crowd, said = self.scene.crowd, []
        for s in SLOTS:
            person, phone = crowd.player(s.number), phones.get(s.number)
            if phone is None:                  # not connected (any more): hand their walker back
                self.keeps.discard(s.number)
                if person is not None:
                    crowd.let_go(s.number)
                    said.append(f"{s.name}: disconnected, its walker walks on")
                continue
            let_go, new = "let_go" in phone.actions, "new" in phone.actions
            start = "join" in phone.actions or abs(phone.forward) + abs(phone.side) > 0.1
            if let_go:
                self.keeps.discard(s.number)
            elif start or new:
                self.keeps.add(s.number)
            previous = person
            if person is not None and (let_go or new):
                crowd.let_go(s.number)
                said.append(f"{s.name}: let go of the walker, who walks on")
                person = None
            if person is None and s.number in self.keeps:    # a random walker near the chair
                person = crowd.take_over(s.number, self.rng, not_this=previous)
                if person is not None:
                    said.append(f"{s.name}: now walking the person on the {s.colour_name} ring")
                elif start or new:
                    said.append(f"{s.name}: nobody free to walk")
            if person is not None:
                forward = max(-1.0, min(1.0, phone.forward))
                person._player_move = (forward * (PLAYER_WALK_SPEED if forward > 0 else PLAYER_BACK_SPEED), 0.0)
                person._player_side = max(-1.0, min(1.0, phone.side)) * PLAYER_SIDE_SPEED
                if phone.look:                 # the right thumb turns them straight away
                    h = person.orientation + phone.look
                    person.orientation = math.atan2(math.sin(h), math.cos(h))
        self.draw()
        return said

    def draw(self) -> None:
        for s in SLOTS:
            person = self.scene.crowd.player(s.number)
            self.rings[s.number].setSFVec3f([person.x, person.y, RING_Z] if person is not None else HIDDEN)

    def status(self) -> str:
        """One line for the screen: who is walking whom, or how to join."""
        active = [s for s in SLOTS if self.walker(s.number) is not None]
        if not active:
            return "scan the QR code: phones walk people round the chair"
        return "   ".join(f"{s.name}: {s.colour_name}" for s in active)
