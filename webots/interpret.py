"""What the rider meant, from what they said in their own words (voice.py, when it is not an exact
command): a language model on the course's LiteLLM gateway is told what is on the screen right now
(the places the chair has read, the numbered list, the markers up, who is driving) and answers with one
command as JSON. Nothing is trained: the places come with every request, so it works in any building.
Its answer is only taken if the action is a real one and the place, marker or number is on the screen.

The gateway's address and key come from the environment or from G19's .env file (never written in
the code), the model's name from COMP3110_MODEL_NAME or --voice-model:
    COMP3110_MODEL_BASE_URL, COMP3110_API_KEY, COMP3110_MODEL_NAME

    python webots/interpret.py "I need to pick up my prescription"   # one request, the hospital on screen
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
TIMEOUT_S = 3.0                 # the rider waits no longer than this: then only exact commands work
ACTIONS = ("stop", "hand_back", "go", "place", "marker", "number", "page", "none")
MOVES = {"hand_back", "go", "place", "marker", "number"}

SYSTEM = """You turn what a wheelchair rider says into one command for their chair. The rider speaks in
their own words, and speech recognition often writes a name down as other words that sound like it
("car dee ology" for Cardiology, "or tho pedics" for Orthopaedics). Go by what the rider most likely meant.

Answer with one JSON object and nothing else: {"action": ..., "target": ..., "reply": ...}

Actions:
- "stop": the chair stops and the rider steers it themselves ("just a moment", "I'll do it myself").
- "hand_back": the rider, who is driving, gives the driving back to the chair ("you steer from here").
  Only when the rider is driving.
- "go": go to what is PICKED ("set off now"). Only when something is picked.
- "place": go to a place in PLACES. "target" is its name, copied exactly from PLACES.
- "marker": go to a marker in MARKERS. "target" is its name, copied exactly from MARKERS.
- "number": go to an entry of the numbered LIST. "target" is its number, such as "3".
- "page": show the next page of the list.
- "none": anything else: small talk, a question that is not asking to go somewhere, a place that is
  not in PLACES, or a request that fits two places equally well.

Rules:
- A place may be named, or described by what is done or found there. Only choose a place that is in
  PLACES, never one that is not.
- Choose the most specific place in PLACES that fits: a room the chair has read nearby rather than
  the department it is in.
- When the words sound like a name in PLACES or MARKERS, choose that name, even if the words also
  mean something else.
- A number said with a room, bed or ward ("room 12") is part of the place's name, not a LIST number.
- If a request fits two places equally (two meeting rooms when the rider just says "the meeting
  room"), use "none" and ask which.
- "reply" is a few words for the screen: what the chair will do, why it does nothing, or the question.
- "target" is null unless the action is place, marker or number."""


@dataclass
class Screen:
    """What the rider can see and choose from right now."""
    places: list[str] = field(default_factory=list)      # places the chair can be sent to (read so far)
    wings: dict = field(default_factory=dict)            # place: its wing, from the directory
    listed: list[str] = field(default_factory=list)      # the numbered list on the screen (1 to 9)
    markers: list[str] = field(default_factory=list)     # markers up on the walls
    driving: bool = False                                 # the rider has control
    picked: str | None = None                             # the place or marker picked already (G goes there)
    here: str | None = None                               # the department the chair has arrived at, if any


@dataclass
class Meaning:
    """The model's answer once checked."""
    action: str = "none"
    target: str | None = None
    reply: str = ""
    valid: bool = True           # the model answered with a usable JSON command
    invented: bool = False       # it named a place or marker that is not on the screen
    error: str = ""              # the request failed (no network, no key, timed out)
    seconds: float = 0.0
    tokens: tuple[int, int] = (0, 0)   # prompt and answer tokens
    raw: str = ""


def load_env(path: Path = ENV_FILE) -> None:
    """Sets the variables in a .env file (KEY=value or export KEY=value) that are not set already."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        if sep and key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip("'\"")


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def screen_text(screen: Screen, said: str) -> str:
    """The request: what is on the screen, then what the rider said."""
    places = [p + (f" ({screen.wings[p]})" if screen.wings.get(p) else "") for p in screen.places]
    listed = [f"{i}. {n}" for i, n in enumerate(screen.listed, 1)]
    return (f"PLACES: {'; '.join(places) if places else '(none read yet)'}\n"
            f"LIST: {'; '.join(listed) if listed else '(empty)'}\n"
            f"MARKERS: {', '.join(screen.markers) if screen.markers else '(none up)'}\n"
            f"PICKED: {screen.picked or '(nothing)'}\n"
            + (f"The chair is at {screen.here}.\n" if screen.here else "") +
            f"The rider is {'driving' if screen.driving else 'not driving (the chair drives itself)'}.\n"
            f'The rider said: "{said}"')


def first_json(text: str) -> dict | None:
    """The first JSON object in a model's answer (some wrap it in ``` or add words around it)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        found = json.loads(text[start:i + 1])
                        return found if isinstance(found, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def check(answer: dict | None, screen: Screen) -> Meaning:
    """The model's answer as a command the chair can carry out, or "none": the action must be a real
    one and its place, marker or number must be on the screen."""
    if answer is None:
        return Meaning(valid=False, reply="the model's answer could not be read")
    action = str(answer.get("action", "")).strip().lower().replace(" ", "_")
    target = answer.get("target")
    reply = str(answer.get("reply") or "")[:120]
    if action not in ACTIONS:
        return Meaning(valid=False, reply=f"the model answered with no command ({action or 'nothing'})")
    if action == "place":
        name = match(target, screen.places)
        return Meaning("place", name, reply) if name else \
            Meaning(reply=f"{target} is not a place the chair has read", invented=True)
    if action == "marker":
        name = match(target, screen.markers)
        return Meaning("marker", name, reply) if name else Meaning(reply=f"no {target} marker is up", invented=True)
    if action == "number":
        digits = re.findall(r"\d+", str(target))
        n = int(digits[0]) if digits else 0
        if 1 <= n <= len(screen.listed):
            return Meaning("number", str(n), reply)
        return Meaning(reply=f"there is no number {target} on the list", invented=True)
    return Meaning(action, None, reply)


def match(target, options: list[str]) -> str | None:
    """The option the model named: exactly, or the same apart from case, punctuation and a wing in
    brackets ("Pharmacy (Central Wing)")."""
    if not target:
        return None
    want = norm(re.sub(r"\(.*?\)", "", str(target)))
    for option in options:
        if norm(option) == want:
            return option
    return None


class Interpreter:
    """A chat model on the course gateway, asked for one command per request."""

    def __init__(self, model: str | None = None, timeout: float = TIMEOUT_S):
        load_env()
        self.base = os.environ.get("COMP3110_MODEL_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("COMP3110_API_KEY", "")
        self.model = model or os.environ.get("COMP3110_MODEL_NAME", "")
        self.timeout = timeout
        self.json_mode = True               # asked for JSON; turned off for a model that refuses it

    def missing(self) -> str:
        """Why it cannot be used ("" if it can)."""
        if not self.base or not self.key:
            return "no gateway address or key (COMP3110_MODEL_BASE_URL, COMP3110_API_KEY in G19/.env)"
        if not self.model:
            return "no model name (COMP3110_MODEL_NAME in G19/.env, or --voice-model)"
        return ""

    def _post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {self.key}",
                                                  "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def models(self) -> list[str]:
        """Every model the gateway offers."""
        request = urllib.request.Request(self.base + "/models", headers={"Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(request, timeout=max(self.timeout, 20)) as response:
            return sorted(m["id"] for m in json.loads(response.read()).get("data", []))

    def interpret(self, said: str, screen: Screen) -> Meaning:
        """What the rider meant, checked against the screen. On any failure: action "none" with the
        reason in `error`, so the caller falls back to exact commands."""
        if self.missing():
            return Meaning(error=self.missing())
        body = {"model": self.model, "temperature": 0, "max_tokens": 600,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": screen_text(screen, said)}]}
        started = time.monotonic()
        use_json = self.json_mode
        for attempt in range(2):
            if use_json:
                body["response_format"] = {"type": "json_object"}
            else:
                body.pop("response_format", None)
            try:
                out = self._post("/chat/completions", body)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and use_json and attempt == 0:   # this model will not do JSON mode
                    self.json_mode = use_json = False
                    continue
                return Meaning(error=f"the gateway said {exc.code} {exc.reason}", seconds=time.monotonic() - started)
            except Exception as exc:  # noqa: BLE001 (timeout, no network: fall back, never stop the trip)
                return Meaning(error=f"no answer from the model ({exc.__class__.__name__})",
                               seconds=time.monotonic() - started)
        seconds = time.monotonic() - started
        message = (out.get("choices") or [{}])[0].get("message") or {}
        text = message.get("content") or ""
        usage = out.get("usage") or {}
        meaning = check(first_json(text), screen)
        meaning.seconds, meaning.raw = seconds, text[:400]
        meaning.tokens = (usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0)
        return meaning


def hospital_directory() -> list[tuple[str, str]]:
    """The hospital's directory as the chair reads it: [(entry, wing)], Emergency first, as signs.py
    lays it out (each department with an entrance sign, else its places)."""
    sys.path.insert(0, str(ROOT))
    from webots import signs
    places = signs._hospital()
    named = {p.name for p in places}
    groups: dict[str, list] = {}
    for p in places:
        if not p.staff and p.wall:
            groups.setdefault(p.group, []).append(p)
    entrances = [signs.Place(g, g, m[0].colour, 0.0, 0.0, m[0].wall, area=True, entrance=True)
                 for g, m in groups.items() if len(m) >= 2 and g not in named]
    wings = signs.wing_of("fp_hospital")
    return [(name, wings.get(group, "")) for name, group in signs.entries(places + entrances)]


def hospital_screen() -> Screen:
    """The hospital as the open world shows it once its directory is read (for trying a request)."""
    directory = hospital_directory()
    names = [n for n, _ in directory]
    return Screen(places=names, wings=dict(directory), listed=names[:9])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    interpreter = Interpreter()
    if interpreter.missing():
        sys.exit(f"Cannot ask the model: {interpreter.missing()}")
    meaning = interpreter.interpret(" ".join(sys.argv[1:]), hospital_screen())
    print(f"{meaning.action} {meaning.target or ''}  ({meaning.reply})  {meaning.seconds:.2f} s"
          + (f"  error: {meaning.error}" if meaning.error else ""))
