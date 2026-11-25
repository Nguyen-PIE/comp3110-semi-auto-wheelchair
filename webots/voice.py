"""Voice commands for the rider (always on in run.py): hold K, say a command, let go.

Speech is recognised on the laptop, offline, by Vosk's small English model (bash
webots/get_voice_model.sh, once), twice: once listening only for the exact commands below (with the
places the chair has read and the markers up), and once writing down whatever was said. An exact
command that both agree on is carried out at once. Anything else, in the rider's own words ("I need
my prescription"), goes to a language model on the course gateway (interpret.py), which picks one
command from what is on the screen, and the chair asks before it moves ("Pharmacy? Say yes or press
G"). "stop", "wait" and "hold on" anywhere stop the chair at once, without the gateway. A command does
what its key does, so nothing here changes how the chair drives.

    stop / stop the chair / wait          take control, so the chair stops at once      (O)
    take control / let me drive           the same
    hand back / you drive                 the assistant drives again                     (O)
    go ahead / carry on                   go to the place or marker picked               (G)
    go to <place> / take me to <place>    open world: a place on the directory or read nearby
    go to the <colour> <shape>            a marker on the wall (--landmarks, and always in the open world)
    go to the <colour> one                the same: each marker up has a colour of its own
    go to number <1 to 9>                 the place or marker with that number on the screen (1-9, G)
    next page                             the next page of the open world's list         (P)
    yes / no                              after the chair asks: go there / do not

    python webots/voice.py --test         check the recogniser on phrases spoken by the computer (macOS)
    python webots/voice.py --mic          check the microphone: say a command, see how loud it was and what was heard
"""

from __future__ import annotations

import collections
import json
import math
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "vosk-model-small-en-us-0.15"
RATE = 16000                    # samples a second, mono, 16-bit: what the model was trained on
PUSH_TO_TALK = ord("K")         # held while speaking (T is the tracks view)
PRE_ROLL_S = 0.3                # sound kept from just before K, so the first word is not cut
RELEASE_S = 0.25                # K counts as let go once it has been up this long (a key can flicker while held)
QUIET_DB = -45                  # loudest sound below this: the microphone hardly heard the rider
MIN_CONFIDENCE = 0.7            # a command that moves the chair needs every word heard this surely
SHOW_S = 6.0                    # how long what was heard stays on the screen
CONFIRM_S = 15.0                # how long the chair waits for yes or no after asking

COMMANDS = {                    # phrase: action (the keys each action stands for are in VoiceControl)
    "stop": "stop", "stop the chair": "stop", "wait": "stop",
    "take control": "take", "let me drive": "take",
    "hand back": "hand_back", "you drive": "hand_back",
    "go ahead": "go", "carry on": "go",
    "next page": "page",
}
CONFIRM = {"yes": "yes", "yes please": "yes", "yeah": "yes", "okay": "yes",   # only while the chair is asking
           "no": "no", "no thanks": "no", "cancel": "no"}
MOVES = {"hand_back", "go", "place", "marker", "number", "yes"}   # these set the chair moving: heard surely
STOP_WORDS = ("stop", "wait", "hold on", "hang on", "halt", "whoa", "careful")   # anywhere in what was said
NOT_WORDS = {"not", "don't", "dont", "never", "isn't", "doesn't", "wouldn't", "shouldn't", "can't", "cannot"}
FILLER = {"the", "please", "um", "uh", "er", "ah", "oh", "now", "just", "a"}
LONGER = 3                      # words written down beyond the exact command's: a sentence, not the command
O, G, P = ord("O"), ord("G"), ord("P")

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


class VoiceUnavailable(RuntimeError):
    """Voice commands cannot start here (no package, no model or no microphone): the reason says why."""


def number_words(n: int) -> str:
    """How a number on a sign is said: 4 -> "four", 21 -> "twenty one", 100 and up digit by digit."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    return " ".join(_ONES[int(d)] for d in str(n))


def spoken(name: str) -> str:
    """How a name on a sign is said, in the recogniser's lower-case words: numbers and short capital
    abbreviations are spelt out ("CT Scan" -> "c t scan", "4B Room 3" -> "four b room three",
    "A&E" -> "a and e", "Cafe" stays "cafe"). Apostrophes go: "Nurses' Station" -> "nurses station"."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = text.replace("&", " and ").replace("'", "")
    words: list[str] = []
    for token in re.findall(r"[A-Za-z]+|\d+", text):        # "4B" -> "4" and "B", "X-Ray" -> "X" and "Ray"
        if token.isdigit():
            words.append(number_words(int(token)))
        elif token.isupper() and len(token) <= 3:
            words.extend(token.lower())                        # CT, MRI, a single letter: letter by letter
        else:
            words.append(token.lower())
    return " ".join(words)


def normal(text: str) -> str:
    """Recogniser output as the phrase table writes it: lower case, single spaces."""
    return " ".join(text.lower().split())


def loudest_db(pcm: bytes) -> float:
    """The loudest moment of 16-bit sound, in dB below the loudest possible (0): speech near a laptop
    is about -30 to -5, a quiet room about -60. -120 for silence."""
    import numpy as np
    x = np.frombuffer(pcm, dtype=np.int16)
    peak = int(np.abs(x.astype(np.int32)).max()) if len(x) else 0
    return 20 * math.log10(peak / 32768) if peak else -120.0


def resample(pcm: bytes, rate: int) -> bytes:
    """16-bit mono sound at `rate` as 16 kHz."""
    if rate == RATE or not pcm:
        return pcm
    import numpy as np
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    n = int(len(x) * RATE / rate)
    x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(x, -32768, 32767).astype(np.int16).tobytes()


@dataclass
class Command:
    action: str                   # stop, take, hand_back, go, page, place, marker or number
    target: str | None = None     # the place or marker, named as on the screen
    heard: str = ""               # what the recogniser returned
    confidence: float = 1.0       # its least sure word, 0 to 1


def sure(command: Command | None) -> bool:
    """Whether a command is acted on: it must be a whole phrase, and one that sets the chair moving
    must be heard surely. Stopping never waits."""
    return command is not None and (command.action not in MOVES or command.confidence >= MIN_CONFIDENCE)


class Phrases:
    """Everything the rider can say right now, each with the command it means: the commands, the places
    the chair has read, the markers up on the walls and, when there is a numbered list on the screen,
    its numbers."""

    def __init__(self, places: list[str] = (), markers: list[str] = (), numbered: bool = False,
                 asking: bool = False):
        self.table: dict[str, Command] = {phrase: Command(action) for phrase, action in COMMANDS.items()}
        if asking:                     # the chair has asked "Pharmacy? Say yes": yes and no mean something
            self.table.update({phrase: Command(action) for phrase, action in CONFIRM.items()})
        if numbered:
            for n in range(1, 10):
                self.table[f"go to number {number_words(n)}"] = Command("number", str(n))
        for name in places:
            said = spoken(name)
            if said:
                for lead in ("go to", "take me to"):
                    self.table.setdefault(f"{lead} {said}", Command("place", name))
        colours = [m.split()[0] for m in markers]
        for name in markers:          # "yellow sphere"
            said = spoken(name)
            for lead in ("go to the", "go to"):
                self.table.setdefault(f"{lead} {said}", Command("marker", name))
            colour = name.split()[0]
            if colours.count(colour) == 1:        # "go to the yellow one": the colour alone names it
                for end in ("one", "marker"):
                    self.table.setdefault(f"go to the {colour} {end}", Command("marker", name))

    def phrases(self) -> list[str]:
        return sorted(self.table)

    def match(self, heard: str, confidence: float = 1.0) -> Command | None:
        """The command a recogniser result means, or None (nothing heard, or not a whole phrase)."""
        said = normal(heard)
        found = self.table.get(said)
        if found is None and said.split()[:1] == ["stop"]:
            found = Command("stop")      # "stop it", "stop now": stopping is always safe, so any stop will do
        if found is None:
            return None
        return Command(found.action, found.target, heard=said, confidence=confidence)


class Recogniser:
    """Vosk with its small English model, listening only for the phrases given."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        try:
            import vosk
        except ImportError as exc:
            raise VoiceUnavailable("the vosk package is missing: pip install vosk sounddevice") from exc
        if not (Path(model_dir) / "am").exists():
            raise VoiceUnavailable(f"no speech model in {model_dir}: run bash webots/get_voice_model.sh once")
        vosk.SetLogLevel(-1)
        self.vosk, self.model = vosk, vosk.Model(str(model_dir))

    def recognise(self, pcm: bytes, phrases: list[str]) -> tuple[str, float]:
        """The phrase heard in `pcm` (16 kHz mono 16-bit), and how sure the recogniser is of its least
        sure word. "" when it heard none of the phrases."""
        rec = self.vosk.KaldiRecognizer(self.model, RATE, json.dumps(list(phrases) + ["[unk]"]))
        rec.SetWords(True)
        rec.AcceptWaveform(pcm)
        out = json.loads(rec.FinalResult())
        words = [w for w in out.get("result", []) if w.get("word") != "[unk]"]
        if not words or "[unk]" in out.get("text", ""):
            return "", 0.0
        return out.get("text", ""), min(w.get("conf", 0.0) for w in words)


    def transcribe(self, pcm: bytes) -> str:
        """Whatever was said in `pcm`, written down with the model's whole English vocabulary."""
        rec = self.vosk.KaldiRecognizer(self.model, RATE)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get("text", "")


def says_stop(text: str) -> bool:
    """Whether a stop word is anywhere in what was said ("hang on a second", "wait for me")."""
    padded = f" {normal(text)} "
    return any(f" {w} " in padded for w in STOP_WORDS)


def agrees(phrase: str, written: str) -> bool:
    """Whether the words written down fit the exact command heard: not a whole sentence around it
    ("what time does the cafe close tonight" for "go to cafe") and not saying no to it ("I don't want
    to go to pharmacy"). They need not match word for word: the free pass often mishears ("go to see t
    scan"), and with speech read out by espeak, comparing word for word turned away 60% of good
    commands (voice_eval.py), this 7%."""
    a = [w for w in normal(phrase).split() if w not in FILLER]
    b = [w for w in normal(written).split() if w not in FILLER]
    if NOT_WORDS & set(b) and not NOT_WORDS & set(a):
        return False
    return len(b) < len(a) + LONGER


@dataclass
class Understood:
    """What one recording was taken to mean."""
    command: Command | None = None   # what to do, or None
    how: str = ""                    # exact, stop word, model, or "" (nothing)
    heard: str = ""                  # the exact pass's phrase
    written: str = ""                # the words as said (the free pass)
    confidence: float = 0.0
    meaning: object = None           # the model's answer (interpret.Meaning), if it was asked
    seconds: float = 0.0


def understand(recogniser, pcm: bytes, phrases: Phrases, interpreter=None, screen=None, clock=time.monotonic) -> Understood:
    """One recording, as voice commands take it: an exact command both passes agree on; else a stop
    word anywhere; else, with a model, what it makes of the words; else nothing."""
    started = clock()
    heard, confidence = recogniser.recognise(pcm, phrases.phrases())
    command = phrases.match(heard, confidence)
    out = Understood(heard=heard, confidence=confidence)
    if command is not None and command.action in ("stop", "take", "no"):       # never held back, never waits
        out.command, out.how, out.seconds = command, "exact", clock() - started
        return out
    written = out.written = recogniser.transcribe(pcm) if hasattr(recogniser, "transcribe") else ""
    if says_stop(written):
        out.command, out.how = Command("stop", heard=written), "stop word"
    elif sure(command) and (not written or agrees(heard, written)):
        out.command, out.how = command, "exact"
    elif interpreter is not None and written:
        meaning = interpreter.interpret(written, screen)
        out.meaning = meaning
        if not meaning.error and meaning.action != "none":
            action = "take" if meaning.action == "stop" else meaning.action
            out.command, out.how = Command(action, meaning.target, heard=written), "model"
    else:
        out.command = command                     # unsure or not a command: said so on the screen
    out.seconds = clock() - started
    return out


class Microphone:
    """The laptop's microphone, mono. Sound is kept only while K is held, plus the moment before it,
    so the first word is not cut. It records at 16 kHz, or at the microphone's own rate if it cannot
    (end() gives 16 kHz either way)."""

    def __init__(self, device=None):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise VoiceUnavailable("the sounddevice package is missing: pip install vosk sounddevice") from exc
        self.lock, self.recording, self.chunks = threading.Lock(), False, []
        self.before = collections.deque(maxlen=max(1, round(PRE_ROLL_S * 10)))   # 0.1 s blocks
        try:
            info = sd.query_devices(device, "input")
            self.name = info["name"]
            rates = list(dict.fromkeys((RATE, int(info["default_samplerate"]))))
            for i, rate in enumerate(rates):
                try:
                    self.stream = sd.RawInputStream(samplerate=rate, channels=1, dtype="int16", blocksize=rate // 10,
                                                    device=device, callback=self._heard)
                    self.rate = rate
                    break
                except Exception:  # noqa: BLE001 (this rate is not offered: try the microphone's own)
                    if i == len(rates) - 1:
                        raise
            self.stream.start()
        except Exception as exc:  # noqa: BLE001 (PortAudio raises its own types)
            raise VoiceUnavailable(f"no microphone ({exc}). On macOS, allow Terminal to use it in "
                                   "System Settings > Privacy & Security > Microphone") from exc

    def _heard(self, data, frames, when, status) -> None:
        with self.lock:
            (self.chunks if self.recording else self.before).append(bytes(data))

    def begin(self) -> None:
        with self.lock:
            self.chunks, self.recording = list(self.before), True

    def end(self) -> bytes:
        with self.lock:
            self.recording = False
            pcm, self.chunks = b"".join(self.chunks), []
        return resample(pcm, getattr(self, "rate", RATE))

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


def markers_up(marks) -> list[str]:
    """The names of the markers on the walls now."""
    return [lm.name for lm in marks.items] if marks is not None else []


def numbers_are_markers(manual: bool, marks, world_open) -> bool:
    """Whether 1 to 9 pick a marker now (as the keys do): while the rider drives with markers, or
    with --landmarks outside the open world. Otherwise they pick from the open world's list."""
    return marks is not None and (manual or world_open is None)


class VoiceControl:
    """Push to talk for run.py. keys() is called every step with the keys held: when K is let go, what
    was said is worked out (in the background, so the trip goes on) and comes back as the keys it
    stands for, with the line to show on the screen. A command the model worked out from the rider's
    own words is asked about first ("Pharmacy? Say yes or press G")."""

    def __init__(self, recogniser, microphone, key: int = PUSH_TO_TALK, clock=time.monotonic, interpreter=None):
        self.recogniser, self.mic, self.key, self.clock = recogniser, microphone, key, clock
        self.interpreter = interpreter
        self.talking, self.pending, self.last_held = False, [], -1e9
        self.asking: tuple[Command, float, str] | None = None    # the command asked about, when, the question
        self.busy = 0                                            # recordings still being worked out
        self.marker_go = False                                   # G this step is for the marker picked by voice
        self.jobs: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.message, self.message_at = "", -1e9
        self.log: list[tuple[str, str, float, float]] = []   # heard, action, confidence, seconds to work out
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        while True:
            pcm, phrases, screen = self.jobs.get()
            try:
                result = understand(self.recogniser, pcm, phrases, self.interpreter, screen, self.clock)
            except Exception as exc:  # noqa: BLE001 (keep the trip going whatever the recogniser does)
                result = Understood()
                print(f"Voice: recogniser failed: {exc}", flush=True)
            silent = not pcm.strip(b"\0")    # only zeros: macOS gives these when Terminal may not use the microphone
            self.results.put((result, silent, loudest_db(pcm)))

    def _say(self, text: str) -> None:
        self.message, self.message_at = text, self.clock()
        print(text, flush=True)

    def line(self) -> str:
        """The voice line for the status rows."""
        if self.talking:
            return "Voice: listening... let go of K when you have said it"
        if self.busy:
            return "Voice: working out what you said..."
        if self.clock() - self.message_at < SHOW_S:
            return self.message
        if self.asking is not None:
            return self.asking[2]
        return "Voice: hold K and say what to do, such as stop, hand back or go to the pharmacy"

    def _screen(self, manual: bool, marks, world_open, markers: list[str]):
        """What is on the screen now, for the model."""
        sys.path.insert(0, str(ROOT))
        from webots.interpret import Screen
        selected = getattr(getattr(marks, "selector", None), "selected", None)
        marker = selected.name if selected is not None and markers else None
        if world_open is None:
            return Screen(listed=markers, markers=markers, driving=manual, picked=marker)
        session = getattr(world_open, "session", None)
        wings = session.directory() if session is not None else {}
        listed = markers if numbers_are_markers(manual, marks, world_open) else world_open.names()
        picked = getattr(world_open, "picked", None)
        here = session.department if session is not None and getattr(session, "mode", "") == "arrived" else None
        return Screen(places=world_open.places(), wings=wings, listed=listed, markers=markers, driving=manual,
                      picked=marker if markers and manual else picked if isinstance(picked, str) else None,
                      here=here if isinstance(here, str) else None)

    def keys(self, held: set, *, manual: bool, marks=None, world_open=None) -> set:
        """The keys the rider's voice stands for this step."""
        out: set = set()
        if self.asking is not None and (G in held or self.clock() - self.asking[1] > CONFIRM_S):
            self.asking = None                 # G pressed (it goes, the place is picked), or no answer
        if self.pending:                   # the second half of a command: handed back last step, now go
            command = self.pending.pop(0)
            if manual:                     # O did nothing (paused)
                self._say(f"Voice: paused, so the chair did not go to {command.target}. Space resumes")
            else:
                out |= self._act(command, manual=manual, marks=marks, world_open=world_open)
        if self.key in held:
            self.last_held = self.clock()
        talking = self.key in held or (self.talking and self.clock() - self.last_held < RELEASE_S)
        if talking and not self.talking:
            self.mic.begin()
            print("Voice: listening (let go of K when you have said it)", flush=True)
        elif self.talking and not talking:
            markers = markers_up(marks)
            phrases = Phrases(world_open.places() if world_open is not None else [], markers,
                              numbered=world_open is not None or bool(markers), asking=self.asking is not None)
            pcm = self.mic.end()
            print(f"Voice: K let go, {len(pcm) / 2 / RATE:.1f} s of sound, loudest {loudest_db(pcm):.0f} dB", flush=True)
            self.busy += 1
            self.jobs.put((pcm, phrases, self._screen(manual, marks, world_open, markers)))
        self.talking = talking
        while not self.results.empty():
            result, silent, loudest = self.results.get()
            self.busy = max(0, self.busy - 1)
            command = result.command
            self.log.append((result.written or result.heard, command.action if command else "",
                             result.confidence, result.seconds))
            if result.written or result.heard:
                print(f"Voice: exact pass '{result.heard}' ({result.confidence:.0%}), words '{result.written}', "
                      f"{result.seconds:.2f} s" + (f", model: {result.meaning.action} {result.meaning.target or ''}"
                                                    f" {result.meaning.error}" if result.meaning else ""), flush=True)
            if silent:
                self._say("Voice: the microphone gives only silence. On macOS allow Terminal in System Settings > "
                          "Privacy & Security > Microphone, then start again")
            elif result.how == "model":
                out |= self._model_said(command, manual=manual, marks=marks, world_open=world_open)
            elif result.how:
                out |= self._exact(command, manual=manual, marks=marks, world_open=world_open)
            elif result.meaning is not None:          # the model was asked: it found nothing to do, or failed
                meaning = result.meaning
                self._say(f"Voice: heard '{result.written}', but {meaning.error}. Exact commands still work"
                          if meaning.error else f"Voice: heard '{result.written}': {meaning.reply or 'nothing to do'}")
            elif command is None and loudest < QUIET_DB:
                self._say(f"Voice: the microphone hardly heard you (loudest {loudest:.0f} dB). Speak up, or check "
                          "the input in System Settings > Sound")
            elif command is None:
                said = result.written or result.heard
                self._say("Voice: didn't catch that. Hold K and try again" if not said
                          else f"Voice: heard '{said}', which is not a command here")
            elif not sure(command):
                self._say(f"Voice: not sure I heard '{result.heard}' ({result.confidence:.0%}). Say it again")
            else:                                     # the two passes disagree, and there is no model to ask
                self._say(f"Voice: heard '{result.written}', not sure it was '{result.heard}'. Say it again")
        return out

    def _exact(self, command: Command, *, manual: bool, marks, world_open) -> set:
        """An exact command, a stop word, or yes and no to a question."""
        if command.action == "yes":
            if self.asking is None:
                self._say("Voice: nothing to say yes to")
                return set()
            asked, self.asking = self.asking[0], None
            return self._act(asked, manual=manual, marks=marks, world_open=world_open)
        if command.action == "no":
            if self.asking is not None:
                self.asking = None
                self._say("Voice: OK, the chair stays as it is")
            return set()
        self.asking = None                            # anything else replaces the question
        return self._act(command, manual=manual, marks=marks, world_open=world_open)

    def _model_said(self, command: Command, *, manual: bool, marks, world_open) -> set:
        """A command the model worked out. Stopping and paging are done at once; a move is asked about,
        with the place or marker picked already, so G (or yes) goes there."""
        said = f"Voice: '{command.heard}'"
        if command.action in ("take", "page"):
            self.asking = None
            return self._act(command, manual=manual, marks=marks, world_open=world_open)
        markers, numbered_markers = markers_up(marks), numbers_are_markers(manual, marks, world_open)
        if command.action == "number":                # the list on the screen now: a marker while driving
            listed = markers if numbered_markers else world_open.names() if world_open is not None else []
            name = listed[int(command.target) - 1] if int(command.target) <= len(listed) else None
            if name is None:
                self._say(f"{said}: there is no number {command.target} on the screen")
                return set()
            command = Command("marker" if numbered_markers else "place", name, command.heard)
        g_goes = numbered_markers if command.action == "marker" else not (manual and marks is not None)
        g = " or press G" if g_goes else ""         # G is the list's or the markers', as the keys are
        if command.action == "place":
            if world_open is None or not world_open.pick(command.target):
                self._say(f"{said}: {command.target} is not on the list")
                return set()
            question = f"{said} -> {command.target}? Say yes{g}"
        elif command.action == "marker":
            if command.target not in markers:
                self._say(f"{said}: that marker is not up any more")
                return set()
            marks.selector.select(markers.index(command.target))
            question = f"{said} -> the {command.target}? Say yes{g}"
        elif command.action == "go":
            question = f"{said} -> go? Say yes or press G"
        else:                                         # hand back
            question = f"{said} -> let the chair drive? Say yes or press O"
        self.asking = (command, self.clock(), question)
        self._say(question)
        return set()

    def _act(self, command: Command, *, manual: bool, marks, world_open) -> set:
        heard = f"Voice: heard '{command.heard}'" if command.heard else "Voice"
        action = command.action
        if action in ("stop", "take"):
            if manual:
                self._say(f"{heard}: you already have control")
                return set()
            self._say(f"{heard}: stopped, you have control (say hand back to let it drive)")
            return {O}
        if action == "hand_back":
            if not manual:
                self._say(f"{heard}: the assistant is already driving")
                return set()
            self._say(f"{heard}: the assistant drives")
            return {O}
        if action == "go":
            self._say(f"{heard}: going")
            return {G}
        if action == "page":
            if world_open is None:
                self._say(f"{heard}: there is no list to page through")
                return set()
            self._say(f"{heard}: next page")
            return {P}
        markers, numbered_markers = markers_up(marks), numbers_are_markers(manual, marks, world_open)
        if action == "number":          # what 1 to 9 picks now: a marker while the rider drives, else the list
            n = int(command.target)
            listed = markers if numbered_markers else world_open.names() if world_open is not None else []
            if n > len(listed):
                self._say(f"{heard}: there is no number {n} on the screen")
                return set()
            action, command = ("marker" if numbered_markers else "place"), Command("", listed[n - 1], command.heard)
        if action == "marker":
            if command.target not in markers:
                self._say(f"{heard}: that marker is not up any more")
                return set()
            marks.selector.select(markers.index(command.target))
            self.marker_go = not numbered_markers     # the open world, chair driving: G would be the list's
            self._say(f"{heard}: going to the {command.target}")
            return {G}
        if action == "place":
            if world_open is None:
                self._say(f"{heard}: places are only known in the open world (--open-world)")
                return set()
            if manual and marks is not None:    # G is the markers' while the rider drives: hand back first
                self.pending.append(Command("place", command.target, command.heard))
                self._say(f"{heard}: the assistant drives")
                return {O}
            if not world_open.pick(command.target):
                self._say(f"{heard}: {command.target} is not on the list any more")
                return set()
            self._say(f"{heard}: going to {command.target}")
            return {G}
        return set()

    def take_marker_go(self) -> bool:
        """Whether this step's G is for the marker picked by voice (run.py asks once a step)."""
        go, self.marker_go = self.marker_go, False
        return go

    def close(self) -> None:
        self.mic.close()


def start(model_dir: Path = MODEL_DIR, model: str | None = None) -> VoiceControl | None:
    """Voice commands with the laptop's microphone, or None (saying why) if they cannot start. The
    rider's own words are worked out by the model when the gateway's key and a model name are set."""
    try:
        mic = Microphone()
        control = VoiceControl(Recogniser(model_dir), mic)
    except VoiceUnavailable as exc:
        print(f"Voice commands are off: {exc}", flush=True)
        return None
    sys.path.insert(0, str(ROOT))
    from webots.interpret import Interpreter
    interpreter = Interpreter(model)
    if interpreter.missing():
        print(f"Voice: exact commands only, because there is {interpreter.missing()}", flush=True)
    else:
        control.interpreter = interpreter
        print(f"Voice: own words are worked out by {interpreter.model}", flush=True)
    print(f"Voice commands on ({mic.name}, {mic.rate} Hz): hold K (click the 3D view first), say what to do, "
          "let go. Exact commands: stop, take control, hand back, go ahead, go to <place>, go to the <colour> <shape>.",
          flush=True)
    return control


# --- checking the recogniser without the simulator --------------------------------------------------

def tts_engine() -> str | None:
    """A text-to-speech program on this computer: macOS say, or espeak-ng / espeak on Linux."""
    for name in ("say", "espeak-ng", "espeak"):
        if shutil.which(name):
            return name
    return None


def speak(text: str, engine: str, voice: str | None = None) -> bytes:
    """`text` spoken by the computer, as 16 kHz mono 16-bit sound, with a quarter of a second of silence
    before and after (as the microphone keeps the moment before K): without it the first word is lost."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "said.wav"
        if engine == "say":
            cmd = ["say", "-o", str(path), "--data-format=LEI16@16000"] + (["-v", voice] if voice else []) + [text]
        else:
            cmd = [engine, "-w", str(path)] + (["-v", voice] if voice else []) + [text]
        subprocess.run(cmd, check=True, capture_output=True)
        silence = bytes(2 * RATE // 4)
        return silence + read_wav(path) + silence


def read_wav(path) -> bytes:
    """A WAV file as 16 kHz mono 16-bit sound (resampled and mixed down if it is not)."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width, data = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    if (rate, channels, width) == (RATE, 1, 2):
        return data
    import numpy as np
    x = np.frombuffer(data, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[width]).astype(np.float64)
    if width == 1:
        x = (x - 128) * 256
    elif width == 4:
        x = x / 65536
    x = x.reshape(-1, channels).mean(axis=1)
    return resample(np.clip(x, -32768, 32767).astype(np.int16).tobytes(), rate)


def write_wav(path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def hospital_phrases() -> Phrases:
    """Every phrase a hospital trip could offer: all its places, every marker and the numbers."""
    from webots import landmarks, signs
    hospital = signs._hospital()
    places = sorted({p.name for p in hospital} | {p.group for p in hospital if p.group})
    return Phrases(places, [f"{c} {s}" for c in landmarks.PALETTE for s in landmarks.SHAPES], numbered=True)


def mic_check(seconds: float = 4.0, model_dir: Path = MODEL_DIR) -> int:
    """Records the microphone for a few seconds while the rider says a command, then says how loud it
    was and what was heard (every hospital phrase listened for). The sound is saved to
    data/voice/mic_check.wav to listen to (afplay on macOS)."""
    try:
        recogniser, mic = Recogniser(model_dir), Microphone()
    except VoiceUnavailable as exc:
        print(f"Voice commands cannot start: {exc}")
        return 1
    print(f"Microphone: {mic.name}, recording at {mic.rate} Hz")
    time.sleep(1.0)                          # the microphone settles
    print(f'Say a command now, such as "go to pharmacy" ({seconds:.0f} s)...', flush=True)
    mic.begin()
    time.sleep(seconds)
    pcm = mic.end()
    mic.close()
    path = ROOT / "data" / "voice" / "mic_check.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, pcm)
    print(f"Recorded {len(pcm) / 2 / RATE:.1f} s, loudest {loudest_db(pcm):.0f} dB "
          f"(speech near a laptop is about -30 to -5 dB, a quiet room about -60). Saved to {path}")
    if not pcm.strip(b"\0"):
        print("Only silence: on macOS allow Terminal in System Settings > Privacy & Security > Microphone")
        return 1
    phrases = hospital_phrases()
    heard, confidence = recogniser.recognise(pcm, phrases.phrases())
    command = phrases.match(heard, confidence)
    done = f"{command.action} {command.target or ''}".strip() if sure(command) else \
        "not sure enough: it would ask to say it again" if command else "not a command: nothing"
    print(f"Heard '{heard}' ({confidence:.0%}) -> {done}")
    return 0 if sure(command) else 1


def self_test(model_dir: Path = MODEL_DIR) -> int:
    """Each command and some hospital places, read out by the computer as a person would read the sign,
    and recognised with every hospital place and marker listened for at once (more phrases than a real
    trip ever has). Prints what was heard and what would be done."""
    engine = tts_engine()
    if engine is None:
        print("No text-to-speech program here (macOS has say). "
              "Record yourself instead: python webots/voice_eval.py record <name>")
        return 1
    phrases = hospital_phrases()
    recogniser = Recogniser(model_dir)
    tests = list(COMMANDS) + [f"go to {p}" for p in ("Emergency", "CT Scan", "Pharmacy", "Main Reception",
                                                     "4B Room 3", "X-Ray")] + \
        ["go to the yellow sphere", "go to the blue cube", "go to number three"]
    right = 0
    for text in tests:
        meant = phrases.match(normal(text)) or phrases.match(spoken(text))
        started = time.monotonic()
        heard, confidence = recogniser.recognise(speak(text, engine), phrases.phrases())
        took = time.monotonic() - started
        command = phrases.match(heard, confidence)
        ok = sure(command) and (command.action, command.target) == (meant.action, meant.target)
        right += ok
        done = f"{command.action} {command.target or ''}".strip() if sure(command) else \
            "not sure enough: nothing" if command else "not a command: nothing"
        print(f"{'ok  ' if ok else 'MISS'} said '{text}'  heard '{heard}'  {confidence:.2f}  "
              f"{took * 1000:.0f} ms  -> {done}")
    print(f"{right}/{len(tests)} right, {len(phrases.phrases())} phrases listened for")
    return 0 if right == len(tests) else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    if "--test" in sys.argv:
        sys.exit(self_test())
    if "--mic" in sys.argv:
        sys.exit(mic_check())
    print(__doc__)
