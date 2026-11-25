"""Checks for the voice commands (webots/voice.py): how names on signs are said, which phrases mean
which command, and the keys each command comes in as. A stand-in recogniser and microphone are used,
so no microphone, speech model or simulator is needed. The last check speaks a few commands with the
computer's own voice (macOS say) and recognises them for real. It is skipped without the speech model
(bash webots/get_voice_model.sh) or a text-to-speech program.

    python tests/test_voice.py          (or: python -m pytest tests, if pytest is installed)
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webots import voice  # noqa: E402
from webots.open_world import OpenWorld  # noqa: E402

K, O, G, P = voice.PUSH_TO_TALK, ord("O"), ord("G"), ord("P")


class FakeRecogniser:
    """Hears whatever the test says it heard, and keeps the phrases it was asked to listen for."""

    def __init__(self):
        self.reply, self.phrases, self.fail = ("", 0.0), [], False

    def recognise(self, pcm, phrases):
        self.phrases = list(phrases)
        if self.fail:
            raise RuntimeError("broken")
        return self.reply


class FakeMicrophone:
    def __init__(self):
        self.begun = self.ended = 0
        self.closed = False

    def begin(self):
        self.begun += 1

    def end(self):
        self.ended += 1
        return self.sound

    sound = (8000).to_bytes(2, "little", signed=True) * 1600   # someone speaking: -12 dB

    def close(self):
        self.closed = True


class FakeMarks:
    """landmarks.Landmarks as voice.py sees it: the markers up and the selector."""

    def __init__(self, names):
        self.items = [SimpleNamespace(name=n) for n in names]
        self.selector = SimpleNamespace(index=None)
        self.selector.select = lambda i: setattr(self.selector, "index", i)


class FakeOpenWorld:
    def __init__(self, places, page=None):
        self._places, self._page, self.picked = list(places), list(page or places[:9]), []

    def places(self):
        return self._places

    def names(self):
        return self._page

    def pick(self, name):
        if name not in self._places:
            return False
        self.picked.append(name)
        return True


class Clock:
    """Time that only moves when the test says so."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def control():
    return voice.VoiceControl(FakeRecogniser(), FakeMicrophone(), clock=Clock())


def let_go(c, **state):
    """K comes up and stays up long enough to count as let go."""
    c.clock.t += voice.RELEASE_S
    return c.keys(set(), **state)


def say(c, heard, confidence=1.0, **state):
    """Hold K for a step, let go, and give back the keys the command came in as (the recogniser works
    in the background, so this waits for it)."""
    state = {"manual": False, "marks": None, "world_open": None, **state}
    c.recogniser.reply = (heard, confidence)
    before = len(c.log)
    out = c.keys({K}, **state) | let_go(c, **state)
    deadline = time.monotonic() + 3
    while len(c.log) == before and time.monotonic() < deadline:
        time.sleep(0.005)
        out |= c.keys(set(), **state)
    assert len(c.log) == before + 1, "the recogniser never answered"
    return out


def test_spoken_names():
    assert voice.spoken("CT Scan") == "c t scan"
    assert voice.spoken("MRI") == "m r i"
    assert voice.spoken("4B Room 3") == "four b room three"
    assert voice.spoken("A&E") == "a and e"
    assert voice.spoken("X-Ray") == "x ray"
    assert voice.spoken("Café") == "cafe"
    assert voice.spoken("Nurses' Station 4A") == "nurses station four a"
    assert voice.spoken("Meeting Room 21") == "meeting room twenty one"
    assert voice.spoken("Pharmacy") == "pharmacy"


def test_number_words():
    assert [voice.number_words(n) for n in (0, 7, 13, 20, 45, 99)] == \
        ["zero", "seven", "thirteen", "twenty", "forty five", "ninety nine"]
    assert voice.number_words(101) == "one zero one"


def test_phrases():
    ph = voice.Phrases(["CT Scan", "Ward 4A"], ["yellow sphere"], numbered=True)
    assert ph.match("stop").action == "stop"
    assert ph.match("Go  Ahead").action == "go"
    assert ph.match("you drive").action == "hand_back"
    assert ph.match("take control").action == "take"
    place = ph.match("go to c t scan")
    assert (place.action, place.target) == ("place", "CT Scan")
    assert ph.match("take me to ward four a").target == "Ward 4A"
    marker = ph.match("go to the yellow sphere")
    assert (marker.action, marker.target) == ("marker", "yellow sphere")
    assert ph.match("go to yellow sphere").target == "yellow sphere"
    number = ph.match("go to number three")
    assert (number.action, number.target) == ("number", "3")
    for nothing in ("", "[unk]", "go to", "go to ward four", "go to ward four a ray", "pharmacy", "hello"):
        assert ph.match(nothing) is None, nothing
    assert ph.match("stop the to go").action == "stop"      # any stop stops


def test_only_what_is_known_is_listened_for():
    assert "go to pharmacy" not in voice.Phrases(["CT Scan"]).table
    assert not any("number" in p for p in voice.Phrases().phrases())
    c = control()
    world = FakeOpenWorld(["Emergency", "CT Scan"])
    say(c, "stop", world_open=world)
    assert "go to c t scan" in c.recogniser.phrases and "go to number two" in c.recogniser.phrases
    assert not any("sphere" in p for p in c.recogniser.phrases)     # no markers while the assistant drives
    say(c, "stop", manual=True, marks=FakeMarks(["red cube"]), world_open=world)
    assert "go to the red cube" in c.recogniser.phrases


def test_stop_and_take_control():
    c = control()
    assert say(c, "stop") == {O}
    assert "you have control" in c.line()
    assert say(c, "stop", manual=True) == set()
    assert say(c, "take control") == {O}
    assert say(c, "stop", confidence=0.1) == {O}             # stopping is never held back


def test_hand_back_and_go():
    c = control()
    assert say(c, "hand back", manual=True) == {O}
    assert say(c, "hand back") == set()
    assert say(c, "go ahead") == {G}
    assert say(c, "next page") == set()                      # no list without the open world
    assert say(c, "next page", world_open=FakeOpenWorld(["CT Scan"])) == {P}


def test_unsure_moves_are_not_made():
    c = control()
    assert say(c, "hand back", confidence=voice.MIN_CONFIDENCE - 0.01, manual=True) == set()
    assert "say it again" in c.line().lower()
    assert say(c, "hand back", confidence=voice.MIN_CONFIDENCE, manual=True) == {O}


def test_not_a_command():
    c = control()
    assert say(c, "") == set()
    assert "didn't catch" in c.line()
    assert say(c, "go to ward four a ray") == set()
    assert "not a command" in c.line()
    c.recogniser.fail = True
    assert say(c, "stop") == set()                           # a broken recogniser never stops the trip


def test_go_to_a_place():
    c, world = control(), FakeOpenWorld(["Emergency", "CT Scan", "Pharmacy"])
    assert say(c, "go to pharmacy", world_open=world) == {G}
    assert world.picked == ["Pharmacy"]
    assert "going to Pharmacy" in c.line()


def test_go_to_a_place_while_driving_with_markers():
    """While the rider drives with markers up, G is the markers': the chair is handed back first."""
    c, world, marks = control(), FakeOpenWorld(["CT Scan"]), FakeMarks(["red cube"])
    assert say(c, "go to c t scan", manual=True, marks=marks, world_open=world) == {O}
    assert world.picked == []
    assert c.keys(set(), manual=False, marks=marks, world_open=world) == {G}
    assert world.picked == ["CT Scan"]


def test_paused_hand_back_is_dropped():
    c, world, marks = control(), FakeOpenWorld(["CT Scan"]), FakeMarks(["red cube"])
    assert say(c, "go to c t scan", manual=True, marks=marks, world_open=world) == {O}
    assert c.keys(set(), manual=True, marks=marks, world_open=world) == set()   # O did nothing: paused
    assert c.keys(set(), manual=True, marks=marks, world_open=world) == set() and not c.pending
    assert world.picked == []


def test_go_to_a_marker():
    c, marks = control(), FakeMarks(["yellow sphere", "red cube"])
    assert say(c, "go to the red cube", manual=True, marks=marks) == {G}
    assert marks.selector.index == 1
    marks.items = marks.items[:1]                            # new markers went up meanwhile
    assert c._act(voice.Command("marker", "red cube"), manual=True, marks=marks, world_open=None) == set()


def test_go_to_a_number():
    c, world = control(), FakeOpenWorld(["Emergency", "CT Scan", "Pharmacy"])
    assert say(c, "go to number two", world_open=world) == {G}
    assert world.picked == ["CT Scan"]
    assert say(c, "go to number nine", world_open=world) == set()
    assert "no number 9" in c.line()
    marks = FakeMarks(["yellow sphere", "red cube"])         # while the rider drives: the markers' numbers
    assert say(c, "go to number one", manual=True, marks=marks, world_open=world) == {G}
    assert marks.selector.index == 0 and world.picked == ["CT Scan"]


def test_a_silent_microphone_is_explained():
    c = control()
    c.mic.sound = b"\0\0" * 1600                           # what macOS gives without permission
    assert say(c, "stop") == set()
    assert "Privacy & Security > Microphone" in c.line()
    c.mic.sound = b"\x05\0" * 1600                         # a whisper far away: -76 dB
    assert say(c, "") == set()
    assert "hardly heard you" in c.line()
    assert say(c, "stop") == {O}                             # quiet but understood: acted on


def test_loudness_and_resampling():
    assert voice.loudest_db(b"") == -120 and voice.loudest_db(b"\0\0" * 10) == -120
    assert round(voice.loudest_db(b"\x00\x80")) == 0       # -32768, the loudest possible
    assert round(voice.loudest_db((3277).to_bytes(2, "little", signed=True))) == -20
    one_second_48k = b"\1\0" * 48000
    assert len(voice.resample(one_second_48k, 48000)) == 2 * voice.RATE
    assert voice.resample(b"\1\0" * 10, voice.RATE) == b"\1\0" * 10


def test_markers_by_colour_alone():
    ph = voice.Phrases([], ["yellow sphere", "red cube"])
    assert ph.match("go to the yellow one").target == "yellow sphere"
    assert ph.match("go to the red marker").target == "red cube"
    twice = voice.Phrases([], ["red sphere", "red cube"])          # two reds: the colour alone is not enough
    assert twice.match("go to the red one") is None and twice.match("go to the red cube").target == "red cube"


def test_open_world_markers_while_the_chair_drives():
    """The open world keeps markers up while the chair drives itself: one can be named any time.
    G then is the list's, so run.py is told the G is for the marker."""
    c, world, marks = control(), FakeOpenWorld(["CT Scan", "Pharmacy"]), FakeMarks(["yellow sphere", "red cube"])
    assert say(c, "go to the red one", manual=False, marks=marks, world_open=world) == {G}
    assert marks.selector.index == 1 and world.picked == []
    assert c.take_marker_go() and not c.take_marker_go()          # once
    assert "go to the yellow sphere" in c.recogniser.phrases
    assert say(c, "go to number two", manual=False, marks=marks, world_open=world) == {G}
    assert world.picked == ["Pharmacy"] and not c.take_marker_go()   # numbers are the list's here
    assert say(c, "go to the yellow sphere", manual=True, marks=marks, world_open=world) == {G}
    assert not c.take_marker_go()                                 # while the rider drives, G is the markers' anyway


def test_push_to_talk():
    c = control()
    assert c.keys({K}, manual=False) == set() and c.talking and c.mic.begun == 1
    assert "listening" in c.line()
    c.keys({K, O}, manual=False)                              # still held: nothing new
    assert c.mic.begun == 1 and c.mic.ended == 0
    c.clock.t += 0.1
    c.keys(set(), manual=False)                               # up for a moment only: a flicker, still talking
    assert c.talking and c.mic.ended == 0
    c.clock.t += 0.1
    c.keys({K}, manual=False)
    assert c.mic.begun == 1
    let_go(c, manual=False)
    assert c.mic.ended == 1 and not c.talking
    c.close()
    assert c.mic.closed


def test_microphone_keeps_the_moment_before_k():
    mic = voice.Microphone.__new__(voice.Microphone)          # without opening a real microphone
    import collections
    import threading
    mic.lock, mic.recording, mic.chunks = threading.Lock(), False, []
    mic.before = collections.deque(maxlen=3)
    for i in range(5):
        mic._heard(bytes([i]) * 4, 2, None, None)
    mic.begin()
    mic._heard(b"\x09" * 4, 2, None, None)
    assert mic.end() == bytes([2]) * 4 + bytes([3]) * 4 + bytes([4]) * 4 + b"\x09" * 4
    assert mic.end() == b""


def test_open_world_pick_turns_the_list():
    world = OpenWorld.__new__(OpenWorld)                       # its list, without a simulator
    directory = {f"Place {i:02d}": "" for i in range(12)} | {"Emergency": ""}
    world.session = SimpleNamespace(directory=lambda: directory, nearby=lambda: ["Room 1", "Room 2"])
    world.showing, world.page, world.picked = "directory", 0, None
    assert world.places()[0] == "Emergency" and "Room 2" in world.places()
    assert world.pick("Place 10")
    assert (world.showing, world.page, world.picked) == ("directory", 1, "Place 10")
    assert world.pick("Room 2")
    assert (world.showing, world.page, world.picked) == ("nearby", 0, "Room 2")
    assert not world.pick("Pharmacy") and world.picked == "Room 2"


def test_real_recognition():
    """Commands spoken by the computer, recognised by the real model (skipped without them)."""
    engine = voice.tts_engine()
    if engine is None or not (voice.MODEL_DIR / "am").exists():
        print("     (skipped: needs the speech model and a text-to-speech program)")
        return
    rec = voice.Recogniser()
    ph = voice.Phrases(["CT Scan", "Pharmacy", "Emergency", "Main Reception"], ["yellow sphere", "blue cube"],
                       numbered=True)
    for text, action, target in [("stop", "stop", None), ("hand back", "hand_back", None),
                                 ("go to pharmacy", "place", "Pharmacy"),
                                 ("go to the blue cube", "marker", "blue cube")]:
        heard, confidence = rec.recognise(voice.speak(text, engine), ph.phrases())
        command = ph.match(heard, confidence)
        assert command is not None and (command.action, command.target) == (action, target), (text, heard)


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
