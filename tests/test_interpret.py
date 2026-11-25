"""Checks for the rider's own words (webots/interpret.py and how voice.py uses it): reading .env, the
request sent, reading and checking the model's answer, falling back when the gateway fails, and the
chair asking before it moves. A small web server on this computer stands in for the gateway, so no
key, network or simulator is needed.

    python tests/test_interpret.py          (or: python -m pytest tests, if pytest is installed)
"""

import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from webots import interpret, voice  # noqa: E402
from webots.interpret import Meaning, Screen  # noqa: E402

K, O, G = voice.PUSH_TO_TALK, ord("O"), ord("G")
HOSPITAL = ["Emergency", "Cafe", "Medical Imaging", "Pharmacy", "Ward 4A", "Ward 4B"]


# ------------------------------------------------------------------ a stand-in gateway
class Gateway(BaseHTTPRequestHandler):
    """Answers /chat/completions with whatever the test put in `replies`, and remembers the requests."""
    replies: list = []
    requests: list = []
    refuse_json_mode = False
    delay_s = 0.0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Gateway.requests.append((self.headers.get("Authorization"), body))
        time.sleep(Gateway.delay_s)
        if Gateway.refuse_json_mode and "response_format" in body:
            self.send_response(400)
            self.end_headers()
            return
        content = Gateway.replies.pop(0) if Gateway.replies else "{}"
        data = json.dumps({"choices": [{"message": {"content": content}}],
                           "usage": {"prompt_tokens": 420, "completion_tokens": 25}}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):   # the test stopped waiting (a time-out check)
            pass

    def do_GET(self):
        data = json.dumps({"data": [{"id": "model-b"}, {"id": "model-a"}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def gateway():
    server = HTTPServer(("127.0.0.1", 0), Gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    Gateway.replies, Gateway.requests, Gateway.refuse_json_mode, Gateway.delay_s = [], [], False, 0.0
    return server


def stop(server):
    """The stand-in gateway gone: no more answers, and its port closed."""
    server.shutdown()
    server.server_close()


def interpreter(server, timeout=2.0):
    i = interpret.Interpreter.__new__(interpret.Interpreter)
    i.base, i.key, i.model = f"http://127.0.0.1:{server.server_port}/v1", "test-key", "model-a"
    i.timeout, i.json_mode = timeout, True
    return i


# ------------------------------------------------------------------ interpret.py
def test_env_file():
    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / ".env"
        env.write_text("# the gateway\nexport G19_TEST_A='one'\nG19_TEST_B=\"two\"\nG19_TEST_C=three\n\n")
        os.environ["G19_TEST_C"] = "already set"
        interpret.load_env(env)
        assert (os.environ["G19_TEST_A"], os.environ["G19_TEST_B"], os.environ["G19_TEST_C"]) == \
            ("one", "two", "already set")
        for k in ("G19_TEST_A", "G19_TEST_B", "G19_TEST_C"):
            del os.environ[k]


def test_reading_answers():
    assert interpret.first_json('{"action": "stop"}') == {"action": "stop"}
    assert interpret.first_json('Sure! ```json\n{"action": "place", "target": "Cafe"}\n```') == \
        {"action": "place", "target": "Cafe"}
    assert interpret.first_json('thinking {not json} then {"action": "go"}') == {"action": "go"}
    assert interpret.first_json("no json here") is None


def test_answers_are_checked_against_the_screen():
    screen = Screen(places=HOSPITAL, listed=HOSPITAL[:4], markers=["yellow sphere"])
    assert interpret.check({"action": "place", "target": "pharmacy (Central Wing)"}, screen).target == "Pharmacy"
    made_up = interpret.check({"action": "place", "target": "Chapel"}, screen)
    assert made_up.action == "none" and made_up.invented
    assert interpret.check({"action": "marker", "target": "Yellow Sphere"}, screen).target == "yellow sphere"
    assert interpret.check({"action": "marker", "target": "red cube"}, screen).invented
    assert interpret.check({"action": "number", "target": "4"}, screen).target == "4"
    assert interpret.check({"action": "number", "target": 5}, screen).invented
    assert interpret.check({"action": "Hand Back", "target": None}, screen).action == "hand_back"
    assert not interpret.check({"action": "fly"}, screen).valid
    assert not interpret.check(None, screen).valid


def test_the_request_shows_the_screen():
    text = interpret.screen_text(Screen(places=["Pharmacy"], wings={"Pharmacy": "Central Wing"},
                                        listed=["Pharmacy"], markers=["red cube"], driving=True), "my pills")
    assert "Pharmacy (Central Wing)" in text and "1. Pharmacy" in text and "red cube" in text
    assert "is driving" in text and '"my pills"' in text


def test_hospital_directory():
    names = [n for n, _ in interpret.hospital_directory()]
    assert names[0] == "Emergency" and "Pharmacy" in names and "Ward 4A" in names and len(names) == 13
    assert dict(interpret.hospital_directory())["Ward 4A"] == "West Wing"


def test_a_request_through_the_gateway():
    server = gateway()
    Gateway.replies = ['{"action": "place", "target": "Pharmacy", "reply": "to the pharmacy"}']
    meaning = interpreter(server).interpret("I need my prescription", Screen(places=HOSPITAL))
    assert (meaning.action, meaning.target, meaning.tokens) == ("place", "Pharmacy", (420, 25))
    auth, body = Gateway.requests[0]
    assert auth == "Bearer test-key" and body["model"] == "model-a" and body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    assert "I need my prescription" in body["messages"][1]["content"]
    assert interpreter(server).models() == ["model-a", "model-b"]
    stop(server)


def test_a_model_without_json_mode_is_asked_again_without_it():
    server = gateway()
    Gateway.refuse_json_mode = True
    Gateway.replies = ['{"action": "stop", "target": null, "reply": "stopping"}']
    i = interpreter(server)
    assert i.interpret("hang on", Screen()).action == "stop"
    assert not i.json_mode and "response_format" not in Gateway.requests[-1][1]
    stop(server)


def test_failures_fall_back():
    server = gateway()
    Gateway.delay_s = 0.5
    slow = interpreter(server, timeout=0.2).interpret("cafe please", Screen(places=HOSPITAL))
    assert slow.action == "none" and "no answer" in slow.error
    stop(server)
    gone = interpreter(server, timeout=0.5).interpret("cafe please", Screen(places=HOSPITAL))
    assert gone.action == "none" and gone.error
    nothing = interpret.Interpreter.__new__(interpret.Interpreter)
    nothing.base, nothing.key, nothing.model = "", "", ""
    assert "no gateway" in nothing.missing()
    assert nothing.interpret("cafe", Screen()).error


# ------------------------------------------------------------------ voice.py with a model
class TwoPassRecogniser:
    """The exact pass hears `exact`, the free pass writes down `written`."""

    def __init__(self):
        self.exact, self.written = ("", 0.0), ""

    def recognise(self, pcm, phrases):
        return self.exact

    def transcribe(self, pcm):
        return self.written


class FakeInterpreter:
    def __init__(self):
        self.reply, self.asked = Meaning(), []

    def interpret(self, said, screen):
        self.asked.append((said, screen))
        return self.reply


class FakeMicrophone:
    def begin(self):
        pass

    def end(self):
        return (8000).to_bytes(2, "little", signed=True) * 1600

    def close(self):
        pass


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeOpenWorld:
    def __init__(self, places):
        self._places, self.picked = list(places), []
        self.session = SimpleNamespace(directory=lambda: {p: "Central Wing" for p in places})

    def places(self):
        return self._places

    def names(self):
        return self._places[:9]

    def pick(self, name):
        self.picked.append(name)
        return name in self._places


def control(model=True):
    return voice.VoiceControl(TwoPassRecogniser(), FakeMicrophone(), clock=Clock(),
                              interpreter=FakeInterpreter() if model else None)


def say(c, exact="", written="", confidence=1.0, meaning=None, **state):
    state = {"manual": False, "marks": None, "world_open": None, **state}
    c.recogniser.exact, c.recogniser.written = (exact, confidence), written
    if meaning is not None:
        c.interpreter.reply = meaning
    before = len(c.log)
    out = c.keys({K}, **state)
    c.clock.t += voice.RELEASE_S
    out |= c.keys(set(), **state)
    deadline = time.monotonic() + 3
    while len(c.log) == before and time.monotonic() < deadline:
        time.sleep(0.005)
        out |= c.keys(set(), **state)
    assert len(c.log) == before + 1, "nothing came back"
    return out


def test_exact_commands_skip_the_model():
    c, world = control(), FakeOpenWorld(HOSPITAL)
    assert say(c, "take me to pharmacy", "can you take me to the pharmacy please", world_open=world) == {G}
    assert world.picked == ["Pharmacy"] and c.interpreter.asked == []
    assert say(c, "go to c t scan", "go to see t scan") == set()     # CT Scan is not read here: a model question
    assert say(c, "stop", "stop") == {O}


def test_saying_no_goes_to_the_model():
    c, world = control(), FakeOpenWorld(HOSPITAL)
    meaning = Meaning("none", None, "staying put")
    assert say(c, "go to pharmacy", "i don't want to go to pharmacy", meaning=meaning, world_open=world) == set()
    assert world.picked == [] and c.interpreter.asked[0][0] == "i don't want to go to pharmacy"
    assert "staying put" in c.line()


def test_stop_words_never_wait_for_the_model():
    c = control()
    assert say(c, "go ahead", "hang on a second") == {O}
    assert say(c, "", "wait for me", manual=False) == {O}
    assert c.interpreter.asked == []
    assert not voice.says_stop("go to the waiting area")


def test_the_chair_asks_before_a_model_move():
    c, world = control(), FakeOpenWorld(HOSPITAL)
    meaning = Meaning("place", "Pharmacy", "your prescription")
    assert say(c, "", "i need to pick up my prescription", meaning=meaning, world_open=world) == set()
    assert world.picked == ["Pharmacy"] and c.asking is not None
    assert "Pharmacy? Say yes or press G" in c.line()
    screen = c.interpreter.asked[0][1]
    assert screen.places == HOSPITAL and screen.wings["Pharmacy"] == "Central Wing" and not screen.driving
    assert say(c, "yes", "yes", world_open=world) == {G}            # yes: it goes
    assert c.asking is None


def test_no_and_g_and_waiting_too_long():
    c, world = control(), FakeOpenWorld(HOSPITAL)
    meaning = Meaning("place", "Cafe", "coffee")
    say(c, "", "i could do with a coffee", meaning=meaning, world_open=world)
    assert say(c, "no", "no thanks", world_open=world) == set() and c.asking is None
    say(c, "", "i could do with a coffee", meaning=meaning, world_open=world)
    c.keys({G}, manual=False, world_open=world)                      # the rider pressed G: run.py goes
    assert c.asking is None
    say(c, "", "i could do with a coffee", meaning=meaning, world_open=world)
    c.clock.t += voice.CONFIRM_S + 1
    c.keys(set(), manual=False, world_open=world)
    assert c.asking is None
    asked = len(c.interpreter.asked)                                 # nothing asked now: yes is no command,
    assert say(c, "yes", "yes", meaning=Meaning("none", None, "nothing to say yes to"), world_open=world) == set()
    assert len(c.interpreter.asked) == asked + 1 and "nothing to say yes to" in c.line()   # so the model hears it


def test_yes_is_only_listened_for_while_asking():
    assert "yes" not in voice.Phrases(["Cafe"]).table
    assert voice.Phrases(["Cafe"], asking=True).match("yes please").action == "yes"


def test_model_stop_and_page_are_done_at_once():
    c, world = control(), FakeOpenWorld(HOSPITAL)
    assert say(c, "", "i'll take it from here", meaning=Meaning("stop"), world_open=world) == {O}
    assert say(c, "", "show me more", meaning=Meaning("page"), world_open=world) == {ord("P")}


def test_model_markers_and_numbers():
    c = control()
    marks = SimpleNamespace(items=[SimpleNamespace(name="yellow sphere"), SimpleNamespace(name="red cube")],
                            selector=SimpleNamespace(index=None))
    marks.selector.select = lambda i: setattr(marks.selector, "index", i)
    assert say(c, "", "the red box", meaning=Meaning("marker", "red cube"), manual=True, marks=marks) == set()
    assert marks.selector.index == 1 and "red cube? Say yes or press G" in c.line()
    assert say(c, "yes", "yes", manual=True, marks=marks) == {G}
    world = FakeOpenWorld(HOSPITAL)
    assert say(c, "", "the third one", meaning=Meaning("number", "3"), world_open=world) == set()
    assert world.picked == ["Medical Imaging"]
    assert say(c, "", "number nine", meaning=Meaning("number", "9"), world_open=world) == set()
    assert "no number 9" in c.line()


def test_model_markers_while_the_chair_drives_in_the_open_world():
    c, world = control(), FakeOpenWorld(HOSPITAL)
    marks = SimpleNamespace(items=[SimpleNamespace(name="yellow sphere"), SimpleNamespace(name="red cube")],
                            selector=SimpleNamespace(index=None, selected=None))
    marks.selector.select = lambda i: setattr(marks.selector, "index", i)
    assert say(c, "", "that red box over there", meaning=Meaning("marker", "red cube"), marks=marks,
               world_open=world) == set()
    assert marks.selector.index == 1 and c.line().endswith("red cube? Say yes")   # G would be the list's
    assert "red cube" in c.interpreter.asked[-1][1].markers
    assert say(c, "yes", "yes", marks=marks, world_open=world) == {G} and c.take_marker_go()


def test_the_model_failing_leaves_exact_commands():
    c = control()
    assert say(c, "", "take me somewhere nice", meaning=Meaning(error="no answer from the model (timeout)")) == set()
    assert "Exact commands still work" in c.line()
    offline = control(model=False)
    assert say(offline, "go ahead", "go ahead") == {G}
    assert say(offline, "go to pharmacy", "i don't want to go to pharmacy",
               world_open=FakeOpenWorld(HOSPITAL)) == set()
    assert "not sure it was" in offline.line()


def test_agreeing():
    assert voice.agrees("take me to pharmacy", "can you take me to the pharmacy please")
    assert voice.agrees("go to c t scan", "go to see t scan")
    assert voice.agrees("hand back", "hand back")
    assert not voice.agrees("go to pharmacy", "i don't want to go to pharmacy")
    assert not voice.agrees("go to cafe", "what time does the cafe close tonight")


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
