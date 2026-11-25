"""Which model should work out the rider's own words (interpret.py)? Every chat model on the course
gateway is tried on the same requests (webots/voice_requests.jsonl: 231 requests in riders' own
words, each with the screen it is said on and the answers that are right), against two that need no
gateway: the exact phrases alone (what the voice commands do offline) and a small classifier trained on
example sentences (naive Bayes, as in tech-5).

    python webots/interpret_eval.py models                  # the gateway's models and their prices
    python webots/interpret_eval.py screen                  # every chat model on 40 validation requests
    python webots/interpret_eval.py compare MODEL MODEL ... # these on all 117 validation requests, twice
    python webots/interpret_eval.py test MODEL              # the chosen one on the 114 test requests, once
    python webots/interpret_eval.py spoken MODEL            # the test requests read out (macOS say), end to end
    python webots/interpret_eval.py all                     # all of these in turn, the best 4 of the screen compared

The requests are split in half by kind (seed 0): validation to choose, test to report. A model is
chosen by a rule fixed before any were tried:
    1. wrong moves (the chair sent somewhere it was not asked) at most 1% of requests
    2. answers readable as a command at least 99%
    3. 90% of answers within 2.5 s
    4. then the most right answers, then the cheapest
An answer slower than interpret.TIMEOUT_S (3 s) counts as no answer, as in the chair. Every request's
result goes to data/voice/interpret_<stage>.csv, and everything printed to data/voice/interpret_report.txt.
The key and address come from G19/.env.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webots import interpret, voice  # noqa: E402
from webots.interpret import Meaning, Screen  # noqa: E402

REQUESTS = ROOT / "webots" / "voice_requests.jsonl"
OUT = ROOT / "data" / "voice"
MAX_WRONG, MIN_READABLE, MAX_P90_S = 0.01, 0.99, 2.5
NOT_CHAT = re.compile(r"embed|whisper|tts|dall|image|rerank|moderation|transcri|audio|speech|sora|vision-only", re.I)
MARKERS = ["yellow sphere", "red cone", "blue cube", "green cylinder", "orange sphere", "purple cube"]
IMAGING = ["Imaging Reception", "X-Ray", "CT Scan", "Ultrasound", "MRI", "Nuclear Medicine"]
WARD_4A = ["Nurses' Station 4A", "4A Room 1", "4A Room 2", "4A Room 3", "Treatment Room", "4A Room 4", "4A Room 5",
           "4A Room 6", "Clean Utility", "Dirty Utility", "Medication Room", "Staff Room"]


def screens() -> dict[str, Screen]:
    """The screens the requests are said on: the hospital in the open world at its usual moments."""
    directory = interpret.hospital_directory()
    names, wings = [n for n, _ in directory], dict(directory)
    return {
        "start": Screen(),                                                       # nothing read yet
        "directory": Screen(places=names, wings=wings, listed=names[:9]),         # the directory read
        "picked": Screen(places=names, wings=wings, listed=names[:9], picked="Pharmacy"),
        "imaging": Screen(places=names + IMAGING, wings=wings, listed=IMAGING, here="Medical Imaging"),
        "ward4a": Screen(places=names + WARD_4A, wings=wings, listed=WARD_4A[:9], here="Ward 4A"),
        "driving": Screen(places=names, wings=wings, listed=MARKERS, markers=MARKERS, driving=True),
        "ordinary": Screen(listed=MARKERS, markers=MARKERS),                     # run.py --landmarks, no open world
    }


def load(split: str | None = None, sample: int | None = None, seed: int = 0) -> list[dict]:
    """The requests (numbered in file order), all or one half: half of each kind is validation."""
    rows = [dict(r, id=i) for i, r in enumerate(json.loads(line) for line in REQUESTS.read_text().splitlines() if line)]
    if split:
        rng, keep = random.Random(seed), []
        by_kind = defaultdict(list)
        for r in rows:
            by_kind[r["kind"]].append(r)
        for kind in sorted(by_kind):
            group = by_kind[kind][:]
            rng.shuffle(group)
            half = (len(group) + 1) // 2
            keep += group[:half] if split == "validation" else group[half:]
        rows = sorted(keep, key=lambda r: r["id"])
    if sample:
        rng = random.Random(seed + 1)
        by_kind = defaultdict(list)
        for r in rows:
            by_kind[r["kind"]].append(r)
        picked = [rng.choice(group) for group in by_kind.values()]          # every kind at least once
        rest = [r for r in rows if r not in picked]
        picked += rng.sample(rest, max(0, min(sample - len(picked), len(rest))))
        rows = sorted(picked, key=lambda r: r["id"])
    return rows


def resolve(action: str, target, screen: Screen) -> str:
    """An answer as what the chair would do (voice.py): a number becomes the place or marker it stands
    for, go becomes what is picked (or the directory, before one is read), and handing back when the
    rider is not driving does nothing."""
    if action == "go":
        if screen.picked:
            return f"{'marker' if screen.picked in screen.markers else 'place'}:{screen.picked}"
        return "go" if not screen.places and not screen.markers else "none"
    if action == "hand_back" and not screen.driving:
        return "none"
    if action == "number" and target:
        n = int(target)
        if 1 <= n <= len(screen.listed):
            name = screen.listed[n - 1]
            return f"{'marker' if name in screen.markers else 'place'}:{name}"
        return "none"
    if action in ("place", "marker"):
        return f"{action}:{target}"
    return action


def outcome(got: str, want: list[str], meaning: Meaning) -> str:
    if meaning.error:
        return "no answer"
    if meaning.seconds > interpret.TIMEOUT_S:
        return "too slow"
    if not meaning.valid:
        return "unreadable"
    if got in want:
        return "right"
    if got.split(":")[0] in interpret.MOVES:
        return "wrong move"
    return "miss"                         # nothing done, or a stop: safe


# ------------------------------------------------------------------ the two that need no gateway
class Exact:
    """What the voice commands do without a model: the exact phrases, and a stop word anywhere."""
    name = "exact phrases (no model)"

    def interpret(self, said: str, screen: Screen) -> Meaning:
        phrases = voice.Phrases(screen.places, screen.markers, numbered=bool(screen.listed))
        command = phrases.match(voice.normal(said)) or phrases.match(voice.spoken(said))
        if voice.says_stop(said) or (command and command.action in ("stop", "take")):
            return Meaning("stop")
        if command is None:
            return Meaning()
        return Meaning(command.action, command.target)


TRAINING = {   # example sentences for the classifier: general wordings, none taken from the requests
    "stop": ["stop", "stop the chair", "please stop", "stop now", "hold on", "wait", "take control", "let me drive",
             "i will drive", "i want control", "stop moving", "halt", "stop please", "give me control"],
    "hand_back": ["hand back", "you drive", "you can drive", "drive yourself", "take over", "your turn to drive",
                  "drive on your own", "you steer", "back to automatic", "let it drive"],
    "go": ["go ahead", "carry on", "go", "go now", "start going", "move on", "keep going", "set off", "start"],
    "place": ["go to {p}", "take me to {p}", "i want to go to {p}", "{p} please", "can we go to {p}", "i need {p}",
              "the {p}", "head to {p}", "bring me to {p}", "i am going to {p}", "to {p}", "{p}"],
    "marker": ["go to the {m}", "the {m}", "head to the {m}", "drive to the {m}", "{m} please"],
    "number": ["number {n}", "go to number {n}", "pick {n}", "option {n}", "{n}"],
    "page": ["next page", "more", "show more", "next", "the next page", "more places"],
    "none": ["hello", "thanks", "nice day", "how are you", "what is that", "good", "yes", "no", "i see", "right",
             "interesting", "goodbye", "who is that", "what is happening"],
}
NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


class Trained:
    """A small trained classifier (multinomial naive Bayes on words and word pairs) for the action,
    with the place or marker found by spelling likeness to those on the screen."""
    name = "trained classifier (naive Bayes)"

    def __init__(self):
        rows = []
        places = [n for n, _ in interpret.hospital_directory()] + IMAGING + WARD_4A
        for action, templates in TRAINING.items():
            for t in templates:
                for p in (places if "{p}" in t else [None]):
                    for m in (MARKERS if "{m}" in t else [None]):
                        for n in (NUMBERS if "{n}" in t else [None]):
                            rows.append((t.format(p=voice.spoken(p or ""), m=m or "", n=n or ""), action))
        self.actions = sorted(TRAINING)
        counts = {a: Counter() for a in self.actions}
        docs = Counter()
        for text, action in rows:
            counts[action].update(self.features(text))
            docs[action] += 1
        self.vocab = set().union(*counts.values())
        self.prior = {a: math.log(docs[a] / len(rows)) for a in self.actions}
        self.like = {a: (counts[a], sum(counts[a].values())) for a in self.actions}

    @staticmethod
    def features(text: str) -> list[str]:
        words = voice.normal(re.sub(r"[^a-zA-Z0-9 ]", " ", text)).split()
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def interpret(self, said: str, screen: Screen) -> Meaning:
        feats = [f for f in self.features(said) if f in self.vocab]
        v = len(self.vocab)
        scores = {a: self.prior[a] + sum(math.log((c[f] + 1) / (total + v)) for f in feats)
                  for a, (c, total) in self.like.items()}
        action = max(scores, key=scores.get)
        if action == "place":
            name = self.closest(said, screen.places)
            return Meaning("place", name) if name else Meaning()
        if action == "marker":
            name = self.closest(said, screen.markers)
            return Meaning("marker", name) if name else Meaning()
        if action == "number":
            n = next((NUMBERS.index(w) + 1 for w in voice.normal(said).split() if w in NUMBERS), 0)
            return Meaning("number", str(n)) if 1 <= n <= len(screen.listed) else Meaning()
        return Meaning(action)

    @staticmethod
    def closest(said: str, options: list[str]) -> str | None:
        import difflib
        words = voice.normal(re.sub(r"[^a-zA-Z0-9 ]", " ", said)).split()
        best, best_score = None, 0.0
        for option in options:
            target = voice.spoken(option)
            k = len(target.split())
            for i in range(len(words)):
                for j in range(i + 1, min(len(words), i + k + 2) + 1):
                    score = difflib.SequenceMatcher(a=" ".join(words[i:j]), b=target).ratio()
                    if score > best_score:
                        best, best_score = option, score
        return best if best_score >= 0.75 else None


# ------------------------------------------------------------------ running and counting
def prices(gateway: interpret.Interpreter) -> dict[str, tuple[float, float]]:
    """US$ per token (in, out) for each model, if the gateway tells (LiteLLM's /model/info)."""
    found = {}
    for path in (gateway.base + "/model/info", re.sub(r"/v1$", "", gateway.base) + "/model/info"):
        try:
            request = urllib.request.Request(path, headers={"Authorization": f"Bearer {gateway.key}"})
            with urllib.request.urlopen(request, timeout=20) as response:
                for m in json.loads(response.read()).get("data", []):
                    info = m.get("model_info") or {}
                    if info.get("input_cost_per_token") is not None:
                        found[m.get("model_name")] = (float(info["input_cost_per_token"]),
                                                      float(info.get("output_cost_per_token") or 0))
            if found:
                return found
        except Exception:  # noqa: BLE001 (the gateway may not offer it)
            continue
    return found


def chat_models(gateway: interpret.Interpreter) -> list[str]:
    return [m for m in gateway.models() if not NOT_CHAT.search(m)]


def run(model, requests: list[dict], repeats: int = 1, workers: int = 4) -> list[dict]:
    """Every request to one model (an interpret.Interpreter, or Exact or Trained), `repeats` times."""
    shown = screens()
    name = getattr(model, "model", None) or model.name

    def one(job):
        repeat, r = job
        screen = shown[r["screen"]]
        if voice.says_stop(r["said"]):            # run.py stops on these itself, without asking a model
            meaning = Meaning("stop")
        else:
            meaning = model.interpret(r["said"], screen)
            for wait in (2, 5, 10):               # the gateway is busy (429): the chair would fall back,
                if "429" not in meaning.error:    # but the comparison waits and asks again
                    break
                time.sleep(wait)
                meaning = model.interpret(r["said"], screen)
        got = resolve(meaning.action, meaning.target, screen)
        want = [resolve(*(w.split(":", 1) + [None])[:2], screen) for w in r["want"]]
        return {"model": name, "where": for_rows, "repeat": repeat, "id": r["id"], "kind": r["kind"], "screen": r["screen"],
                "said": r["said"], "want": " | ".join(want), "got": got,
                "outcome": outcome(got, want, meaning), "seconds": round(meaning.seconds, 3),
                "tokens_in": meaning.tokens[0], "tokens_out": meaning.tokens[1],
                "invented": meaning.invented, "error": meaning.error, "reply": meaning.reply,
                "raw": meaning.raw.replace("\n", " ")[:300]}

    jobs = [(k, r) for k in range(1, repeats + 1) for r in requests]
    remote = isinstance(model, interpret.Interpreter)
    for_rows = "gateway" if remote else "offline"
    with ThreadPoolExecutor(max_workers=workers if remote else 1) as pool:
        return list(pool.map(one, jobs))


def summary(rows: list[dict], price: dict) -> dict:
    n = len(rows)
    count = Counter(r["outcome"] for r in rows)
    answered = [r["seconds"] for r in rows if r["outcome"] not in ("no answer",)]
    model = rows[0]["model"]
    p_in, p_out = price.get(model, (None, None))
    tin = np.mean([r["tokens_in"] for r in rows]) if rows else 0
    tout = np.mean([r["tokens_out"] for r in rows]) if rows else 0
    out = {"model": model, "requests": n, "right": count["right"] / n, "wrong_moves": count["wrong move"],
           "wrong_share": count["wrong move"] / n, "readable": 1 - count["unreadable"] / n,
           "no_answer": (count["no answer"] + count["too slow"]) / n,
           "median_s": float(np.median(answered)) if answered else math.nan,
           "p90_s": float(np.percentile(answered, 90)) if answered else math.nan,
           "tokens_in": tin, "tokens_out": tout,
           "usd_per_1000": (1000 * (tin * p_in + tout * p_out)) if p_in is not None else math.nan,
           "invented": sum(1 for r in rows if r["invented"] in (True, "True"))}
    repeats = defaultdict(dict)
    for r in rows:
        repeats[r["id"]][r["repeat"]] = r["got"]
    pairs = [v for v in repeats.values() if len(v) > 1]
    out["same_twice"] = (sum(1 for v in pairs if len(set(v.values())) == 1) / len(pairs)) if pairs else math.nan
    remote = rows[0].get("where") == "gateway"
    out["where"] = rows[0].get("where")
    out["passes"] = (out["wrong_share"] <= MAX_WRONG and out["readable"] >= MIN_READABLE
                     and (not remote or out["p90_s"] <= MAX_P90_S))
    return out


def table(summaries: list[dict]) -> None:
    print(f"\n{'model':44s} {'right':>6s} {'wrong':>6s} {'readable':>8s} {'no ans':>6s} {'median':>7s} "
          f"{'p90':>6s} {'same 2x':>7s} {'$/1000':>7s} {'invented':>8s}  rule")
    for s in sorted(summaries, key=lambda s: (-s["passes"], -s["right"], s["usd_per_1000"] if s["usd_per_1000"] ==
                                              s["usd_per_1000"] else 1e9, s["median_s"])):
        print(f"{s['model'][:44]:44s} {s['right']:6.0%} {s['wrong_moves']:6d} {s['readable']:8.0%} {s['no_answer']:6.0%} "
              f"{s['median_s']:6.2f}s {s['p90_s']:5.2f}s {s['same_twice']:7.0%} {s['usd_per_1000']:7.3f} "
              f"{s['invented']:8d}  {'passes' if s['passes'] else 'fails'}")


def choose(summaries: list[dict]) -> dict | None:
    remote = [s for s in summaries if s["where"] == "gateway" and s["passes"]]
    if not remote:
        return None
    return sorted(remote, key=lambda s: (-s["right"], s["usd_per_1000"] if s["usd_per_1000"] == s["usd_per_1000"]
                                         else 1e9, s["median_s"]))[0]


def save(rows: list[dict], stage: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"interpret_{stage}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def by_kind(rows: list[dict]) -> None:
    models = list(dict.fromkeys(r["model"] for r in rows))
    kinds = list(dict.fromkeys(r["kind"] for r in sorted(rows, key=lambda r: r["id"])))
    print(f"\nRight answers by kind of request:\n{'kind':26s}" + "".join(f"{m[:14]:>15s}" for m in models))
    for kind in kinds:
        cells = []
        for m in models:
            rs = [r for r in rows if r["model"] == m and r["kind"] == kind]
            cells.append(f"{sum(r['outcome'] == 'right' for r in rs) / len(rs):15.0%}" if rs else f"{'-':>15s}")
        print(f"{kind:26s}" + "".join(cells))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["models", "screen", "compare", "test", "spoken", "all"])
    p.add_argument("models", nargs="*", help="compare, test, spoken: the gateway models")
    p.add_argument("--sample", type=int, default=40, help="screen: validation requests per model (default 40)")
    p.add_argument("--repeats", type=int, default=2, help="compare: times through the requests (default 2)")
    p.add_argument("--workers", type=int, default=4, help="requests at once to the gateway (default 4)")
    p.add_argument("--timeout", type=float, default=20.0, help="seconds to wait for each answer while measuring "
                   "(answers over 3 s still count as none, as in the chair)")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.stdout, OUT / "interpret_report.txt")
    print(f"\n=== interpret_eval.py {' '.join(argv if argv is not None else sys.argv[1:])}  "
          f"({time.strftime('%Y-%m-%d %H:%M')})")
    gateway = interpret.Interpreter(timeout=args.timeout)
    if gateway.missing() and not gateway.model and gateway.base and gateway.key:
        pass                                 # no model name needed to list or compare models
    elif not gateway.base or not gateway.key:
        sys.exit(f"Cannot reach the gateway: {gateway.missing()}")
    price = prices(gateway)

    if args.stage == "models":
        for m in gateway.models():
            p_in, p_out = price.get(m, (None, None))
            cost = f"US${p_in * 1e6:.2f} / ${p_out * 1e6:.2f} per million tokens in / out" if p_in is not None else ""
            print(f"{m:50s} {'(not chat)' if NOT_CHAT.search(m) else '':11s} {cost}")
        return

    if args.stage == "spoken":
        return spoken(args, gateway)
    if args.stage == "all":
        return everything(args, gateway, price)

    if args.stage == "screen":
        requests, names, repeats = load("validation", args.sample), args.models or chat_models(gateway), 1
        print(f"{len(names)} chat models, {len(requests)} validation requests each")
    elif args.stage == "compare":
        requests, names, repeats = load("validation"), args.models, args.repeats
    else:
        requests, names, repeats = load("test"), args.models, 1
    if args.stage != "screen" and not names:
        sys.exit("Name the models to try, such as: python webots/interpret_eval.py compare model-a model-b")

    rows, summaries = compared(names, requests, repeats, args, gateway, price, baselines=args.stage != "screen")
    table(summaries)
    if args.stage != "screen":
        by_kind(rows)
    best = choose(summaries)
    print(f"\nBy the rule, the {'best' if args.stage != 'test' else 'result'}: "
          f"{best['model'] if best else 'none passes (see the table)'}")
    print(f"Every request: {save(rows, args.stage)}")


def compared(names, requests, repeats, args, gateway, price, baselines=True):
    """Each model (and the two offline ones) on the requests: every request's result, and a summary each."""
    rows, summaries = [], []
    for name in names:
        started = time.monotonic()
        model = interpret.Interpreter(name, timeout=args.timeout)
        model.base, model.key = gateway.base, gateway.key
        got = run(model, requests, repeats, args.workers)
        rows += got
        summaries.append(summary(got, price))
        s = summaries[-1]
        print(f"  {name}: {s['right']:.0%} right, {s['wrong_moves']} wrong moves, {s['no_answer']:.0%} no answer "
              f"({time.monotonic() - started:.0f} s)", flush=True)
    if baselines:
        for baseline in (Exact(), Trained()):
            got = run(baseline, requests, 1)
            rows += got
            summaries.append(summary(got, price))
    return rows, summaries


def everything(args, gateway, price) -> None:
    """screen, compare the best 4 of it, test the chosen one, then spoken: one command for the lot."""
    names = chat_models(gateway)
    print(f"\n--- screen: {len(names)} chat models on 40 validation requests each")
    rows, summaries = compared(names, load("validation", args.sample), 1, args, gateway, price, baselines=False)
    table(summaries)
    save(rows, "screen")
    ranked = sorted([s for s in summaries if s["no_answer"] < 0.5],      # 40 requests are too few for the rule:
                    key=lambda s: (-s["right"], s["wrong_moves"], s["median_s"]))   # the most right go on
    shortlist = [s["model"] for s in ranked[:4]]
    print(f"\n--- compare: {', '.join(shortlist)} on all {len(load('validation'))} validation requests, "
          f"{args.repeats} times")
    rows, summaries = compared(shortlist, load("validation"), args.repeats, args, gateway, price)
    table(summaries)
    by_kind(rows)
    save(rows, "compare")
    best = choose(summaries)
    if best is None:
        print("\nNo model passes the rule on validation: see the tables. Stopping before the test.")
        return
    print(f"\nBy the rule, chosen on validation: {best['model']}")
    print(f"\n--- test: {best['model']} on the {len(load('test'))} test requests, once")
    rows, summaries = compared([best["model"]], load("test"), 1, args, gateway, price)
    table(summaries)
    by_kind(rows)
    save(rows, "test")
    if voice.tts_engine() is not None:
        print("\n--- spoken: the test requests read out, recognised and worked out as run.py does")
        args.models = [best["model"]]
        spoken(args, gateway)
    usage = sum(r["tokens_in"] for r in rows)
    print(f"\nDone. Chosen: {best['model']}. Put it in G19/.env as: export COMP3110_MODEL_NAME={best['model']}"
          f"  ({usage} prompt tokens in the test stage)")


class Tee:
    """Prints to the screen and to a file at once."""

    def __init__(self, screen, path: Path):
        self.screen, self.file = screen, path.open("a")

    def write(self, text):
        self.screen.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.screen.flush()


def spoken(args, gateway) -> None:
    """The test requests read out by the computer and recognised by Vosk, then worked out as run.py
    would: exact pass, words written down, stop words, then the model."""
    engine = voice.tts_engine()
    if engine is None or not args.models:
        sys.exit("Needs a model name and a text-to-speech program (macOS say)")
    recogniser, shown = voice.Recogniser(), screens()
    model = interpret.Interpreter(args.models[0], timeout=interpret.TIMEOUT_S)
    model.base, model.key = gateway.base, gateway.key
    rows = []
    for r in load("test"):
        if r["kind"] == "speech slip":            # their words are a mishearing already: read out, they are not
            continue                              # what a rider says (real mishearings happen here by themselves)
        screen = shown[r["screen"]]
        phrases = voice.Phrases(screen.places, screen.markers, numbered=bool(screen.listed))
        for name, interpreter in ((model.model, model), ("exact phrases (no model)", None)):
            result = voice.understand(recogniser, voice.speak(r["said"], engine), phrases, interpreter, screen)
            command = result.command
            if command is None or not result.how:
                got = "none"
            else:
                action = "stop" if command.action == "take" else command.action
                got = resolve(action, command.target, screen)
            meaning = result.meaning or Meaning(seconds=result.seconds)
            want = [resolve(*(w.split(":", 1) + [None])[:2], screen) for w in r["want"]]
            rows.append({"model": name, "where": "gateway" if interpreter else "offline", "repeat": 1, "id": r["id"],
                         "kind": r["kind"], "screen": r["screen"],
                         "said": r["said"], "written": result.written, "want": " | ".join(want), "got": got,
                         "outcome": outcome(got, want, Meaning(valid=meaning.valid, error=meaning.error,
                                                                    seconds=meaning.seconds)),
                         "how": result.how, "seconds": round(result.seconds, 3), "tokens_in": meaning.tokens[0],
                         "tokens_out": meaning.tokens[1], "invented": meaning.invented, "error": meaning.error,
                         "reply": meaning.reply, "raw": meaning.raw[:200]})
        print(f"  {r['said']!r} -> {rows[-2]['got']} (exact only: {rows[-1]['got']})", flush=True)
    price = prices(gateway)
    table([summary([x for x in rows if x["model"] == m], price) for m in dict.fromkeys(x["model"] for x in rows)])
    by_kind(rows)
    print(f"Every request: {save(rows, 'spoken')}")


if __name__ == "__main__":
    main()
