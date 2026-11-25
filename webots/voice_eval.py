"""How well the voice commands (voice.py) are understood: recorded or computer voices, in quiet and in
noise, with every phrase the hospital offers listened for at once (more than a real trip ever has).

    python webots/voice_eval.py record scott            # say each phrase when asked: data/voice/scott/
    python webots/voice_eval.py run                     # every speaker recorded, quiet and with noise
    python webots/voice_eval.py run --tts               # the computer's English voices instead (macOS say)
    python webots/voice_eval.py run --noise ward.wav --snr 20 10 5   # a recording of real noise, at these levels

Each phrase is a command, a place on the hospital's signs, a marker, a number on the list, or
something that is not a command at all ("what time is it"). For each it counts:
    right      the command meant was acted on
    wrong      a different command was acted on: the dangerous one (the chair goes somewhere unasked)
    stopped    the chair stopped instead (a stop word heard by mistake): safe, the rider says it again
    rejected   nothing was done (not heard as a whole phrase, or not heard surely enough to move):
               the rider says it again
    stop       of the stops, how many stopped the chair
    false      of the things that are not commands, how many were acted on
and how long recognition took after K was let go. Noise is mixed in at each signal-to-noise ratio
(dB: 20 is a quiet ward, 10 a busy corridor, 5 shouting distance). Without --noise it is pink noise
(more low than high, like a crowd or air conditioning). Every phrase's result goes to
data/voice/results.csv (or --out), and the table to the screen.
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webots import voice  # noqa: E402

DATA = ROOT / "data" / "voice"
RECORD_S = 3.0
NOT_COMMANDS = ["what time is it", "hello there", "I need a coffee", "turn left please", "where is the bathroom",
                "go away", "take a photo", "the weather is nice today", "can you hear me", "thank you very much",
                "excuse me", "is this the right way", "good morning", "I am going home", "call the nurse"]


def phrase_set(sample: int | None = None, seed: int = 0) -> tuple[voice.Phrases, list[tuple[str, str]]]:
    """Everything listened for (the hospital's places, all markers, the numbers) and the phrases to say,
    each as (what is read out, kind). Places are read as written on the sign ("Ward 4A"), as a person would."""
    from webots import landmarks, signs
    hospital = signs._hospital()
    places = sorted({p.name for p in hospital} | {p.group for p in hospital if p.group})
    markers = [f"{c} {s}" for c in landmarks.PALETTE for s in landmarks.SHAPES]
    phrases = voice.Phrases(places, markers, numbered=True)
    rng = random.Random(seed)
    said = [(c, "stop" if voice.COMMANDS[c] == "stop" else "command") for c in voice.COMMANDS]
    said += [(f"go to {p}", "place") for p in places]
    said += [(f"go to the {m}", "marker") for m in rng.sample(markers, 8)]
    said += [(f"go to number {voice.number_words(n)}", "number") for n in (1, 3, 5, 7, 9)]
    said += [(t, "not a command") for t in NOT_COMMANDS]
    if sample:
        said = rng.sample(said, min(sample, len(said)))
    return phrases, said


def expected(phrases: voice.Phrases, text: str) -> voice.Command | None:
    """The command a phrase read out means (None for the ones that are not commands)."""
    return phrases.match(voice.normal(text)) or phrases.match(voice.spoken(text))


def same(got: voice.Command, want: voice.Command | None) -> bool:
    """Whether the command acted on is the one meant. Two names the chair takes for the same place
    ("Outpatient Clinic" and "Outpatient Clinics", explore.same_name) are the same place."""
    from webots import explore
    if want is None or got.action != want.action:
        return False
    return got.target == want.target or (got.action == "place" and explore.same_name(got.target, want.target))


def slug(text: str) -> str:
    return "_".join(voice.spoken(text).split())[:60] or "phrase"


def record(speaker: str, takes: int, device=None) -> None:
    """Asks for each phrase in turn and records it (RECORD_S seconds from Enter)."""
    import sounddevice as sd
    _, said = phrase_set()
    folder = DATA / speaker
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{len(said) * takes} phrases. Press Enter, then say the phrase in a normal voice "
          f"(you have {RECORD_S:.0f} s). Type s and Enter to skip one, q to stop.")
    for take, (text, kind) in [(t, x) for t in range(takes) for x in said]:
        answer = input(f'  say: "{text}"   ').strip().lower()
        if answer == "q":
            break
        if answer == "s":
            continue
        pcm = sd.rec(int(RECORD_S * voice.RATE), samplerate=voice.RATE, channels=1, dtype="int16", device=device)
        sd.wait()
        path = folder / f"{slug(text)}_{take + 1}.wav"
        voice.write_wav(path, pcm.tobytes())
        rows.append({"file": path.name, "text": text, "kind": kind})
    manifest = folder / "manifest.csv"
    old = list(csv.DictReader(manifest.open())) if manifest.exists() else []
    keep = {r["file"]: r for r in old + rows}
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "text", "kind"])
        w.writeheader()
        w.writerows(keep.values())
    print(f"{len(rows)} recordings in {folder}")


MAC_VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa", "Alex", "Fred"]   # US, UK, AU, IE, IN, ZA
ESPEAK_VOICES = ["en-us", "en", "en-gb-x-rp", "en-us+f3", "en-029", "en-gb-scotland"]


def computer_voices(limit: int) -> list[tuple[str, str | None]]:
    """(engine, voice) for up to `limit` English voices of this computer's text-to-speech, each with
    a different accent where it has them."""
    engine = voice.tts_engine()
    if engine == "say":
        listed = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        have = {line.split()[0] for line in listed.splitlines() if line.strip()}
        return [("say", n) for n in MAC_VOICES if n in have][:limit] or [("say", None)]
    if engine:
        return [(engine, v) for v in ESPEAK_VOICES[:limit]]
    return []


def speakers(args) -> list[tuple[str, list[tuple[bytes, str, str]]]]:
    """Each speaker's recordings as (sound, text, kind)."""
    phrases, said = phrase_set(args.sample, args.seed)
    found = []
    if args.tts:
        for engine, name in computer_voices(args.voices):
            print(f"  {name}: speaking {len(said)} phrases", flush=True)
            found.append((f"{name or 'default'} ({engine})",
                          [(voice.speak(t, engine, name), t, k) for t, k in said]))
        if not found:
            sys.exit("No text-to-speech program here (macOS has say). Record yourself: voice_eval.py record <name>")
        return found
    for manifest in sorted(DATA.glob("*/manifest.csv")):
        rows = list(csv.DictReader(manifest.open()))
        found.append((manifest.parent.name, [(voice.read_wav(manifest.parent / r["file"]), r["text"], r["kind"])
                                             for r in rows if (manifest.parent / r["file"]).exists()]))
    if not found:
        sys.exit(f"No recordings in {DATA}: python webots/voice_eval.py record <name>, or run --tts")
    return found


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum /= np.sqrt(np.maximum(np.arange(len(spectrum)), 1))
    return np.fft.irfft(spectrum, n)


def with_noise(pcm: bytes, snr_db: float | None, noise: np.ndarray | None, rng: np.random.Generator) -> bytes:
    """The speech with noise mixed in at snr_db (None: as it is). The noise is matched to the speech's
    loudness while it is speaking (its loudest half), so 10 dB means the same for every phrase."""
    if snr_db is None:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if noise is None:
        n = pink_noise(len(x), rng)
    else:
        start = rng.integers(0, max(1, len(noise) - len(x)))
        n = np.resize(noise[start:start + len(x)], len(x))
    frames = x[: len(x) // 160 * 160].reshape(-1, 160)
    loud = np.sort((frames ** 2).mean(axis=1))[len(frames) // 2:]
    speech = loud.mean() if len(loud) else (x ** 2).mean()
    n *= np.sqrt(speech / max((n ** 2).mean(), 1e-9) / 10 ** (snr_db / 10))
    return np.clip(x + n, -32768, 32767).astype(np.int16).tobytes()


def evaluate(args) -> None:
    phrases, _ = phrase_set()
    listened = phrases.phrases()
    recogniser = voice.Recogniser()
    noise = np.frombuffer(voice.read_wav(args.noise), dtype=np.int16).astype(np.float64) if args.noise else None
    levels = ([] if args.no_quiet else [None]) + list(args.snr)
    rng = np.random.default_rng(args.seed)
    rows = []
    for speaker, recordings in speakers(args):
        for snr in levels:
            for pcm, text, kind in recordings:
                want = expected(phrases, text)
                result = voice.understand(recogniser, with_noise(pcm, snr, noise, rng), phrases)  # as run.py, offline
                heard, confidence, took, got = result.heard, result.confidence, result.seconds, result.command
                acted = result.how in ("exact", "stop word")
                if not acted:
                    outcome = "rejected"
                elif same(got, want):
                    outcome = "right"
                elif got.action in ("stop", "take"):
                    outcome = "stopped"            # a stop word heard by mistake: the chair stops, which is safe
                else:
                    outcome = "wrong"
                rows.append({"speaker": speaker, "snr_db": "quiet" if snr is None else f"{snr:g}", "kind": kind,
                             "said": text, "heard": heard, "written": result.written, "confidence": round(confidence, 3),
                             "acted_on": f"{got.action} {got.target or ''}".strip() if acted else "",
                             "outcome": outcome, "seconds": round(took, 3)})
    out = Path(args.out) if args.out else DATA / ("results_tts.csv" if args.tts else "results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(listened)} phrases listened for. Commands, places, markers and numbers:")
    print(f"{'speaker':28s} {'noise':>6s} {'right':>7s} {'wrong':>7s} {'stopped':>8s} {'rejected':>9s} {'stop':>7s} "
          f"{'false':>7s} {'median s':>9s}")
    for speaker in dict.fromkeys(r["speaker"] for r in rows):
        for snr in dict.fromkeys(r["snr_db"] for r in rows):
            print(summary_line(speaker[:28], snr, [r for r in rows if r["speaker"] == speaker and r["snr_db"] == snr]))
    if len({r["speaker"] for r in rows}) > 1:
        for snr in dict.fromkeys(r["snr_db"] for r in rows):
            print(summary_line("all", snr, [r for r in rows if r["snr_db"] == snr]))
    print(f"Every phrase: {out}")


def summary_line(speaker: str, snr, rows: list[dict]) -> str:
    commands = [r for r in rows if r["kind"] != "not a command"]
    stops = [r for r in rows if r["kind"] == "stop"]
    others = [r for r in rows if r["kind"] == "not a command"]

    def share(part, whole):
        return f"{len(part) / len(whole):6.0%}" if whole else "     -"
    return (f"{speaker:28s} {str(snr):>6s} {share([r for r in commands if r['outcome'] == 'right'], commands):>7s} "
            f"{share([r for r in commands if r['outcome'] == 'wrong'], commands):>7s} "
            f"{share([r for r in commands if r['outcome'] == 'stopped'], commands):>8s} "
            f"{share([r for r in commands if r['outcome'] == 'rejected'], commands):>9s} "
            f"{share([r for r in stops if r['outcome'] == 'right'], stops):>7s} "
            f"{share([r for r in others if r['outcome'] != 'rejected'], others):>7s} "
            f"{np.median([r['seconds'] for r in rows]) if rows else 0:9.2f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    rec = sub.add_parser("record", help="record yourself saying every phrase")
    rec.add_argument("speaker", help="a name for the folder, such as scott")
    rec.add_argument("--takes", type=int, default=1, help="times through the list (default 1)")
    rec.add_argument("--device", help="microphone (default: the computer's own)")
    run = sub.add_parser("run", help="recognise every recording and count")
    run.add_argument("--tts", action="store_true", help="the computer's voices instead of recordings")
    run.add_argument("--voices", type=int, default=6, help="with --tts: how many voices (default 6)")
    run.add_argument("--noise", help="a WAV of real background noise to mix in (default: pink noise)")
    run.add_argument("--snr", type=float, nargs="*", default=[20, 10, 5], help="signal-to-noise ratios in dB "
                     "besides quiet (default 20 10 5)")
    run.add_argument("--no-quiet", action="store_true", help="only the noise levels, not quiet (to split a long run)")
    run.add_argument("--sample", type=int, help="only this many phrases, picked at random")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", help="CSV of every phrase (default data/voice/results.csv)")
    args = p.parse_args(argv)
    try:
        if args.what == "record":
            record(args.speaker, args.takes, args.device)
        else:
            evaluate(args)
    except voice.VoiceUnavailable as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
