#!/usr/bin/env python3
"""QC-filter synthesized clips: ASR WER + duration sanity + tag-event hints.

Never deletes audio. Verdicts are written into synth_state.db (qc_* columns)
and data/qc/qc_results.jsonl; the split-maker filters on qc_passed later, so
thresholds can be retuned without re-synthesizing anything.

Checks per clip:
  1. WER — faster-whisper transcript vs reference text (tags stripped, both
     sides lightly normalized). Per-category thresholds: the normalization
     category gets a looser default because ASR formats numbers differently
     ("$5" vs "five dollars") even when speech is perfect — tune with the
     printed WER distribution, don't trust absolute values blindly there.
  2. Duration / speech-rate — chars-per-second must be plausible (catches
     truncations, rambles, silent tails).
  3. Tag events (tags category only) — WEAK verification for now: Whisper
     hints like "(laughs)", "haha", bracketed events in the transcript, plus
     a duration-surplus heuristic (a laugh adds time but no reference chars).
     A proper audio-event classifier (PANNs/AST) can replace this later;
     failures here mark qc_notes, they do NOT fail the clip on their own yet.

Resumable: clips with an existing verdict are skipped unless --redo.

Usage:
  ./.venv/bin/pip install faster-whisper        # once; ~1.5 GB model download
  ./.venv/bin/python scripts/05_filter_qc.py run --voice-name adam
  ./.venv/bin/python scripts/05_filter_qc.py report
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("pip install tqdm")

CANONICAL_TAGS = ["<laugh>", "<chuckle>", "<sigh>", "<gasp>", "<whisper>", "<pause>"]

# WER above this fails the clip; per-category overrides of the default.
WER_THRESHOLDS = {"default": 0.10, "normalization": 0.30, "tags": 0.20, "edge": 0.20}
# plausible speech-rate band, chars of reference text per second of audio.
# edge one-worders legitimately run slow (long pauses around "No."), tags
# clips carry non-speech time; both get wider bands.
RATE_BANDS = {"default": (6.0, 25.0), "edge": (1.0, 30.0), "tags": (4.0, 25.0)}

LAUGH_HINTS = re.compile(r"\(laugh|\[laugh|haha|hehe|\(chuckle|\[chuckle", re.I)
EVENT_HINTS = {
    "<laugh>": LAUGH_HINTS,
    "<chuckle>": LAUGH_HINTS,
    "<sigh>": re.compile(r"\(sigh|\[sigh|\bugh\b|\bhah\b", re.I),
    "<gasp>": re.compile(r"\(gasp|\[gasp|\boh\b my|\bwhoa\b", re.I),
    "<whisper>": None,   # not detectable from transcript; duration heuristic only
    "<pause>": None,     # verified via duration surplus only
}

QC_COLUMNS = """
ALTER TABLE clips ADD COLUMN qc_wer REAL;
ALTER TABLE clips ADD COLUMN qc_asr_text TEXT;
ALTER TABLE clips ADD COLUMN qc_rate REAL;
ALTER TABLE clips ADD COLUMN qc_event TEXT;
ALTER TABLE clips ADD COLUMN qc_passed INTEGER;
ALTER TABLE clips ADD COLUMN qc_notes TEXT;
"""


def norm_words(text: str) -> list[str]:
    for t in CANONICAL_TAGS:
        text = text.replace(t, " ")
    text = re.sub(r"\[[a-z ]+\]", " ", text.lower())      # elevenlabs-style tags
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [w for w in text.split() if w]


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / len(ref)


def db_open(data_dir: Path) -> sqlite3.Connection:
    con = sqlite3.connect(data_dir / "synth_state.db")
    con.row_factory = sqlite3.Row
    have = {r[1] for r in con.execute("PRAGMA table_info(clips)")}
    for stmt in QC_COLUMNS.strip().split(";"):
        stmt = stmt.strip()
        if stmt and stmt.split()[-2] not in have:
            con.execute(stmt)
    con.commit()
    return con


def judge(row: sqlite3.Row, asr_text: str) -> dict:
    cat = row["category"]
    ref = norm_words(row["text"])
    hyp = norm_words(asr_text)
    w = wer(ref, hyp)
    notes = []

    lo, hi = RATE_BANDS.get(cat, RATE_BANDS["default"])
    speech_chars = len(re.sub(r"<[a-z]+>", "", row["text"]))
    rate = speech_chars / row["duration_s"] if row["duration_s"] else 0.0
    rate_ok = lo <= rate <= hi
    if not rate_ok:
        notes.append(f"rate {rate:.1f} outside [{lo},{hi}]")

    event = None
    if cat == "tags":
        tag = next((t for t in CANONICAL_TAGS if t in row["text"]), None)
        hint = EVENT_HINTS.get(tag)
        transcript_hit = bool(hint and hint.search(asr_text))
        # duration surplus: event time beyond what the words alone need
        surplus = row["duration_s"] - speech_chars / 14.0
        event = "transcript" if transcript_hit else ("surplus" if surplus > 0.4 else "none")
        if event == "none":
            notes.append(f"no event evidence for {tag}")

    threshold = WER_THRESHOLDS.get(cat, WER_THRESHOLDS["default"])
    passed = w <= threshold and rate_ok  # event weakness noted, not fatal (yet)
    return {"wer": round(w, 4), "rate": round(rate, 2), "event": event,
            "passed": int(passed), "notes": "; ".join(notes) or None,
            "asr": asr_text.strip()}


def cmd_run(args) -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("pip install faster-whisper  (inside the venv)")
    con = db_open(args.data)
    where = "status='ok'" + ("" if args.redo else " AND qc_passed IS NULL")
    rows = con.execute(f"SELECT * FROM clips WHERE {where} ORDER BY id").fetchall()
    if not rows:
        print("nothing to QC.")
        return
    print(f"loading whisper model '{args.whisper_model}' ({args.compute_type},"
          f" {args.qc_threads} parallel) ...")
    model = WhisperModel(args.whisper_model, device=args.device,
                         compute_type=args.compute_type,
                         cpu_threads=max(2, (os.cpu_count() or 8) // args.qc_threads),
                         num_workers=args.qc_threads)
    qc_dir = args.data / "qc"
    qc_dir.mkdir(exist_ok=True)
    out = (qc_dir / "qc_results.jsonl").open("a", encoding="utf-8")
    passed = failed = 0

    def transcribe(row):
        wav = args.data / row["audio_path"]
        if not wav.exists():  # FLAC mirror (bandwidth-friendly upload) fallback
            wav = wav.with_suffix(".flac")
        if not wav.exists():
            return row, None
        segments, _ = model.transcribe(str(wav), language="en",
                                       beam_size=args.beam_size,
                                       condition_on_previous_text=False)
        return row, " ".join(s.text for s in segments)

    bar = tqdm(total=len(rows), unit="clip", desc="qc", dynamic_ncols=True)
    with ThreadPoolExecutor(max_workers=args.qc_threads) as pool:
        futures = [pool.submit(transcribe, r) for r in rows]
        for fut in as_completed(futures):
            row, asr_text = fut.result()
            bar.update(1)
            if asr_text is None:
                continue
            v = judge(row, asr_text)
            con.execute("UPDATE clips SET qc_wer=?, qc_asr_text=?, qc_rate=?,"
                        " qc_event=?, qc_passed=?, qc_notes=? WHERE id=?",
                        (v["wer"], v["asr"], v["rate"], v["event"], v["passed"],
                         v["notes"], row["id"]))
            con.commit()
            out.write(json.dumps({"id": row["id"], **v}, ensure_ascii=False) + "\n")
            passed += v["passed"]
            failed += not v["passed"]
            bar.set_postfix(passed=passed, failed=failed)
    bar.close()
    out.close()
    print(f"\nqc done: {passed:,} passed, {failed:,} failed")
    cmd_report(args)


def cmd_report(args) -> None:
    con = db_open(args.data)
    print(f"\n  {'category':<15} {'qc''d':>6} {'passed':>7} {'failed':>7} {'median WER':>11} {'p90 WER':>8}")
    for r in con.execute("SELECT category, COUNT(*) n, SUM(qc_passed) p"
                         " FROM clips WHERE qc_passed IS NOT NULL GROUP BY category"):
        wers = [x[0] for x in con.execute(
            "SELECT qc_wer FROM clips WHERE category=? AND qc_wer IS NOT NULL"
            " ORDER BY qc_wer", (r["category"],))]
        med = wers[len(wers) // 2] if wers else 0
        p90 = wers[int(len(wers) * 0.9)] if wers else 0
        print(f"  {r['category']:<15} {r['n']:>6} {r['p'] or 0:>7}"
              f" {r['n'] - (r['p'] or 0):>7} {med:>11.3f} {p90:>8.3f}")
    ev = dict(con.execute("SELECT qc_event, COUNT(*) FROM clips"
                          " WHERE category='tags' AND qc_event IS NOT NULL"
                          " GROUP BY qc_event").fetchall())
    if ev:
        print(f"\n  tag-event evidence: {ev}")
    fails = con.execute("SELECT id, qc_wer, qc_notes FROM clips WHERE qc_passed=0"
                        " ORDER BY qc_wer DESC LIMIT 8").fetchall()
    if fails:
        print("\n  worst failures:")
        for f in fails:
            print(f"    {f['id']}: WER {f['qc_wer']:.2f}  {f['qc_notes'] or ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--voice-name", default="adam")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--whisper-model", default="distil-large-v3",
                    help="distil-large-v3 (fast, en) | large-v3 (best) | small (quick pass)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--redo", action="store_true", help="re-QC clips that already have verdicts")
    ap.add_argument("--beam-size", type=int, default=1,
                    help="1 = greedy, plenty for pass/fail filtering; 5 = careful")
    ap.add_argument("--qc-threads", type=int, default=3,
                    help="parallel transcription workers")
    args = ap.parse_args()
    cmd_run(args) if args.command == "run" else cmd_report(args)


if __name__ == "__main__":
    main()
