#!/usr/bin/env python3
"""Synthesize the prompts/v1 corpus through ElevenLabs (teacher distillation).

Progress lives in a SQLite DB (data/synth_state.db) — every clip is a row,
every state change is committed immediately, so a network hiccup, Ctrl-C, or
laptop sleep costs at most the clips currently in flight. Rerunning the same
command resumes exactly where it stopped and never re-bills a finished clip.

Subcommands:
  plan    seed/refresh the selection into the DB, print the todo summary (free)
  run     synthesize everything pending (resumable; respects --max-chars)
  status  progress + spend summary from the DB
  export  write clean metadata JSONL (roadmap schema) from all ok clips

Key properties (see TTS_RESEARCH_AND_ROADMAP.txt):
  * PCM wrapped as WAV -> data/raw/en/<voice>/<id>.wav. Default 24kHz to
    match the current corpus run; --sample-rate 44100 exists for a future
    max-fidelity archive (billing is per char, format is free).
  * DETERMINISTIC SELECTION: --fraction takes a seeded-shuffle PREFIX per
    category (whole conversations for prosody), so 0.5 now and 1.0 later
    synthesizes exactly the complement.
  * Canonical tags (<laugh>) adapted to v3 audio tags ([laughs]); both texts
    stored. Prosody turns send previous_text so dialogue prosody is coherent.
  * Per-category voice settings: expressive sets run at lower stability,
    accuracy-critical sets higher. Tune in CATEGORY_SETTINGS.

Usage:
  export ELEVENLABS_API_KEY=...
  python3 scripts/00_synthesize_teacher.py plan --voice-id <id> --fraction 0.5
  python3 scripts/00_synthesize_teacher.py run  --voice-id <id> --fraction 0.5 --max-chars 490000
"""

import argparse
import json
import os
import random
import re
import sqlite3
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    sys.exit("pip install requests")
try:
    from tqdm import tqdm
except ImportError:
    sys.exit("pip install tqdm")

SEED = 42
MIN_DURATION_S = 0.15  # anything shorter is a failed/empty generation
API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_{rate}"

# canonical corpus tag -> ElevenLabs v3 audio tag. 05_filter_qc.py verifies
# the event actually happened; adjust here if a tag underperforms per-voice.
TAG_ADAPTER = {
    "<laugh>": "[laughs]",
    "<chuckle>": "[chuckles]",
    "<sigh>": "[sighs]",
    "<gasp>": "[gasps]",
    "<whisper>": "[whispers]",
    "<pause>": "[pause]",
}

# stability: lower = more expressive/varied delivery (tags land, prosody
# moves), higher = more consistent/accurate reading. similarity + speaker
# boost high everywhere: voice consistency is what the student learns.
# speed stays 1.0 — never warp training data.
BASE_SETTINGS = {"similarity_boost": 0.85, "use_speaker_boost": True, "speed": 1.0}
CATEGORY_SETTINGS = {
    "tags":          {"stability": 0.30},  # expressiveness is the whole point
    "prosody":       {"stability": 0.40},  # conversational movement
    "edge":          {"stability": 0.50},
    "general":       {"stability": 0.50},
    "phonetic":      {"stability": 0.60},  # accuracy-critical
    "normalization": {"stability": 0.60},  # numbers must be read exactly
}

CORPUS_FILES = ["arctic.jsonl", "harvard.jsonl", "general.jsonl",
                "normalization.jsonl", "prosody.jsonl", "tags.jsonl", "edge.jsonl"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
  id TEXT PRIMARY KEY,
  category TEXT, conversation_id TEXT, turn_index INTEGER,
  text TEXT, synth_text TEXT, prev_text TEXT,
  status TEXT DEFAULT 'pending',            -- pending | ok | error
  attempts INTEGER DEFAULT 0,
  chars INTEGER, duration_s REAL, audio_path TEXT,
  http_code INTEGER, error TEXT,
  voice TEXT, model TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_status ON clips(status);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def adapt_text(text: str) -> str:
    for canon, eleven in TAG_ADAPTER.items():
        text = text.replace(canon, eleven)
    return text


def select_fraction(rows: list[dict], fraction: float, rng: random.Random) -> list[dict]:
    """Seeded-shuffle prefix; prosody conversations are never split."""
    if rows and rows[0].get("conversation_id"):
        convs: dict[str, list[dict]] = {}
        for r in rows:
            convs.setdefault(r["conversation_id"], []).append(r)
        keys = sorted(convs)
        rng.shuffle(keys)
        take = keys[: round(len(keys) * fraction)]
        return [r for k in take for r in sorted(convs[k], key=lambda x: x["turn_index"])]
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in idx[: round(len(rows) * fraction)]]


def wav_write(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
        f.write(b"data" + struct.pack("<I", len(pcm)) + pcm)
    tmp.rename(path)  # atomic: a .wav either exists complete or not at all


# ------------------------------------------------------------------- db ----

def db_open(data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(data_dir / "synth_state.db")
    con.executescript(SCHEMA)
    con.row_factory = sqlite3.Row
    return con


def db_seed(con: sqlite3.Connection, prompts: Path, fraction: float) -> None:
    """Insert selected clips as pending. INSERT OR IGNORE => growing the
    fraction later only ADDS clips; finished work is untouched."""
    for fname in CORPUS_FILES:
        rows = read_jsonl(prompts / fname)
        rows.sort(key=lambda r: r["id"])
        picked = select_fraction(rows, fraction, random.Random(f"{SEED}:{fname}"))
        prev_by_conv: dict[str, str] = {}
        for r in picked:
            synth = adapt_text(r["text"])
            prev = None
            cid = r.get("conversation_id")
            if cid:
                prev = prev_by_conv.get(cid)
                prev_by_conv[cid] = synth
            con.execute(
                "INSERT OR IGNORE INTO clips (id, category, conversation_id, turn_index,"
                " text, synth_text, prev_text, chars, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (r["id"], r["category"], cid, r.get("turn_index"),
                 r["text"], synth, prev, len(synth), now()))
    con.commit()


def db_verify_files(con: sqlite3.Connection, root: Path) -> int:
    """Clips marked ok whose WAV vanished go back to pending."""
    lost = 0
    for row in con.execute("SELECT id, audio_path FROM clips WHERE status='ok'"):
        if not row["audio_path"] or not (root / row["audio_path"]).exists():
            con.execute("UPDATE clips SET status='pending', updated_at=? WHERE id=?",
                        (now(), row["id"]))
            lost += 1
    con.commit()
    return lost


def summary(con: sqlite3.Connection) -> None:
    print(f"\n  {'category':<15} {'total':>6} {'ok':>6} {'error':>6} {'pending':>8} {'todo chars':>11}")
    for r in con.execute(
            "SELECT category, COUNT(*) n,"
            " SUM(status='ok') ok, SUM(status='error') err,"
            " SUM(status='pending') pend,"
            " SUM(CASE WHEN status!='ok' THEN chars ELSE 0 END) todo_chars"
            " FROM clips GROUP BY category ORDER BY category"):
        print(f"  {r['category']:<15} {r['n']:>6} {r['ok'] or 0:>6}"
              f" {r['err'] or 0:>6} {r['pend'] or 0:>8} {r['todo_chars'] or 0:>11,}")
    t = con.execute(
        "SELECT COUNT(*) n, SUM(status='ok') ok,"
        " SUM(CASE WHEN status='ok' THEN chars ELSE 0 END) spent,"
        " SUM(CASE WHEN status='ok' THEN duration_s ELSE 0 END) secs,"
        " SUM(CASE WHEN status!='ok' THEN chars ELSE 0 END) todo FROM clips").fetchone()
    spent, todo = t["spent"] or 0, t["todo"] or 0
    print(f"\n  billed so far: {spent:,} chars, {(t['secs'] or 0) / 3600:.2f} audio hours"
          f" ({t['ok'] or 0:,}/{t['n']:,} clips)")
    print(f"  remaining:     {todo:,} chars"
          f" (~{todo / 55_000:.1f} h, ~{todo / 55_000 * 0.3175:.2f} GB PCM at 44.1k)")


# ------------------------------------------------------------------ http ----

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, connect=5, read=3, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["POST"], respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    return s


def synthesize(session: requests.Session, clip: sqlite3.Row, args, api_key: str) -> dict:
    body = {
        "text": clip["synth_text"],
        "model_id": args.model,
        "voice_settings": {**BASE_SETTINGS,
                           **CATEGORY_SETTINGS.get(clip["category"], {"stability": 0.5})},
    }
    # NOTE: eleven_v3 rejects previous_text/next_text ("unsupported_model").
    # prev_text stays in the DB as dataset metadata (CSM-style context
    # training later), it just can't condition the teacher's delivery.
    try:
        resp = session.post(API.format(voice=args.voice_id, rate=args.sample_rate),
                            headers={"xi-api-key": api_key}, json=body, timeout=180)
    except requests.RequestException as e:
        return {"id": clip["id"], "status": "error", "http_code": 0,
                "error": f"{type(e).__name__}: {e}"[:300]}
    if resp.status_code != 200:
        return {"id": clip["id"], "status": "error", "http_code": resp.status_code,
                "error": resp.text[:300]}
    pcm = resp.content
    duration = len(pcm) / (args.sample_rate * 2)
    if duration < MIN_DURATION_S:
        return {"id": clip["id"], "status": "error", "http_code": 200,
                "error": f"empty audio ({duration:.3f}s)"}
    wav = args.data / "raw" / "en" / args.voice_name / f"{clip['id']}.wav"
    wav_write(wav, pcm, args.sample_rate)
    return {"id": clip["id"], "status": "ok", "http_code": 200, "error": None,
            "duration_s": round(duration, 3),
            "audio_path": str(wav.relative_to(args.data))}


# -------------------------------------------------------------- commands ----

def cmd_run(con: sqlite3.Connection, args) -> None:
    api_key = os.environ.get("ELEVENLABS_API_KEY") or sys.exit("set ELEVENLABS_API_KEY")
    lost = db_verify_files(con, args.data)
    if lost:
        print(f"  note: {lost} ok clips had missing WAVs -> reset to pending")
    todo = con.execute(
        "SELECT * FROM clips WHERE status!='ok'"
        " AND (attempts < ? OR status='pending') ORDER BY id", (args.max_attempts,)
    ).fetchall()
    if not todo:
        print("nothing to do — everything selected is synthesized.")
        return

    budget = args.max_chars
    if budget:
        trimmed, planned = [], 0
        for c in todo:
            if planned + c["chars"] > budget:
                break
            planned += c["chars"]
            trimmed.append(c)
        skipped = len(todo) - len(trimmed)
        todo = trimmed
        if skipped:
            print(f"  budget: --max-chars {budget:,} covers {len(todo):,} clips this run"
                  f" ({skipped:,} deferred to the next run/month)")

    session = make_session()
    spent = ok = err = 0
    bar = tqdm(total=len(todo), unit="clip", dynamic_ncols=True,
               desc=f"synthesizing ({args.voice_name})")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(synthesize, session, c, args, api_key): c for c in todo}
            for fut in as_completed(futures):
                clip, res = futures[fut], fut.result()
                con.execute(
                    "UPDATE clips SET status=?, attempts=attempts+1, http_code=?,"
                    " error=?, duration_s=?, audio_path=?, voice=?, model=?, updated_at=?"
                    " WHERE id=?",
                    (res["status"], res["http_code"], res["error"],
                     res.get("duration_s"), res.get("audio_path"),
                     args.voice_name, args.model, now(), res["id"]))
                con.commit()  # every clip durable immediately
                if res["status"] == "ok":
                    ok += 1
                    spent += clip["chars"]
                else:
                    err += 1
                    if res["http_code"] == 401:
                        raise RuntimeError("401 unauthorized — check ELEVENLABS_API_KEY")
                    tqdm.write(f"  error {res['id']}: [{res['http_code']}] {res['error'][:120]}")
                bar.update(1)
                bar.set_postfix(ok=ok, err=err, chars=f"{spent:,}")
    except (KeyboardInterrupt, RuntimeError) as e:
        tqdm.write(f"\nstopping ({e if str(e) else 'interrupted'}) — progress is saved;"
                   " rerun the same command to resume.")
    finally:
        bar.close()
    print(f"\nthis run: {ok:,} ok, {err} errors, {spent:,} chars billed.")
    summary(con)


def cmd_export(con: sqlite3.Connection, args) -> None:
    out = args.data / "metadata" / "en" / f"{args.voice_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for r in con.execute("SELECT * FROM clips WHERE status='ok' ORDER BY id"):
            row = dict(r)
            rec = {
                "id": row["id"], "text": row["text"], "synth_text": row["synth_text"],
                "audio": f"data/{row['audio_path']}", "category": row["category"],
                "speaker": row["voice"], "lang": "en", "source": "elevenlabs",
                "prompt_version": "v1", "conversation_id": row["conversation_id"],
                "turn_index": row["turn_index"], "duration_s": row["duration_s"],
                "model": row["model"],
            }
            if row.get("qc_passed") is not None:  # qc columns exist once 05 has run
                rec["qc"] = {"wer": row["qc_wer"], "rate": row["qc_rate"],
                             "event": row["qc_event"], "passed": bool(row["qc_passed"]),
                             "notes": row["qc_notes"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"exported {n:,} ok clips -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "run", "status", "export"])
    ap.add_argument("--voice-id", help="ElevenLabs voice id (plan/run)")
    ap.add_argument("--voice-name", default=None,
                    help="speaker label / folder (default: eleven_<id prefix>)")
    ap.add_argument("--model", default="eleven_v3")
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--max-chars", type=int, default=None,
                    help="character budget for THIS run (monthly credits)")
    ap.add_argument("--max-attempts", type=int, default=4,
                    help="give up on a clip after this many failed runs")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sample-rate", type=int, default=24000, choices=[24000, 44100],
                    help="PCM rate to request; keep raw at max, downsample in processed/")
    ap.add_argument("--prompts", type=Path, default=Path("prompts/v1"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()

    con = db_open(args.data)
    if args.command in ("plan", "run"):
        if not args.voice_id:
            ap.error(f"{args.command} requires --voice-id")
        args.voice_name = args.voice_name or f"eleven_{args.voice_id[:8]}"
        db_seed(con, args.prompts, args.fraction)
    else:
        args.voice_name = args.voice_name or "eleven"

    if args.command == "plan" or args.command == "status":
        summary(con)
    elif args.command == "run":
        cmd_run(con, args)
    elif args.command == "export":
        cmd_export(con, args)


if __name__ == "__main__":
    main()
