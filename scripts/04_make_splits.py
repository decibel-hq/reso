#!/usr/bin/env python3
"""Build train / diagnostic-heldout splits from QC-passed, encoded clips.

Rules (see TTS_RESEARCH_AND_ROADMAP.txt):
  * only clips with qc_passed=1 AND an encoded .npz enter any split
  * prosody splits by WHOLE conversation — turns never straddle the boundary
  * heldout is DIAGNOSTIC: fixed per-category quotas so each eval run reads
    as "numbers regressed, tags fine", not one opaque score
  * seeded => stable across reruns; splits/ is committed to git (ids + text
    only, no audio)

Usage:
  ./.venv/bin/python scripts/04_make_splits.py
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path

SEED = 42
HELDOUT_QUOTA = {  # clips per category (prosody counts conversations)
    "phonetic": 30, "general": 40, "normalization": 40,
    "tags": 36, "edge": 30, "prosody_convs": 8,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("splits"))
    args = ap.parse_args()

    enc = {p.stem for p in (args.data / "encoded" / "orpheus_snac").glob("*.npz")}
    con = sqlite3.connect(args.data / "synth_state.db")
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT id, text, category, conversation_id, turn_index, duration_s"
        " FROM clips WHERE status='ok' AND qc_passed=1 ORDER BY id")
        if r["id"] in enc]
    if not rows:
        raise SystemExit("nothing eligible — run QC and 03_encode_snac.py first")

    rng = random.Random(SEED)
    heldout, train = [], []
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    for cat, items in sorted(by_cat.items()):
        if cat == "prosody":
            convs: dict[str, list[dict]] = {}
            for r in items:
                convs.setdefault(r["conversation_id"], []).append(r)
            keys = sorted(convs)
            rng.shuffle(keys)
            held_keys = set(keys[: HELDOUT_QUOTA["prosody_convs"]])
            for k in keys:
                (heldout if k in held_keys else train).extend(
                    sorted(convs[k], key=lambda x: x["turn_index"]))
        else:
            items = items[:]
            rng.shuffle(items)
            q = min(HELDOUT_QUOTA.get(cat, 20), len(items) // 10)
            heldout.extend(items[:q])
            train.extend(items[q:])

    args.out.mkdir(exist_ok=True)
    for name, split in (("train", train), ("heldout", heldout)):
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in sorted(split, key=lambda x: x["id"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours = sum(r["duration_s"] or 0 for r in train) / 3600
    print(f"train: {len(train):,} clips ({hours:.1f} h) | heldout: {len(heldout):,}")
    for cat in sorted(by_cat):
        h = sum(1 for r in heldout if r["category"] == cat)
        print(f"  {cat:<14} train {sum(1 for r in train if r['category'] == cat):>5}  heldout {h:>3}")


if __name__ == "__main__":
    main()
