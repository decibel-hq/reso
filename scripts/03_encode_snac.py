#!/usr/bin/env python3
"""SNAC-encode QC-passed clips into the Orpheus token cache.

Reads clip list + verdicts from data/synth_state.db, encodes audio with
SNAC 24kHz, stores RAW hierarchical codes per clip as .npz under
data/encoded/orpheus_snac/<id>.npz (c0, c1, c2 int16 arrays + duration).
Flattening into Orpheus token ids happens at train time (orpheus_tokens.py)
so the cache stays model-revision-agnostic.

Resumable: existing .npz files are skipped.

Usage (GPU box):
  ./.venv/bin/python scripts/03_encode_snac.py            # all qc_passed clips
  ./.venv/bin/python scripts/03_encode_snac.py --include-unqcd   # if QC skipped
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import soundfile as sf
    from snac import SNAC
    from tqdm import tqdm
except ImportError as e:
    sys.exit(f"missing dep ({e}); pip install -r requirements-gpu.txt")

TARGET_SR = 24_000


def resolve_audio(data: Path, audio_path: str) -> Path | None:
    p = data / audio_path
    if p.exists():
        return p
    p = p.with_suffix(".flac")
    return p if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--include-unqcd", action="store_true",
                    help="also encode clips without a QC verdict (not recommended)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    con = sqlite3.connect(args.data / "synth_state.db")
    con.row_factory = sqlite3.Row
    where = "status='ok'" + ("" if args.include_unqcd else " AND qc_passed=1")
    rows = con.execute(f"SELECT id, audio_path FROM clips WHERE {where} ORDER BY id").fetchall()
    if not rows:
        sys.exit("no clips to encode — run QC first (or --include-unqcd)")

    out_dir = args.data / "encoded" / "orpheus_snac"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading SNAC 24kHz on {args.device} ...")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(args.device)

    done = skipped = failed = 0
    for row in tqdm(rows, unit="clip", desc="snac", dynamic_ncols=True):
        out = out_dir / f"{row['id']}.npz"
        if out.exists():
            skipped += 1
            continue
        wav_path = resolve_audio(args.data, row["audio_path"])
        if wav_path is None:
            failed += 1
            continue
        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != TARGET_SR:
            tqdm.write(f"  {row['id']}: unexpected sr {sr}, skipping")
            failed += 1
            continue
        x = torch.from_numpy(audio).reshape(1, 1, -1).to(args.device)
        with torch.inference_mode():
            codes = snac.encode(x)  # [c0 (T), c1 (2T), c2 (4T)]
        np.savez(out,
                 c0=codes[0].squeeze(0).cpu().numpy().astype(np.int16),
                 c1=codes[1].squeeze(0).cpu().numpy().astype(np.int16),
                 c2=codes[2].squeeze(0).cpu().numpy().astype(np.int16),
                 duration_s=np.float32(len(audio) / TARGET_SR))
        done += 1
    print(f"\nencoded {done:,}, skipped {skipped:,} (cached), failed {failed}")


if __name__ == "__main__":
    main()
