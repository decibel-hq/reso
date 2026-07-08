#!/usr/bin/env python3
"""Pack synthesized audio for upload to a GPU box.

Converts data/raw/<lang>/<voice>/*.wav to a FLAC mirror (lossless, ~50%
smaller — decodes to bit-identical PCM) and tars it together with
synth_state.db. QC/encoding scripts fall back to .flac automatically when
the .wav is absent, so the remote side needs no conversion back.

Usage:
  ./.venv/bin/python scripts/90_pack_audio.py --voice-name adam
  # -> data/upload_adam.tar  (scp/rsync this to the GPU box)
"""

import argparse
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def to_flac(pair):
    src, dst = pair
    if dst.exists():
        return 0
    audio, rate = sf.read(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, audio, rate, format="FLAC")
    return src.stat().st_size - dst.stat().st_size


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voice-name", default="adam")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    src_dir = args.data / "raw" / args.lang / args.voice_name
    flac_dir = args.data / "raw_flac" / args.lang / args.voice_name
    wavs = sorted(src_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no wavs under {src_dir}")
    pairs = [(w, flac_dir / (w.stem + ".flac")) for w in wavs]

    saved = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for s in tqdm(pool.map(to_flac, pairs, chunksize=64), total=len(pairs),
                      unit="clip", desc="flac", dynamic_ncols=True):
            saved += s

    tar_path = args.data / f"upload_{args.voice_name}.tar"
    with tarfile.open(tar_path, "w") as tar:  # flac is already compressed
        # land inside data/raw/... on the remote so audio_path in the DB
        # resolves (with the .flac fallback) without any remapping
        tar.add(flac_dir, arcname=f"raw/{args.lang}/{args.voice_name}")
        tar.add(args.data / "synth_state.db", arcname="synth_state.db")

    wav_gb = sum(w.stat().st_size for w in wavs) / 1e9
    tar_gb = tar_path.stat().st_size / 1e9
    print(f"\n{len(wavs):,} clips: {wav_gb:.2f} GB wav -> {tar_gb:.2f} GB tar"
          f" ({100 * (1 - tar_gb / wav_gb):.0f}% smaller)")
    print(f"upload artifact: {tar_path}")
    print(f"remote unpack:   tar -xf upload_{args.voice_name}.tar -C decibel-tts/data/")


if __name__ == "__main__":
    main()
