#!/usr/bin/env python3
"""Local inference: type text, hear the fine-tuned Reso voice. Mac-friendly.

Runs the quantized GGUF through llama.cpp (Metal) and decodes audio tokens
with SNAC on CPU. Not streaming-optimized — this is the demo/spot-check
path; production streaming lives on GPU serving (vLLM etc.).

Setup (once):
  ./.venv/bin/pip install llama-cpp-python snac torch soundfile
Usage:
  ./.venv/bin/python scripts/say_local.py "Hey! <laugh> it actually works on my laptop."
  ./.venv/bin/python scripts/say_local.py --out hello.wav "Namaste, this is Reso one."
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import orpheus_tokens as ot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text")
    ap.add_argument("--gguf", type=Path,
                    default=Path("checkpoints/reso1_orpheus3b_run01/reso1-q4_k_m.gguf"))
    ap.add_argument("--voice-prefix", default="reso")
    ap.add_argument("--out", type=Path, default=None, help="wav path (default: temp + play)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    args = ap.parse_args()

    try:
        from llama_cpp import Llama
        import torch
        import numpy as np
        import soundfile as sf
        from snac import SNAC
    except ImportError as e:
        sys.exit(f"missing dep ({e}): ./.venv/bin/pip install llama-cpp-python snac torch soundfile")

    if not args.gguf.exists():
        sys.exit(f"gguf not found: {args.gguf}")

    print("loading gguf (Metal)...")
    llm = Llama(model_path=str(args.gguf), n_gpu_layers=-1, n_ctx=4096,
                verbose=False, logits_all=False)

    text = f"{args.voice_prefix}: {args.text}" if args.voice_prefix else args.text
    prompt = ([ot.SOH]
              + llm.tokenize(text.encode("utf-8"), add_bos=True, special=False)
              + [ot.EOT, ot.EOH, ot.SOA, ot.SOS])

    print("generating audio tokens...")
    audio_tokens = []
    for tok in llm.generate(prompt, temp=args.temperature, top_p=args.top_p,
                            repeat_penalty=args.rep_penalty):
        if tok == ot.EOS:
            break
        if tok >= ot.AUDIO_BASE:
            audio_tokens.append(tok)
        if len(audio_tokens) >= 7 * 84 * 30:  # hard cap ~30s
            break

    codes, valid = ot.unflatten_codes(audio_tokens)
    if not valid or len(codes[0]) < 3:
        sys.exit("degenerate generation — try again (sampling) or check the gguf")

    print(f"decoding {len(codes[0])} frames (~{len(codes[0]) * 0.085:.1f}s) with SNAC...")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    with torch.inference_mode():
        dev = [torch.tensor(c).unsqueeze(0) for c in codes]
        audio = snac.decode(dev).squeeze().numpy()

    out = args.out or Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(out, audio, 24_000)
    print(f"wrote {out}")
    if args.out is None and sys.platform == "darwin":
        subprocess.run(["afplay", str(out)])


if __name__ == "__main__":
    main()
