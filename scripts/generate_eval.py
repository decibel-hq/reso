#!/usr/bin/env python3
"""Generate diagnostic heldout samples from a trained adapter.

For every clip in splits/heldout.jsonl: prompt the fine-tuned model with the
text, decode generated audio tokens through SNAC, write WAV to
checkpoints/<run>/samples/<category>/<id>.wav. Listen per-category:
that's the whole point of the diagnostic heldout.

Usage:
  ./.venv/bin/python scripts/generate_eval.py --run reso1_orpheus3b_run01
  ./.venv/bin/python scripts/generate_eval.py --run ... --limit 40   # quick pass
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import orpheus_tokens as ot  # noqa: E402

try:
    from unsloth import FastLanguageModel
    import torch
    import soundfile as sf
    from snac import SNAC
    from tqdm import tqdm
except ImportError as e:
    sys.exit(f"missing dep ({e}); pip install -r requirements-gpu.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--limit", type=int, default=0, help="cap clips (0 = all)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-penalty", type=float, default=1.1,
                    help="Orpheus needs >=1.1 or audio loops")
    args = ap.parse_args()

    run_dir = Path("checkpoints") / args.run
    cfg = json.loads((run_dir / "run_config.json").read_text())
    voice_prefix = cfg.get("voice_prefix", "reso")

    model, tokenizer = FastLanguageModel.from_pretrained(
        str(run_dir / "adapter"), max_seq_length=cfg.get("max_seq_len", 4096),
        dtype=torch.bfloat16, load_in_4bit=False)
    FastLanguageModel.for_inference(model)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")

    rows = [json.loads(l) for l in
            (args.splits / "heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[: args.limit]

    ok = bad = 0
    for row in tqdm(rows, unit="clip", desc="eval", dynamic_ncols=True):
        text = f"{voice_prefix}: {row['text']}" if voice_prefix else row["text"]
        text_ids = tokenizer(text, add_special_tokens=True).input_ids
        prompt = [ot.SOH] + text_ids + [ot.EOT, ot.EOH, ot.SOA, ot.SOS]
        ids = torch.tensor([prompt], device="cuda")
        with torch.inference_mode():
            gen = model.generate(
                input_ids=ids, max_new_tokens=3500,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.rep_penalty,
                eos_token_id=ot.EOS, pad_token_id=ot.PAD)
        audio_tokens = [t for t in gen[0][len(prompt):].tolist()
                        if t >= ot.AUDIO_BASE]
        (codes, valid) = ot.unflatten_codes(audio_tokens)
        if not valid or len(codes[0]) < 3:
            bad += 1
            continue
        dev_codes = [torch.tensor(c, device="cuda").unsqueeze(0) for c in codes]
        with torch.inference_mode():
            audio = snac.decode(dev_codes).squeeze().cpu().numpy()
        out = run_dir / "samples" / row["category"] / f"{row['id']}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, audio, 24_000)
        ok += 1
    print(f"\nsamples: {ok} written, {bad} degenerate -> {run_dir / 'samples'}")
    print("rsync the samples dir home and LISTEN per category.")


if __name__ == "__main__":
    main()
