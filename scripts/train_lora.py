#!/usr/bin/env python3
"""LoRA fine-tune Orpheus 3B on the encoded decibel-tts dataset (Unsloth).

Reads splits/train.jsonl + data/encoded/orpheus_snac/*.npz, builds
Orpheus-format sequences (orpheus_tokens.py), trains a LoRA adapter,
saves to checkpoints/<run>/adapter/.

Usage (A100 80GB):
  ./.venv/bin/python scripts/train_lora.py --run reso1_orpheus3b_run01
  # smoke test first: --max-steps 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import orpheus_tokens as ot  # noqa: E402

try:
    from unsloth import FastLanguageModel  # must import before transformers
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments, Trainer
except ImportError as e:
    sys.exit(f"missing dep ({e}); pip install -r requirements-gpu.txt")


def build_dataset(split_path: Path, enc_dir: Path, tokenizer, voice_prefix: str,
                  max_len: int) -> Dataset:
    examples, dropped = [], 0
    for line in split_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        npz = enc_dir / f"{row['id']}.npz"
        if not npz.exists():
            dropped += 1
            continue
        z = np.load(npz)
        audio_ids = ot.flatten_codes(z["c0"], z["c1"], z["c2"])
        text = f"{voice_prefix}: {row['text']}" if voice_prefix else row["text"]
        text_ids = tokenizer(text, add_special_tokens=True).input_ids
        ids = ot.build_example(text_ids, audio_ids)
        if len(ids) > max_len:
            dropped += 1
            continue
        examples.append({"input_ids": ids, "labels": ids.copy()})
    print(f"dataset: {len(examples):,} examples ({dropped} dropped: missing npz / too long)")
    return Dataset.from_list(examples)


class Collator:
    def __call__(self, feats):
        import torch
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [ot.PAD] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * len(f["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="unsloth/orpheus-3b-0.1-ft")
    ap.add_argument("--voice-prefix", default="reso",
                    help="speaker name prefix baked into prompts; '' to disable"
                         " (must match at inference)")
    ap.add_argument("--splits", type=Path, default=Path("splits"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1, help="e.g. 20 for a smoke test")
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base, max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16, load_in_4bit=False)
    assert len(tokenizer) >= ot.MIN_VOCAB, \
        f"tokenizer vocab {len(tokenizer)} < {ot.MIN_VOCAB} — wrong base model?"

    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", use_gradient_checkpointing="unsloth",
        random_state=42)

    ds = build_dataset(args.splits / "train.jsonl",
                       args.data / "encoded" / "orpheus_snac",
                       tokenizer, args.voice_prefix, args.max_seq_len)

    out_dir = Path("checkpoints") / args.run
    trainer = Trainer(
        model=model, train_dataset=ds, data_collator=Collator(),
        args=TrainingArguments(
            output_dir=str(out_dir / "trainer"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, max_steps=args.max_steps,
            learning_rate=args.lr, lr_scheduler_type="cosine",
            warmup_ratio=0.03, bf16=True, logging_steps=10,
            save_strategy="epoch", save_total_limit=2,
            report_to="none", seed=42,
        ))
    trainer.train()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(f"\nadapter saved -> {adapter_dir}")
    print("next: generate_eval.py --run", args.run)


if __name__ == "__main__":
    main()
