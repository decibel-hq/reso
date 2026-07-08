# reso-tts

Training pipeline for **Reso** voice models — expressive, streaming, tag-controllable
TTS by Decibel Labs. Teacher distillation → LoRA fine-tunes of Llama+SNAC
architecture models (Orpheus now, Veena for Hindi later).

Full design rationale and roadmap: [`docs/ROADMAP.txt`](docs/ROADMAP.txt).
GPU session procedure: [`docs/A100_RUNBOOK.md`](docs/A100_RUNBOOK.md).

## Status

| Phase | State |
|---|---|
| prompts/v1 English corpus (16,213 utterances, 6 categories) | ✅ done |
| Teacher synthesis — 8,105 clips, ElevenLabs "adam", 24kHz | ✅ done |
| QC (Whisper WER + rate + tag events): 7,342 pass / 8.87 h | ✅ done |
| **run01**: LoRA r=64 on orpheus-3b, 3 epochs, loss 3.47 | ✅ done |
| Eval: 198 heldout + 10 fresh samples, 0 degenerate | ✅ done |
| Remaining 50% of corpus synthesis | next credit cycle |
| Voice actor (bilingual en+hi, distribution rights in contract) | planned |
| Hindi via Veena base | planned |

## ⚠️ Distribution status of run01

run01 is distilled from ElevenLabs output and is **ToS-bound: internal
learning only**. HF publishing practice happens on a **PRIVATE** repo
(see [`release/MODEL_CARD.md`](release/MODEL_CARD.md)) — the repo goes
public only when a rights-clean model (voice-actor data) replaces it.
Public-then-delete is not a remedy: downloads and mirrors are permanent.

## Layout

```
prompts/v1/       versioned text corpus (arctic, harvard, general,
                  normalization, prosody dialogues, tags, edge) + manifest
scripts/          numbered pipeline stages:
                  build_prompts_v1 → 00_synthesize_teacher → 05_filter_qc
                  → 03_encode_snac → 04_make_splits → train_lora
                  → generate_eval → say_local (Mac demo) → 90_pack_audio
data/             synth_state.db (source of truth), metadata/, qc/
                  (audio dirs gitignored — never commit audio)
splits/           train.jsonl / heldout.jsonl (diagnostic, per-category)
checkpoints/      run_config + eval samples tracked; weights gitignored
docs/             ROADMAP.txt (the why), A100_RUNBOOK.md (the how)
release/          HF model card draft + publishing checklist
```

## Quickstart (local demo, M-series Mac)

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install llama-cpp-python snac torch
./.venv/bin/python scripts/say_local.py "Hey! <laugh> it runs locally."
# needs checkpoints/reso1_orpheus3b_run01/reso1-q4_k_m.gguf (not in git)
```

Tags: `<laugh> <chuckle> <sigh> <gasp> <whisper> <pause>` inline in text.
Prompt convention: `reso: <text>` (baked into training). Generation needs
`repetition_penalty >= 1.1`.
