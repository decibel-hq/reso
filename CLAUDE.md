# reso-tts — context for Claude sessions

Reso voice models by teacher distillation → LoRA fine-tunes of Llama+SNAC
models (Orpheus for English, Veena planned for Hindi). This is a
standalone repo (moved out of the decibel/labs monorepo, July 2026).

Read first: README.md (status table), docs/ROADMAP.txt (design rationale),
docs/A100_RUNBOOK.md (GPU session procedure).

## Current state (2026-07-08)

- run01 complete: LoRA r=64 on unsloth/orpheus-3b-0.1-ft, 3 epochs,
  loss ~3.47. Adapter + Q4 GGUF live locally (gitignored); 208 eval
  samples tracked in checkpoints/reso1_orpheus3b_run01/samples/.
- data/synth_state.db = source of truth for all 8,105 teacher clips
  (synthesis + QC verdicts). Raw audio is local-only, never in git.
- Remaining 50% of prompts/v1 awaits next month's ElevenLabs credits
  (00_synthesize_teacher.py resumes from the DB).

## Rules

- data/raw*, encoded/, adapter weights, *.gguf: NEVER commit. HF is the
  distribution channel for weights.
- run01 derives from ElevenLabs audio → ToS-bound: HF repos stay PRIVATE
  until a rights-clean (voice actor) model replaces it. Public-then-delete
  is not acceptable.
- Never re-run teacher synthesis casually — it spends ElevenLabs credits.
- Prompt format "reso: <text>"; tags <laugh> <chuckle> <sigh> <gasp>
  <whisper> <pause>; generation needs repetition_penalty >= 1.1.
- Orpheus token layout lives ONLY in scripts/orpheus_tokens.py.
- Voice actor contract (future) must include synthetic-voice
  distribution rights, and record English + Hindi in the same sessions.
