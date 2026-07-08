# HF model card draft — reso1-3b (run01, PRIVATE ONLY)

Copy this into the HF repo README.md when publishing. Keep the repo
PRIVATE while the weights derive from ElevenLabs data (ToS: no
distribution). Flip public only for a rights-clean model.

Publishing checklist:
- [ ] `hf auth login` (org: decibel-labs)
- [ ] `hf upload decibel-labs/reso1-3b checkpoints/reso1_orpheus3b_run01/merged . --private`
- [ ] separate repo `decibel-labs/reso1-3b-GGUF` for reso1-q4_k_m.gguf (--private)
- [ ] upload 6-8 best samples/ wavs into the repo, embed below
- [ ] test `from_pretrained` from a clean environment
- [ ] PUBLIC flip: only after voice-actor model + license review

---

```yaml
license: apache-2.0        # inherits Orpheus; REVIEW before public flip
language: [en]
base_model: canopylabs/orpheus-3b-0.1-ft
pipeline_tag: text-to-speech
tags: [tts, speech, orpheus, snac, streaming, expressive]
```

# Reso-1 3B (English, expressive)

Streaming, tag-controllable English TTS from Decibel Labs. LoRA fine-tune
of Orpheus 3B (Llama backbone + SNAC 24kHz codec).

## Samples

<!-- <audio controls src="samples/tag_laugh_0011.wav"></audio> etc. -->

## Usage

Prompt format: `reso: <text>` with inline tags
`<laugh> <chuckle> <sigh> <gasp> <whisper> <pause>`.
Sampling: temperature 0.6, top_p 0.95, repetition_penalty >= 1.1 (required).
Audio tokens decode via SNAC 24kHz; token layout: 7 tokens/frame from
AUDIO_BASE 128266 (see scripts/orpheus_tokens.py in the training repo).

## Training

~8.9 h single-voice expressive English (QC-filtered synthetic corpus:
WER + speech-rate + tag-event verification), 6-category coverage
(phonetic, prose, normalization, dialogue prosody, paralinguistic tags,
edge cases). LoRA r=64, 3 epochs, bf16, A100.

## Limitations

- Single voice, English only (Hindi planned via Veena base)
- <whisper>/<pause> are corpus-taught (not in Orpheus base tags) — less robust
- Not aligned/safety-filtered; disclose synthetic audio; no impersonation

## License / provenance

run01 trained on ElevenLabs-derived audio — PRIVATE, not distributable.
This card ships publicly only with the rights-clean (voice actor) model.
