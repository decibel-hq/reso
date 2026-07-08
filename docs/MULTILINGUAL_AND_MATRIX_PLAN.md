# Reso — Multilingual Base Selection & Voice×Language Matrix Plan

Research findings and locked decisions for taking Reso beyond English (Phase 2/3
of `TTS_RESEARCH_AND_ROADMAP.txt`). Written July 2026, after the reso1 English
run (Orpheus-3B, ElevenLabs-distilled) proved the pipeline end-to-end.

Source of truth for WHY the base was chosen and HOW the multi-voice /
multi-language product is structured. Supersedes the roadmap's earlier
assumption that Hindi = Veena base swap.

---

## TL;DR

- **Base model = `kenpath/svara-tts-v1`** (Svara-TTS). Apache-2.0, Orpheus
  lineage, token layout VERIFIED identical to our Orpheus-3B → drop-in to the
  existing pipeline. Natively speaks 18 Indic languages + Indian English.
- **Veena is NOT used.** It only speaks Hindi today (Marathi/Tamil/etc. are
  future roadmap), and Svara covers far more with the same architecture.
- **Product shape = one model, `voice × language` matrix.** "Reso" is the
  model; `adam / eve / maya / kabir` are the voices; languages are the second
  axis. One checkpoint, both selected at inference — exactly the ElevenLabs
  shape.
- **Shippability is decided by TRAINING DATA per cell, not by the model.**
  Svara (Apache-2.0) ships. Cells trained on ElevenLabs or Gemini audio are
  ToS-bound LEARNING-ONLY and must be dropped from the ship build. The reso1
  English adapter (adam×en, ElevenLabs-derived) is a pipeline proof, NOT a
  shippable artifact.

---

## 1. Base model decision — Svara-TTS v1

Requirement (unchanged from roadmap §3): autoregressive, streaming-capable,
LoRA-fine-tunable on one GPU, explicit tag control, open + permissive license,
and — new for Indic — native or warm-startable Indian-language ability in the
SAME architecture as our English stack (so the pipeline is reused, not rebuilt).

**`kenpath/svara-tts-v1` meets all of it:**

| Property | Value | Why it matters |
| --- | --- | --- |
| Backbone | `meta-llama/Llama-3.2-3B-Instruct` | Same as Orpheus-3B |
| Token approach | Orpheus-style discrete audio tokens | Our `03_encode_snac` + `orpheus_tokens.py` apply |
| `vocab_size` | **156,940** | **Exactly Orpheus-3B's** (128,256 Llama + 7×4,096 SNAC audio + control tokens). This is the decisive proof of a drop-in token layout. |
| bos / eos / pad | 128000 / 128009 / 128263 | Match Orpheus |
| License | **Apache-2.0** | Ship-legal base (not learning-only) |
| Languages | 18 Indic + Indian English (19) | Marathi ✅, Assamese ✅ native |
| Speakers | ~50 (balanced M/F) | Base already disentangles speaker↔language |
| Fine-tune | "LoRA-friendly"; 9 community finetunes | Fits our LoRA flow |

Full language list: Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Magahi,
Chhattisgarhi, Maithili, Assamese, Bodo, Dogri, Gujarati, Malayalam, Punjabi,
Tamil, Nepali, Sanskrit, Indian English.

**Because Svara natively speaks these languages, adding them is a VOICE
FINE-TUNE (~5-10 h/voice), not new-language teaching.** The base already
separates "who is speaking" from "what language" (that's how it serves 50
voices × 19 languages), which also helps OUR voices transfer across languages.

### Integration deltas (Svara vs our Orpheus-English setup)

1. **Verify token IDs at smoke-test.** vocab_size / bos / eos / pad all match
   Orpheus, so `orpheus_tokens.py` almost certainly works unchanged — but
   confirm the SOH/EOT/EOH/SOA/SOS/EOS + audio-base ids against Svara's
   reference before a full run (the 20-step smoke test is the check). If they
   differ, add `svara_tokens.py`. SNAC 24kHz is strongly implied ("Orpheus-style
   discrete tokens") but not explicitly stated on the card — confirm.
2. **Tag vocabulary differs.** Svara ships `<happy> <sad> <anger> <fear>` +
   `<clear>`, placed at the END of the utterance — NOT our `<laugh> <sigh>
   <chuckle>`. Either adopt Svara's set or teach our canonical tags onto it
   (install-from-scratch = more data). The tag PIPELINE is unchanged.
3. **Speaker/prompt format.** Svara uses `"Language (Gender)"` (e.g.
   `Marathi (Male)`). We adapt the `--voice-prefix` to named voices +
   language (see §3).
4. **Inference repo `github.com/Kenpath/svara-tts-inference` is "coming soon."**
   Exact prompt construction + detokenization not yet published — pin these
   from the repo/Colab before the first Svara run.

### Rejected / donor-only alternatives

- **Veena (maya-research/Veena)** — right architecture (Llama+SNAC) but Hindi +
  English only today; Marathi/Tamil/Telugu/Bengali are unscheduled roadmap.
  Superseded by Svara.
- **AI4Bharat/Indic-TTS** — MIT, covers 13 langs incl. Odia/Bodo/Manipuri/
  Rajasthani, BUT it is FastPitch + HiFi-GAN (non-autoregressive, non-LLM) —
  WRONG architecture to fine-tune as a base (same reason Kokoro/F5 were ruled
  out). Value = open-weight, ship-legal TEACHER / AUDIO DONOR, especially for
  languages Svara/ElevenLabs miss (notably **Odia**).
- **IndicF5 / flow-matching** — non-autoregressive, ruled out.

---

## 2. Teacher & language coverage (for DATA collection)

Two independent axes — do not conflate them:

- **Axis 1 — teacher/data availability:** where do (text, audio) pairs come from?
- **Axis 2 — base warm-start:** does the base already speak the language? (Svara
  answers this for 18 Indic langs.)

A great teacher solves Axis 1 only. Coverage of teachers for our target langs:

| Teacher | Marathi | Assamese | Odia | Tags? | License / ToS |
| --- | --- | --- | --- | --- | --- |
| **ElevenLabs v3** (70+ langs) | ✅ | ✅ | ❌ (STT only) | ✅ real positional `[laughs]` | **Learning-only** (ToS: no competing models) |
| **Gemini 2.5 Flash TTS** (24 GA + 60 preview) | ✅ | ✅ | — | ❌ style prompts only, no positional tags | **Learning-only** (ToS: no competing models; free tier also trains on your data — use paid) |
| **AI4Bharat/Indic-TTS** (13 langs) | ✅ | ✅ | ✅ | limited | **MIT — ship-legal** |
| **Voice actor** | any recorded | any recorded | any recorded | stage directions | **Owned (with AI-training contract) — ship-legal** |

Takeaways:
- For learning-the-pipeline, **prefer ElevenLabs over Gemini** because ElevenLabs
  has real positional tags (Gemini can't position-lock a laugh).
- **Odia** is the gap: not in Svara, not in ElevenLabs TTS, not in Gemini's Indic
  set → needs AI4Bharat/Indic-TTS teacher + heavier teaching, or a different base.
- Text donor for authentic spoken register: IndicVoices transcripts (CC-BY-4.0).
  Audio donor when multi-speaker Indic needed: IndicVoices-R / Rasa.

---

## 3. The product: Reso = one model, voice × language matrix

"Reso" is the model (one Svara-based checkpoint). Voices are the `speaker`
field; languages are the `lang` field — both already in the metadata schema
(roadmap §7). One LoRA over the whole grid; both axes selected at inference.

```
                    Reso  (one Svara-based checkpoint)
        ┌─────────┬────────┬────────┬────────┐
        │   en    │   hi   │   mr   │   as   │
 ┌──────┼─────────┼────────┼────────┼────────┤
 │ adam │ (proof) │        │        │        │   (male)
 │ eve  │         │        │        │        │   (female)
 │ maya │         │        │        │        │   (female)
 │ kabir│         │        │        │        │   (male)
 └──────┴─────────┴────────┴────────┴────────┘
   each cell = (voice, language) training clips
```

### Metadata row (matrix coordinates already supported)

```json
{"id": "mr_adam_0042",
 "text": "आज मीटिंग रद्द झाली <happy>",
 "speaker": "adam", "lang": "mr",
 "audio": "processed/mr/adam/clip_0042.wav",
 "source": "human_va", "prompt_version": "v1",
 "qc": {"wer": 0.06, "passed": true}}
```

### Prompt/training format (adapt `--voice-prefix` to `{voice} ({lang})`)

```
adam (mr): आज मीटिंग रद्द झाली <happy>
└─┬─┘ └┬┘  └──────────┬─────────┘ └──┬──┘
voice  lang        text (script)     tag
```

### Training / splits / heldout

- ONE LoRA over the union of all filled cells → the Reso checkpoint.
- Split by (voice, session, conversation) — no acoustic/context leak within a cell.
- Heldout is per-cell diagnostic (voice × lang × category), sampled not exhaustive.
  Readout example: "adam×mr tags fine, numbers regressed; kabir×hi Anglo accent."

### Filling cells: real data vs transfer

Each cell is GOOD only if it has real audio OR the base transfers acceptably:
- **Real audio (actor / per-lang teacher of that voice):** best, native, expensive.
- **Cross-lingual transfer (voice trained in other langs, zero-shots into L):**
  free, but often an ACCENT (e.g. English-only voice speaking Marathi sounds
  Anglo). Always put a diagnostic on transfer cells — measure, don't assume.
- Every real cell lifts the whole grid (more "separate voice from language"
  examples), so a truly bilingual anchor voice improves transfer everywhere.

Recommended first matrix (prove small, scale weak cells — same loop as English):
2 voices × 2 languages, real data in 3 of 4 cells, 1 cell left as a transfer test.

---

## 4. Ship vs. learn — the rule that governs the product

**The base ships; the data decides each cell.**

- **Svara base:** Apache-2.0 → ✅ ships (fine-tune, sell, redistribute).
- **Cell trained on `source ∈ {elevenlabs, gemini}`:** ToS LEARNING-ONLY →
  ❌ must be dropped from the ship build.
- **Cell trained on `source ∈ {human_va, self, open-licensed}`:** ✅ ships.
- **Transfer cells ship** if the voice's real data is ship-legal (base language
  ability is Apache-2.0 and free; only the VOICE data must be clean).

Mechanism (already designed): the ship build filters
`source NOT IN ('elevenlabs','gemini')` and re-runs the identical
encode→split→train. `04_make_splits` already filters on metadata; add one
`source` predicate. This is the roadmap's "discard ElevenLabs data before
shipping; re-run the pipeline with the voice actor" made concrete.

**Consequence:** the reso1 English model (adam×en, ElevenLabs-derived) is the
proof-of-pipeline, NOT the product. The shippable Reso is architecturally
identical — same Svara base, same scripts — trained on actor + open data.

### Contract requirement (do before hiring actors)

Voice-actor contracts for adam/eve/maya/kabir MUST explicitly grant the right
to TRAIN and SHIP AI/ML voice models on the recordings (work-for-hire / buyout
with AI-training rights). Standard VO contracts often do not — without this
clause, even the "shippable" cells are not shippable.

---

## 5. Code changes needed (none are large)

| Component | Change |
| --- | --- |
| `train_lora.py` | `--base kenpath/svara-tts-v1`; `--voice-prefix` → `{voice} ({lang})` |
| `orpheus_tokens.py` | Verify SOH/EOT/EOH/SOA/SOS/EOS + audio-base vs Svara; add `svara_tokens.py` only if they differ |
| `03_encode_snac.py` | None (same SNAC), pending SNAC confirmation |
| `05_filter_qc.py` | Add `--lang`; hardcoded `language="en"` → per-lang; use `large-v3` (multilingual) instead of `distil-large-v3` (English-only) for Indic; IndicWhisper/IndicConformer for low-resource |
| `02_normalize_text.py` + `normalizers/` | DOES NOT EXIST YET — build `normalizers/<lang>.py` (Devanagari canonical, number/currency normalization). Roadmap planned it; never written. |
| `04_make_splits.py` | Add `source` predicate for ship builds; split by (voice, session, conversation) |

---

## 6. Open items / to verify before first Svara run

1. Confirm SNAC 24kHz + exact control-token ids from the Svara inference repo
   (`github.com/Kenpath/svara-tts-inference`, "coming soon") or model weights.
2. Decide tag strategy: adopt Svara's `<happy>/<sad>/...` vs teach canonical
   `<laugh>/<sigh>`.
3. Odia base/teacher plan (Svara doesn't cover it).
4. Voice-actor contract with AI-training + ship rights.
5. Whisper Hindi/Marathi WER acceptable, or move to IndicWhisper/IndicConformer
   (`INDIC_CONFORMER_ASR_RUNBOOK.md` already in repo root).

---

## Sources

- Svara-TTS v1: https://huggingface.co/kenpath/svara-tts-v1
  (config.json: vocab_size 156940, Llama-3.2-3B, Apache-2.0)
- Svara-TTS launch: https://huggingface.co/blog/kenpath/svara-tts-open-multilingual-speech-for-india
- Orpheus multilingual (Canopy Labs): https://canopylabs.ai/releases/orpheus_can_speak_any_language
- Veena: https://huggingface.co/maya-research/Veena
- AI4Bharat/Indic-TTS: https://github.com/AI4Bharat/Indic-TTS  (MIT, FastPitch+HiFi-GAN, 13 langs)
- ElevenLabs models/languages: https://elevenlabs.io/docs/overview/models
- Gemini 2.5 TTS: https://blog.google/innovation-and-ai/technology/developers-tools/gemini-2-5-text-to-speech/
- Gemini API Additional Terms: https://ai.google.dev/gemini-api/terms
