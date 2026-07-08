# A100 session runbook — QC (and later: encode + train)

One rented GPU box, one upload, everything GPU-shaped in one sitting.
Kill the instance the moment you're done — nothing lives there.

## 0. Local, before renting

```bash
# pack audio (FLAC mirror + synth_state.db -> data/upload_adam.tar, ~1 GB)
./.venv/bin/python scripts/90_pack_audio.py --voice-name adam
```

## 1. On the box (Ubuntu + CUDA image, e.g. Runpod pytorch template)

```bash
tmux new -s work                     # survive SSH drops; reattach: tmux a -t work
git clone https://github.com/decibel-hq/labs.git && cd labs/decibel-tts
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

## 2. Upload the audio (from the Mac, in another terminal)

```bash
rsync -avP "decibel-tts/data/upload_adam.tar" root@<BOX_IP>:~/labs/decibel-tts/data/
# on the box:
cd ~/labs/decibel-tts/data && tar -xf upload_adam.tar && cd ..
# lands: data/raw/en/adam/*.flac  +  data/synth_state.db
# scripts fall back to .flac automatically when the .wav is absent
```

## 3. QC on GPU (~15 min for ~10 h of audio)

```bash
./.venv/bin/python scripts/05_filter_qc.py run \
    --device cuda --compute-type float16 --qc-threads 8
./.venv/bin/python scripts/05_filter_qc.py report
./.venv/bin/python scripts/00_synthesize_teacher.py export --voice-name adam
```

Read the report before proceeding:
- phonetic/general pass rate should be ≳95%
- normalization: judge by median WER, not pass count (ASR formats numbers
  differently even when speech is perfect — retune threshold if needed)
- tags: check the event-evidence table — this is the verdict on whether
  the teacher actually laughed/sighed where told

## 4. Bring the verdicts home (small files — always do this before killing)

```bash
# from the Mac:
rsync -avP root@<BOX_IP>:~/labs/decibel-tts/data/synth_state.db  decibel-tts/data/
rsync -avP root@<BOX_IP>:~/labs/decibel-tts/data/qc/             decibel-tts/data/qc/
rsync -avP root@<BOX_IP>:~/labs/decibel-tts/data/metadata/       decibel-tts/data/metadata/
```

The DB now carries the QC verdicts — local and remote stay interchangeable.

## 5. Encode + split + train + eval (same session, same upload)

```bash
./.venv/bin/pip install -r requirements-gpu.txt   # torch/unsloth/snac — do during upload

./.venv/bin/python scripts/03_encode_snac.py                       # ~15 min
./.venv/bin/python scripts/04_make_splits.py                       # seconds

./.venv/bin/python scripts/train_lora.py --run reso1_orpheus3b_run01 --max-steps 20   # SMOKE TEST first
./.venv/bin/python scripts/train_lora.py --run reso1_orpheus3b_run01                  # real run, ~1-2 h

./.venv/bin/python scripts/generate_eval.py --run reso1_orpheus3b_run01               # heldout wavs
```

If the smoke test crashes, the usual suspect is the Orpheus token layout —
all ids live in scripts/orpheus_tokens.py, verify against the base model's
fine-tune reference before touching anything else.

## 6. Bring the results home, then KILL THE INSTANCE

```bash
# from the Mac:
rsync -avP root@<BOX_IP>:~/labs/decibel-tts/checkpoints/reso1_orpheus3b_run01/adapter/  decibel-tts/checkpoints/reso1_orpheus3b_run01/adapter/
rsync -avP root@<BOX_IP>:~/labs/decibel-tts/checkpoints/reso1_orpheus3b_run01/samples/  decibel-tts/checkpoints/reso1_orpheus3b_run01/samples/
rsync -avP root@<BOX_IP>:~/labs/decibel-tts/splits/ decibel-tts/splits/
```

Adapter ≈ a few hundred MB, samples ≈ tens of MB — light downloads.
Listen to samples/ PER CATEGORY: tags and normalization are the verdicts
that decide what the next iteration fixes.

## Cost sanity

QC ~15 min + encode ~15 min + train ~1–2 h + eval ~15 min ≈ 2–3 h ≈ ₹400–550.
The upload (~1 GB) is the slow part on home bandwidth — start it first,
do the git/venv setup while it transfers.
