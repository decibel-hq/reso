"""Orpheus token layout — single source of truth for train + eval scripts.

Constants follow the canopylabs/orpheus-tts fine-tuning code (Llama-3.2
backbone with extended vocab). generate_eval.py asserts the tokenizer is
large enough at runtime; if an Orpheus revision ever shifts these ids,
fix them HERE only.

Sequence format per training example:
  [SOH] <text tokens (with BOS)> [EOT] [EOH] [SOA] [SOS] <audio tokens> [EOS] [EOA]

Audio tokens: SNAC's 3 hierarchical codebooks flattened to 7 tokens per
~85ms frame, each slot offset by 4096 so ids never collide.
"""

import numpy as np

SOH = 128259   # start of human turn
EOT = 128009   # llama end-of-text
EOH = 128260   # end of human turn
SOA = 128261   # start of ai turn
SOS = 128257   # start of speech
EOS = 128258   # end of speech
EOA = 128262   # end of ai turn
PAD = 128263
AUDIO_BASE = 128266
CODEBOOK = 4096
MIN_VOCAB = AUDIO_BASE + 7 * CODEBOOK  # tokenizer must be at least this big


def flatten_codes(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> list[int]:
    """SNAC hierarchical codes -> flat Orpheus audio token ids (7 per frame)."""
    n = len(c0)
    assert len(c1) == 2 * n and len(c2) == 4 * n, "SNAC code shape mismatch"
    out = []
    for i in range(n):
        out += [
            int(c0[i]) + AUDIO_BASE,
            int(c1[2 * i]) + AUDIO_BASE + CODEBOOK,
            int(c2[4 * i]) + AUDIO_BASE + 2 * CODEBOOK,
            int(c2[4 * i + 1]) + AUDIO_BASE + 3 * CODEBOOK,
            int(c1[2 * i + 1]) + AUDIO_BASE + 4 * CODEBOOK,
            int(c2[4 * i + 2]) + AUDIO_BASE + 5 * CODEBOOK,
            int(c2[4 * i + 3]) + AUDIO_BASE + 6 * CODEBOOK,
        ]
    return out


def unflatten_codes(tokens: list[int]):
    """Flat audio token ids -> SNAC code triple (for decoding eval samples)."""
    frames = len(tokens) // 7
    tokens = tokens[: frames * 7]
    c0, c1, c2 = [], [], []
    for i in range(frames):
        f = tokens[i * 7: (i + 1) * 7]
        c0.append(f[0] - AUDIO_BASE)
        c1 += [f[1] - AUDIO_BASE - CODEBOOK, f[4] - AUDIO_BASE - 4 * CODEBOOK]
        c2 += [f[2] - AUDIO_BASE - 2 * CODEBOOK, f[3] - AUDIO_BASE - 3 * CODEBOOK,
               f[5] - AUDIO_BASE - 5 * CODEBOOK, f[6] - AUDIO_BASE - 6 * CODEBOOK]
    ok = all(0 <= c < CODEBOOK for c in c0 + c1 + c2)
    return (np.array(c0), np.array(c1), np.array(c2)), ok


def build_example(text_ids: list[int], audio_ids: list[int]) -> list[int]:
    return [SOH] + list(text_ids) + [EOT, EOH, SOA, SOS] + list(audio_ids) + [EOS, EOA]
