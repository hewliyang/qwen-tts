---
name: qwen-tts
description: "Clone and fine-tune voices using Qwen3-TTS on Apple Silicon (MLX). Full pipeline: source audio → split → transcribe → prepare codec tokens → LoRA training → inference → ASR verification. Use when the user wants to clone a voice, train a custom TTS voice, or generate speech."
compatibility: "Apple Silicon Mac with ≥16GB unified memory. Python 3.10–3.12. MLX backend."
---

# Qwen3-TTS Voice Cloning & Fine-Tuning

Train custom TTS voices on Apple Silicon via MLX.

## Setup

```bash
cd /Users/m1a1/Developer/qwen-tts
source .venv/bin/activate
```

## CLI Commands

These are the hard operations that require model loading, codec encoding, or training loops:

| Command                                                                       | Purpose                     |
| ----------------------------------------------------------------------------- | --------------------------- |
| `python -m qwen_tts generate -p "text"`                                       | Generate speech from text   |
| `python -m qwen_tts generate -p "text" --voice ref.wav`                       | Zero-shot voice cloning     |
| `python -m qwen_tts generate -p "text" --speaker name --voice-model ./voice/` | Generate with trained voice |
| `python -m qwen_tts prepare --data ./clips/ -o train.jsonl`                   | Encode audio → codec tokens |
| `python -m qwen_tts train --name speaker --data train.jsonl -o ./voice/`      | LoRA SFT training           |
| `python -m qwen_tts speakers`                                                 | List available speakers     |
| `python -m qwen_tts voices ./voices/`                                         | List trained voice models   |

---

## Full Pipeline

### Step 1: Source Audio

Collect audio of the target speaker. Ideal: 5–30 clips, 3–15 seconds each, clean speech, minimal background noise.

If starting from a long recording, split it into clips. Use `librosa.effects.split` for silence-based segmentation:

```python
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

audio, sr = librosa.load("source.wav", sr=24000, mono=True)
intervals = librosa.effects.split(audio, top_db=30)

output_dir = Path("./clips")
output_dir.mkdir(exist_ok=True)

# Merge short segments, split long ones
min_dur, max_dur = 2.0, 15.0
clips = []
cur_start, cur_end = intervals[0]

for start, end in intervals[1:]:
    if (end - cur_start) / sr <= max_dur:
        cur_end = end  # merge
    else:
        if (cur_end - cur_start) / sr >= min_dur:
            clips.append((cur_start, cur_end))
        cur_start, cur_end = start, end

if (cur_end - cur_start) / sr >= min_dur:
    clips.append((cur_start, cur_end))

for i, (start, end) in enumerate(clips):
    sf.write(str(output_dir / f"clip_{i:03d}.wav"), audio[start:end], sr)
    print(f"clip_{i:03d}.wav: {(end - start) / sr:.1f}s")
```

Adjust `top_db` (silence threshold), `min_dur`, `max_dur` based on the source material. For noisy audio, increase `top_db`. For fast speakers, decrease `min_dur`.

### Step 2: Transcribe

Generate transcripts using ASR via mlx-audio. The agent needs accurate transcripts — ASR provides a starting point to review and correct.

```python
from mlx_audio.stt import load as load_stt
from pathlib import Path

model = load_stt("mlx-community/whisper-large-v3-turbo")

clips_dir = Path("./clips")
audio_files = sorted(clips_dir.glob("*.wav"))

lines = []
for af in audio_files:
    result = model.generate(str(af), verbose=False)
    text = result.text.strip()
    lines.append(f"{af.name}|{text}")
    print(f"{af.name}: {text}")

(clips_dir / "transcript.txt").write_text("\n".join(lines) + "\n")
```

**STT model options:**

- `mlx-community/whisper-large-v3-turbo` — fast, good quality (default choice)
- `mlx-community/whisper-large-v3-mlx` — best quality, slower
- `mlx-community/whisper-small` — fastest, lower quality
- `mlx-community/parakeet-tdt-0.6b-v2` — alternative ASR engine
- `mlx-community/parakeet-tdt-1.1b-v2` — larger parakeet

**Review the transcript.** Fix ASR errors before proceeding — transcript accuracy directly affects training quality. The format is `filename|text` per line.

### Step 3: Prepare Codec Tokens

This is a CLI command — it loads the speech tokenizer encoder and encodes each audio clip into 16-codebook codes:

```bash
python -m qwen_tts prepare --data ./clips/ -o train.jsonl
```

Options:

- `--ref-audio ref.wav` — explicit reference audio (default: first clip)
- `--model-id <repo>` — base model (default: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`)

Output: JSONL with `audio_codes` added to each record.

### Step 4: Train

```bash
python -m qwen_tts train \
    --name speaker_name \
    --data train.jsonl \
    --output ./voices/speaker_name/ \
    --epochs 3 \
    --lr 2e-5 \
    --grad-accum 4
```

**Parameters:**

- `--epochs` — 2–5 typical (default: 3)
- `--lr` — 1e-5 to 5e-5 (default: 2e-5)
- `--grad-accum` — effective batch size multiplier (default: 4)
- `--lora-rank` — LoRA rank, higher = more capacity (default: 8)
- `--batch-size` — keep at 1 to avoid OOM
- `--save-every-epoch` — save intermediate checkpoints

**What gets trained:** LoRA adapters on `q_proj`/`v_proj` attention layers (~2–4M params vs ~755M frozen), plus `codec_head` and `text_projection` fully unfrozen.

**Output:** Self-contained voice model directory with merged LoRA weights and burned-in speaker embedding.

### Step 5: Generate

```bash
python -m qwen_tts generate \
    -p "Hello, this is my cloned voice." \
    --speaker speaker_name \
    --voice-model ./voices/speaker_name/ \
    -o output.wav
```

### Step 6: Verify Quality (Round-Trip ASR)

Agents can't hear audio. To validate TTS output, generate speech then transcribe it back and compare:

```python
from difflib import SequenceMatcher
from mlx_audio.stt import load as load_stt

input_text = "The quick brown fox jumps over the lazy dog."
wav_path = "output.wav"  # generated in step 5

stt = load_stt("mlx-community/whisper-large-v3-turbo")
result = stt.generate(wav_path, verbose=False)
transcribed = result.text.strip()

similarity = SequenceMatcher(None, input_text.lower(), transcribed.lower()).ratio()
print(f"Input:       {input_text}")
print(f"Transcribed: {transcribed}")
print(f"Similarity:  {similarity:.1%}")
```

Use judgment on what similarity threshold is acceptable — it depends on the text complexity, language, and use case. Common sense: if the transcription is gibberish, the voice needs more training data or different hyperparameters.

**If quality is poor, consider:**

- More training data (more clips)
- More epochs (5–8) with lower LR (1e-5)
- Checking transcript accuracy
- Cleaner source audio

---

## Zero-Shot Voice Cloning (No Training)

For quick voice cloning without fine-tuning:

```bash
python -m qwen_tts generate -p "Text to speak" --voice reference.wav --voice-text "Transcript of reference"
```

---

## Model Variants

| Model       | Flag                    | Description                                          |
| ----------- | ----------------------- | ---------------------------------------------------- |
| Base        | `--model base`          | Zero-shot voice cloning via reference audio          |
| VoiceDesign | `--model voice-design`  | Describe voice style in natural language (1.7B only) |
| CustomVoice | `--model custom-voice`  | Pre-built speakers with emotion/style control        |
| Fine-tuned  | `--voice-model ./path/` | Your trained voice models                            |

Sizes: `--size 1.7B` (default) or `--size 0.6B` (faster, less quality).

---

## Data Validation

Before training, sanity-check the data. These are things to verify — use whatever approach makes sense for the situation:

**Audio directory:**

- All files in transcript.txt exist as audio files
- Clip durations are reasonable (2–15s, not too short or too long)
- Total audio is 1–10 minutes
- Audio is clean speech (not music, not silence)

**JSONL (after prepare):**

- All records have `audio_codes`
- Referenced audio and ref_audio paths exist
- `audio_codes` have 16 codebooks per frame

```python
import json
from pathlib import Path

with open("train.jsonl") as f:
    records = [json.loads(line) for line in f if line.strip()]

for r in records:
    assert r.get("audio_codes"), f"Missing codes: {r['audio']}"
    assert Path(r["audio"]).exists(), f"Missing audio: {r['audio']}"
    assert Path(r["ref_audio"]).exists(), f"Missing ref: {r['ref_audio']}"
    assert len(r["audio_codes"][0]) == 16, f"Bad codebooks: {r['audio']}"

print(f"✅ {len(records)} records validated")
```

---

## Troubleshooting

**OOM during training:** Keep `--batch-size 1`, reduce `--lora-rank` to 4, use shorter clips, or use 0.6B base model.

**Poor voice quality:** Check transcript accuracy, try more epochs (5–8) with lower LR (1e-5), add more diverse clips, ensure clean audio.

**ASR errors:** Use `whisper-large-v3-mlx` for best accuracy. For non-English, whisper-large-v3 works best. Always manually review.

**Slow model download:** Models are cached after first download (~3–6GB). First run is slow.
