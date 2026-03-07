# qwen-tts

Voice cloning & fine-tuning for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) on Apple Silicon via MLX.

## Pipeline

```
source audio → split → prepare → train → generate → check
```

The CLI handles the full pipeline: ASR-based splitting & transcription, codec encoding, LoRA training, inference, and evaluation (local ASR + Gemini-as-judge). See [SKILL.md](SKILL.md) for the full cookbook.

## CLI commands

| Command    | Description                                        |
| ---------- | -------------------------------------------------- |
| `split`    | ASR-based split + transcribe source audio to clips |
| `prepare`  | Encode audio → codec tokens for training           |
| `train`    | Fine-tune a custom voice (LoRA SFT)                |
| `generate` | Generate speech from text                          |
| `check`    | Evaluate output (ASR intelligibility + Gemini)     |
| `speakers` | List available speakers in a voice model           |
| `voices`   | List trained voice models in a directory           |

### Split

```bash
# ASR-based split + transcribe (outputs clips + transcript.txt)
qwen-tts split raw_audio/source_24k.wav -o ./clips --max-dur 8
```

### Generate

```bash
# Basic TTS
qwen-tts generate -p "Hello world"

# Voice cloning (zero-shot, no training)
qwen-tts generate -p "Hello world" --voice ref.wav

# VoiceDesign (1.7B only)
qwen-tts generate -p "Hello world" --model voice-design --instruct "A warm, cheerful female voice"

# CustomVoice with emotion
qwen-tts generate -p "Hello world" --model custom-voice --speaker Chelsie --instruct "Very excited"

# Fine-tuned voice
qwen-tts generate -p "Hello world" --speaker lky --voice-model ./voices/lky/
```

### Prepare & Train

```bash
# Encode audio clips to codec tokens
qwen-tts prepare --data ./clips/ -o train.jsonl

# Train a custom voice
qwen-tts train --name lky --data train.jsonl -o ./voices/lky/ \
    --epochs 3 --lr 2e-5 --grad-accum 4
```

### Check

```bash
# ASR-only (local, no API key)
qwen-tts check -g ./eval_clips --asr-only \
    --expected-text "gen_00.wav=Hello world."

# Full eval (ASR + Gemini speaker similarity)
qwen-tts check -g ./eval_clips -r ./clips -S "Speaker Name" \
    --max-clips 3 --pairs 3 \
    --expected-text "gen_00.wav=Hello world."
```

## Model variants

| Model          | Description                                               |
| -------------- | --------------------------------------------------------- |
| `base`         | Zero-shot voice cloning via reference audio               |
| `voice-design` | Describe any voice style via natural language (1.7B only) |
| `custom-voice` | Pre-built speakers with optional emotion/style            |

Sizes: 1.7B (default) or 0.6B (smaller/faster).

## Requirements

- Apple Silicon Mac (≥16GB unified memory)
- Python 3.10–3.12
- `mlx`, `mlx-lm`, `mlx-audio`, `librosa`
