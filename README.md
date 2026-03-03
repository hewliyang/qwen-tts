# qwen-tts

Voice cloning & fine-tuning for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) on Apple Silicon via MLX.

## Pipeline

```
source audio → split → transcribe → prepare → train → generate → verify
```

The CLI handles the hard parts (codec encoding, training, inference). Everything else — splitting audio, transcribing, validating output — is done inline with `librosa` and `mlx-audio`'s STT models. See [SKILL.md](SKILL.md) for the full cookbook.

## CLI commands

| Command    | Description                              |
| ---------- | ---------------------------------------- |
| `generate` | Generate speech from text                |
| `prepare`  | Encode audio → codec tokens for training |
| `train`    | Fine-tune a custom voice (LoRA SFT)      |
| `speakers` | List available speakers                  |
| `voices`   | List trained voice models in a directory |

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
