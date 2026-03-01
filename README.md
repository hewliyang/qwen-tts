# qwen-tts

Voice cloning & fine-tuning for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) on Apple Silicon via MLX.

## Pipeline

```
prepare_data.py → train.py → cli.py generate
```

### 1. Prepare data

Encodes WAV clips through the speech tokenizer to produce 16-codebook audio codes.

```bash
python prepare_data.py \
    --input-dir ./clips/ \
    --ref-audio ref.wav \
    --output train.jsonl
```

Input: a directory of WAVs + `transcript.txt` (`001.wav|Hello world` per line), or a pre-built JSONL.  
Output: JSONL with `audio_codes` (16 codebook indices per frame) added to each record.

### 2. Train (`train.py`)

LoRA SFT on the **talker transformer** to learn a target voice from a handful of clips.

**What gets trained:**

- LoRA adapters on `q_proj` / `v_proj` attention layers (~2–4M params vs ~755M frozen)
- `codec_head` and `text_projection` are fully unfrozen

**Forward pass** (mirrors the official PyTorch `sft_12hz.py`):

1. Extract speaker embedding from reference mel via frozen `speaker_encoder`
2. Build input embeddings: text embed + codec embed + sub-codebook embeds (codebooks 1–15), with speaker embedding injected at position 6
3. Forward talker → CE loss on codec-0 prediction (next-token)
4. Forward sub-talker (code predictor) → per-group CE loss on codebooks 1–15
5. **Total loss = talker_loss + 0.3 × sub_talker_loss**

**Checkpoint saving:** LoRA weights are mathematically merged back (`W + scale·B·A`) and the target speaker embedding is burned into `codec_embedding.weight[3000]`, producing a self-contained custom-voice model.

```bash
python train.py \
    --train-jsonl train.jsonl \
    --speaker-name lky \
    --output ./voices/lky/ \
    --epochs 3 \
    --lr 2e-5 \
    --lora-rank 8 \
    --grad-accum 4
```

## CLI usage

Models are auto-downloaded from HuggingFace on first use.

| Model          | Description                                               |
| -------------- | --------------------------------------------------------- |
| `base`         | Zero-shot voice cloning via reference audio               |
| `voice-design` | Describe any voice style via natural language (1.7B only) |
| `custom-voice` | Pre-built speakers with optional emotion/style            |

Sizes: 1.7B (default) or 0.6B (smaller/faster).

```bash
# Basic TTS
qwen-tts generate -p "Hello world"

# Smaller 0.6B model
qwen-tts generate -p "Hello world" --size 0.6B

# Voice cloning (Base model)
qwen-tts generate -p "Hello world" --voice ref.wav

# Style prompting (VoiceDesign, 1.7B only)
qwen-tts generate -p "Hello world" \
    --model voice-design \
    --instruct "A warm, cheerful female voice"

# Named speaker + emotion (CustomVoice)
qwen-tts generate -p "Hello world" \
    --model custom-voice --speaker Chelsie \
    --instruct "Very excited"

# List available speakers
qwen-tts speakers

# Fine-tuned voice
qwen-tts generate -p "Hello world" \
    --speaker lky --voice-model ./voices/lky/
```

## Key files

| File              | Purpose                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| `train.py`        | LoRA SFT loop (loss, LoRA injection, checkpoint merge)                  |
| `dataset.py`      | `TTSDataset` + collation — tokenizes text, builds masks, pads batches   |
| `prepare_data.py` | Encodes WAVs → 16-codebook codes via speech tokenizer                   |
| `cli.py`          | Inference CLI wrapping `mlx-audio` (base / voice-design / custom-voice) |

## Requirements

- Apple Silicon Mac (>16GB unified memory)
- Python 3.10+
- `mlx`, `mlx-lm`, `mlx-audio`, `librosa`
