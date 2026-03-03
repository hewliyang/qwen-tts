import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from mlx_audio.tts.models.qwen3_tts import Model as Qwen3TTSModel

# model_key -> { size -> HF repo }
_MODEL_REPOS = {
    "base": {
        "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    },
    "voice-design": {
        "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    },
    "custom-voice": {
        "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
    },
}

SIZES = ["1.7B", "0.6B"]
DEFAULT_SIZE = "1.7B"


def resolve_model_id(model_key: str, size: str) -> str:
    """Resolve a (model_key, size) pair to a HuggingFace repo ID."""
    repos = _MODEL_REPOS.get(model_key)
    if repos is None:
        raise ValueError(f"Unknown model: {model_key}")
    repo = repos.get(size)
    if repo is None:
        available = ", ".join(repos.keys())
        raise ValueError(
            f"Model '{model_key}' is not available in size {size}. "
            f"Available sizes: {available}"
        )
    return repo


def load_tts_model(model_id: str) -> Qwen3TTSModel:
    from typing import cast

    from mlx_audio.tts.utils import load_model

    print(f"Loading TTS model {model_id}...", end=" ", flush=True)
    t0 = time.time()
    model = cast(Qwen3TTSModel, load_model(model_id))
    print(f"done ({time.time() - t0:.1f}s)")
    return model


# ──────────────────────────────────────────────
#  generate
# ──────────────────────────────────────────────
def cmd_generate(args: argparse.Namespace) -> None:
    from mlx_audio.tts.generate import generate_audio

    if args.seed is not None:
        import mlx.core as mx

        np.random.seed(args.seed)
        mx.random.seed(args.seed)

    if args.voice_model:
        model = load_tts_model(args.voice_model)
    else:
        model_key: str = args.model
        size: str = args.size
        model_id = resolve_model_id(model_key, size)
        model = load_tts_model(model_id)

    prompt: str = args.prompt
    output_prefix: str = str(args.output).removesuffix(".wav")
    play: bool = args.play

    gen_kwargs: dict[str, object] = dict(
        text=prompt,
        model=model,
        file_prefix=output_prefix,
        audio_format="wav",
        join_audio=True,
        play=play,
        verbose=True,
    )

    if args.voice:
        gen_kwargs["ref_audio"] = str(args.voice)
        gen_kwargs["ref_text"] = str(args.voice_text or "")

    if args.instruct:
        gen_kwargs["instruct"] = str(args.instruct)

    if args.speaker:
        gen_kwargs["voice"] = str(args.speaker)

    generate_audio(**gen_kwargs)  # type: ignore[arg-type]


# ──────────────────────────────────────────────
#  speakers
# ──────────────────────────────────────────────
def cmd_speakers(args: argparse.Namespace) -> None:
    if args.voice_model:
        model = load_tts_model(args.voice_model)
    else:
        size = getattr(args, "size", DEFAULT_SIZE)
        model_id = resolve_model_id("custom-voice", size)
        model = load_tts_model(model_id)

    speakers = model.supported_speakers if hasattr(model, "supported_speakers") else []
    if speakers:
        print(f"\nAvailable speakers ({len(speakers)}):")
        for s in sorted(speakers):
            print(f"  {s}")
    else:
        print("No speaker list found on model.")


# ──────────────────────────────────────────────
#  prepare
# ──────────────────────────────────────────────
def cmd_prepare(args: argparse.Namespace) -> None:
    """Prepare training data: encode audio clips to codec tokens."""
    import dataclasses

    import mlx.core as mx

    from .dataset import TrainingRecord
    from .prepare_data import (
        build_records_from_directory,
        encode_audio,
        load_audio_24k,
        load_model_for_encoding,
    )

    data_path = Path(args.data)

    # Build or load records
    if data_path.is_dir():
        transcript = args.transcript
        if transcript is None:
            default_transcript = data_path / "transcript.txt"
            if default_transcript.exists():
                transcript = str(default_transcript)
            else:
                print("ERROR: No transcript.txt found in data directory.")
                print("Provide --transcript or create transcript.txt")
                print("Format: one line per audio file — filename|text")
                sys.exit(1)

        ref_audio = args.ref_audio
        if ref_audio is None:
            wavs = sorted(data_path.glob("*.wav"))
            if not wavs:
                print("ERROR: No .wav files found in data directory")
                sys.exit(1)
            ref_audio = str(wavs[0])
            print(f"Using first clip as reference audio: {ref_audio}")

        records = build_records_from_directory(
            str(data_path), transcript, ref_audio
        )
    elif data_path.suffix in (".jsonl", ".json"):
        with open(data_path, encoding="utf-8") as f:
            records = [
                TrainingRecord(**json.loads(line.strip()))
                for line in f
                if line.strip()
            ]
    else:
        print(f"ERROR: --data must be a directory or JSONL file, got: {data_path}")
        sys.exit(1)

    if not records:
        print("ERROR: No training records found!")
        sys.exit(1)

    print(f"Found {len(records)} training records")
    print("Loading model for speech tokenizer...")

    model = load_model_for_encoding(args.model_id)
    speech_tokenizer = model.speech_tokenizer
    if speech_tokenizer is None:
        print("ERROR: Model is missing speech_tokenizer")
        sys.exit(1)

    # Encode each audio
    output_records: list[TrainingRecord] = []
    for i, record in enumerate(records):
        basename = Path(record.audio).name
        print(
            f"  [{i + 1}/{len(records)}] Encoding {basename}...",
            end=" ",
            flush=True,
        )
        audio_np = load_audio_24k(record.audio)
        record.audio_codes = encode_audio(speech_tokenizer, audio_np)
        print(f"{len(record.audio_codes)} frames")
        output_records.append(record)

        if (i + 1) % 10 == 0:
            mx.clear_cache()

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in output_records:
            row = dataclasses.asdict(record)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ Wrote {len(output_records)} records to {args.output}")


# ──────────────────────────────────────────────
#  train
# ──────────────────────────────────────────────
def cmd_train(args: argparse.Namespace) -> None:
    """Run SFT training directly (no subprocess)."""
    from .train import main as train_main

    train_argv = [
        "train",
        "--train-jsonl", args.data,
        "--speaker-name", args.name,
        "--output", args.output,
        "--model", args.model_id,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--grad-accum", str(args.grad_accum),
    ]
    if args.lora_rank is not None:
        train_argv.extend(["--lora-rank", str(args.lora_rank)])
    if args.lora_scale is not None:
        train_argv.extend(["--lora-scale", str(args.lora_scale)])
    if args.save_every_epoch:
        train_argv.append("--save-every-epoch")
    if args.log_every is not None:
        train_argv.extend(["--log-every", str(args.log_every)])

    old_argv = sys.argv
    sys.argv = train_argv
    try:
        train_main()
    finally:
        sys.argv = old_argv


# ──────────────────────────────────────────────
#  voices
# ──────────────────────────────────────────────
def cmd_voices(args: argparse.Namespace) -> None:
    """List trained voice models in a directory."""
    search_dir = Path(args.directory)
    if not search_dir.exists():
        print(f"Directory not found: {search_dir}")
        sys.exit(1)

    voices: list[dict] = []

    def _check_voice_dir(d: Path) -> dict | None:
        if d.name in ("speech_tokenizer", "speaker_encoder", "__pycache__"):
            return None
        config_path = d / "config.json"
        weights = list(d.glob("model*.safetensors"))
        if config_path.exists() and weights:
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                model_type = cfg.get("tts_model_type", "unknown")
                talker_cfg = cfg.get("talker_config", {})
                spk_ids = talker_cfg.get("spk_id", {})
                return {
                    "path": str(d),
                    "type": model_type,
                    "speakers": list(spk_ids.keys()),
                    "weights_mb": sum(w.stat().st_size for w in weights) / 1e6,
                }
            except Exception:
                return None
        return None

    info = _check_voice_dir(search_dir)
    if info:
        voices.append(info)

    for sub in sorted(search_dir.iterdir()):
        if sub.is_dir():
            info = _check_voice_dir(sub)
            if info:
                voices.append(info)

    if not voices:
        print(f"No voice models found in {search_dir}")
        return

    print(f"Found {len(voices)} voice model(s):\n")
    for v in voices:
        speakers = ", ".join(v["speakers"]) if v["speakers"] else "none"
        print(f"  📁 {v['path']}")
        print(f"     Type: {v['type']}")
        print(f"     Speakers: {speakers}")
        print(f"     Weights: {v['weights_mb']:.1f} MB")
        print()


# ──────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="qwen-tts",
        description="Qwen3-TTS on Apple Silicon (MLX)",
    )
    sub = parser.add_subparsers(dest="command")

    # ── generate ──
    gen = sub.add_parser("generate", aliases=["g"], help="Generate speech from text")
    gen.add_argument("--prompt", "-p", required=True, help="Text to speak")
    gen.add_argument("--output", "-o", default="output.wav", help="Output WAV path")
    gen.add_argument(
        "--model", "-m", default="base", choices=_MODEL_REPOS.keys(),
        help="Model variant (default: base)",
    )
    gen.add_argument("--size", default=DEFAULT_SIZE, choices=SIZES, help="Model size")
    gen.add_argument("--voice", "-v", help="Reference audio for voice cloning (base)")
    gen.add_argument("--voice-text", help="Transcript of reference audio")
    gen.add_argument("--instruct", "-i", help="Style/emotion instruction")
    gen.add_argument("--speaker", "-S", help="Speaker name (custom-voice)")
    gen.add_argument("--voice-model", help="Path to fine-tuned voice model directory")
    gen.add_argument("--seed", "-s", type=int, help="Random seed")
    gen.add_argument("--play", action="store_true", help="Play audio after generation")

    # ── speakers ──
    spk = sub.add_parser("speakers", help="List available speakers")
    spk.add_argument("--size", default=DEFAULT_SIZE, choices=SIZES)
    spk.add_argument("--voice-model", help="Path to fine-tuned voice model")

    # ── prepare ──
    prep = sub.add_parser("prepare", help="Encode audio to codec tokens for training")
    prep.add_argument(
        "--data",
        required=True,
        help="Directory of audio files or input JSONL",
    )
    prep.add_argument("--transcript", help="Transcript file (filename|text per line)")
    prep.add_argument("--ref-audio", help="Reference audio for speaker embedding")
    prep.add_argument("--output", "-o", default="train.jsonl", help="Output JSONL path")
    prep.add_argument(
        "--model-id", default="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        help="Base model for speech tokenizer",
    )

    # ── train ──
    trn = sub.add_parser("train", help="Fine-tune a custom voice (LoRA SFT)")
    trn.add_argument("--name", required=True, help="Speaker name")
    trn.add_argument("--data", required=True, help="Training JSONL (from prepare)")
    trn.add_argument(
        "--output",
        "-o",
        default="./voice_output",
        help="Output directory",
    )
    trn.add_argument(
        "--model-id", default="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        help="Base model to fine-tune",
    )
    trn.add_argument("--epochs", type=int, default=3, help="Training epochs")
    trn.add_argument("--batch-size", type=int, default=1, help="Batch size")
    trn.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    trn.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    trn.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank (default: 8)",
    )
    trn.add_argument("--lora-scale", type=float, default=None, help="LoRA scale/alpha")
    trn.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Log every N optimizer steps",
    )
    trn.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save after every epoch",
    )

    # ── voices ──
    vcs = sub.add_parser("voices", help="List trained voice models in a directory")
    vcs.add_argument("directory", nargs="?", default=".", help="Directory to search")

    args = parser.parse_args()

    commands = {
        "generate": cmd_generate,
        "g": cmd_generate,
        "speakers": cmd_speakers,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "voices": cmd_voices,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
