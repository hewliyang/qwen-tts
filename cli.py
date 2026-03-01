import argparse
import sys
import time

import numpy as np

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


def load_model(model_id):
    from mlx_audio.tts.utils import load_model

    print(f"Loading {model_id}...", end=" ", flush=True)
    t0 = time.time()
    model = load_model(model_id)
    print(f"done ({time.time() - t0:.1f}s)")
    return model


def cmd_generate(args: argparse.Namespace) -> None:
    from mlx_audio.tts.generate import generate_audio

    if args.seed is not None:
        import mlx.core as mx

        np.random.seed(args.seed)
        mx.random.seed(args.seed)

    # If --voice-model is specified, load from local path
    if args.voice_model:
        model = load_model(args.voice_model)
    else:
        model_key: str = args.model
        size: str = args.size
        model_id = resolve_model_id(model_key, size)
        model = load_model(model_id)

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

    # Base model: voice cloning
    if args.voice:
        gen_kwargs["ref_audio"] = str(args.voice)
        gen_kwargs["ref_text"] = str(args.voice_text or "")

    # VoiceDesign / CustomVoice: instruct
    if args.instruct:
        gen_kwargs["instruct"] = str(args.instruct)

    # CustomVoice: speaker
    if args.speaker:
        gen_kwargs["voice"] = str(args.speaker)

    generate_audio(**gen_kwargs)  # type: ignore[arg-type]


def cmd_speakers(args: argparse.Namespace) -> None:
    if args.voice_model:
        model = load_model(args.voice_model)
    else:
        size = getattr(args, "size", DEFAULT_SIZE)
        model_id = resolve_model_id("custom-voice", size)
        model = load_model(model_id)

    speakers = model.supported_speakers if hasattr(model, "supported_speakers") else []
    if speakers:
        print(f"\nAvailable speakers ({len(speakers)}):")
        for s in sorted(speakers):
            print(f"  {s}")
    else:
        print("No speaker list found on model.")


def cmd_prepare(args: argparse.Namespace) -> None:
    """Prepare training data: encode audio clips to codec tokens."""
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "prepare_data",
        "--data",
        args.data,
        "--output",
        args.output,
        "--model",
        args.model_id,
    ]
    if args.transcript:
        cmd.extend(["--transcript", args.transcript])
    if args.ref_audio:
        cmd.extend(["--ref-audio", args.ref_audio])
    subprocess.run(cmd, check=True)


def cmd_train(args: argparse.Namespace) -> None:
    """Run SFT training."""
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "train",
        "--train-jsonl",
        args.data,
        "--speaker-name",
        args.name,
        "--output",
        args.output,
        "--model",
        args.model_id,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--grad-accum",
        str(args.grad_accum),
    ]
    if args.save_every_epoch:
        cmd.append("--save-every-epoch")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="qwen-tts",
        description="Qwen3-TTS on Apple Silicon (MLX)",
    )
    sub = parser.add_subparsers(dest="command")

    # generate
    gen = sub.add_parser("generate", aliases=["g"], help="Generate speech from text")
    gen.add_argument("--prompt", "-p", required=True, help="Text to speak")
    gen.add_argument(
        "--output",
        "-o",
        default="output.wav",
        help="Output WAV path (default: output.wav)",
    )
    gen.add_argument(
        "--model",
        "-m",
        default="base",
        choices=_MODEL_REPOS.keys(),
        help="Model variant (default: base)",
    )
    gen.add_argument(
        "--size",
        default=DEFAULT_SIZE,
        choices=SIZES,
        help=f"Model size (default: {DEFAULT_SIZE})",
    )
    gen.add_argument(
        "--voice",
        "-v",
        help="Reference audio for voice cloning (base model)",
    )
    gen.add_argument("--voice-text", help="Transcript of reference audio")
    gen.add_argument(
        "--instruct",
        "-i",
        help="Style/emotion instruction (voice-design or custom-voice)",
    )
    gen.add_argument(
        "--speaker",
        "-S",
        help="Speaker name (custom-voice model)",
    )
    gen.add_argument(
        "--voice-model",
        help="Path to a fine-tuned voice model directory",
    )
    gen.add_argument(
        "--seed",
        "-s",
        type=int,
        help="Random seed for reproducibility",
    )
    gen.add_argument(
        "--play",
        action="store_true",
        help="Play audio after generation (macOS)",
    )

    # speakers
    spk = sub.add_parser(
        "speakers",
        help="List available speakers for custom-voice model",
    )
    spk.add_argument(
        "--size",
        default=DEFAULT_SIZE,
        choices=SIZES,
        help=f"Model size (default: {DEFAULT_SIZE})",
    )
    spk.add_argument(
        "--voice-model",
        help="Path to a fine-tuned voice model directory",
    )

    # prepare
    prep = sub.add_parser(
        "prepare",
        help="Prepare training data (encode audio to codec tokens)",
    )
    prep.add_argument(
        "--data",
        required=True,
        help="Directory of WAV files or input JSONL",
    )
    prep.add_argument(
        "--transcript",
        help="Transcript file (filename|text per line)",
    )
    prep.add_argument(
        "--ref-audio",
        help="Reference audio for speaker embedding",
    )
    prep.add_argument(
        "--output",
        "-o",
        default="train.jsonl",
        help="Output JSONL path (default: train.jsonl)",
    )
    prep.add_argument(
        "--model-id",
        default="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        help="Base model for speech tokenizer",
    )

    # train
    trn = sub.add_parser("train", help="Fine-tune a custom voice (SFT)")
    trn.add_argument("--name", required=True, help="Speaker name")
    trn.add_argument(
        "--data",
        required=True,
        help="Training JSONL (from prepare)",
    )
    trn.add_argument(
        "--output",
        "-o",
        default="./voice_output",
        help="Output directory for checkpoint",
    )
    trn.add_argument(
        "--model-id",
        default="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
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
        "--save-every-epoch",
        action="store_true",
        help="Save checkpoint after every epoch",
    )

    args = parser.parse_args()

    if args.command in ("generate", "g"):
        cmd_generate(args)
    elif args.command == "speakers":
        cmd_speakers(args)
    elif args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
