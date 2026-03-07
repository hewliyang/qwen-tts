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
    from mlx_audio.utils import get_model_path

    print(f"Loading TTS model {model_id}...", end=" ", flush=True)
    t0 = time.time()
    model = cast(Qwen3TTSModel, load_model(get_model_path(model_id)))
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
    from .prepare_data import PrepareConfig, run_prepare

    kw = {k: v for k, v in vars(args).items() if k != "command"}
    run_prepare(PrepareConfig(**kw))


# ──────────────────────────────────────────────
#  train
# ──────────────────────────────────────────────
def cmd_train(args: argparse.Namespace) -> None:
    """Run SFT training directly."""
    from .train import TrainConfig, run_training

    kw = {k: v for k, v in vars(args).items() if k != "command" and v is not None}
    run_training(TrainConfig(**kw))


# ──────────────────────────────────────────────
#  check
# ──────────────────────────────────────────────
def cmd_check(args: argparse.Namespace) -> None:
    """Evaluate voice quality using ASR and/or Gemini as judge."""
    from .check import CheckConfig, run_check

    # Parse --expected-text FILE=TEXT pairs
    expected_texts: dict[str, str] = {}
    if args.expected_text:
        for item in args.expected_text:
            if "=" in item:
                fname, text = item.split("=", 1)
                expected_texts[fname.strip()] = text.strip()

    config = CheckConfig(
        generated=args.generated,
        reference=args.reference,
        speaker=args.speaker,
        pairs=args.pairs,
        max_clips=args.max_clips,
        model=args.gemini_model,
        api_key=args.api_key,
        expected_texts=expected_texts,
        asr_model=args.asr_model,
        skip_gemini=args.asr_only,
    )
    result = run_check(config)
    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))


# ──────────────────────────────────────────────
#  split
# ──────────────────────────────────────────────
def cmd_split(args: argparse.Namespace) -> None:
    """Split long audio into sentence-aligned clips using ASR."""
    from .split import SplitConfig, run_split

    config = SplitConfig(
        audio=args.audio,
        output=args.output,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        asr_model=args.asr_model,
        pad=args.pad,
    )
    run_split(config)


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
        "--model",
        "-m",
        default="base",
        choices=_MODEL_REPOS.keys(),
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
        "--model-id",
        default="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
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

    # ── check ──
    chk = sub.add_parser("check", help="Evaluate voice quality using Gemini as judge")
    chk.add_argument(
        "--generated",
        "-g",
        required=True,
        help="Generated WAV file or directory of WAVs to evaluate",
    )
    chk.add_argument(
        "--reference",
        "-r",
        default="",
        help="Directory of reference clips from the real speaker",
    )
    chk.add_argument(
        "--speaker",
        "-S",
        default="",
        help="Target speaker name for Gemini eval (e.g. 'Lee Kuan Yew')",
    )
    chk.add_argument(
        "--pairs",
        type=int,
        default=3,
        help="Number of ref/gen pairs to compare",
    )
    chk.add_argument(
        "--max-clips",
        type=int,
        default=5,
        help="Max generated clips to evaluate",
    )
    chk.add_argument(
        "--gemini-model",
        default="gemini-3-flash-preview",
        help="Gemini model to use as judge",
    )
    chk.add_argument(
        "--api-key",
        default="",
        help="Gemini API key (or set GEMINI_API_KEY)",
    )
    chk.add_argument(
        "--expected-text",
        nargs="+",
        metavar="FILE=TEXT",
        help="Expected text per file for ASR check (e.g. gen_01.wav='Hello world')",
    )
    chk.add_argument(
        "--asr-model",
        default="mlx-community/parakeet-tdt-0.6b-v3",
        help="ASR model for intelligibility check",
    )
    chk.add_argument(
        "--asr-only",
        action="store_true",
        help="Run ASR check only, skip Gemini (no API key needed)",
    )
    chk.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output raw JSON",
    )

    # ── split ──
    spl = sub.add_parser(
        "split", help="Split long audio into sentence-aligned clips using ASR"
    )
    spl.add_argument("audio", help="Path to input audio file")
    spl.add_argument(
        "--output", "-o", default="./clips", help="Output directory for clips"
    )
    spl.add_argument(
        "--min-dur", type=float, default=3.0, help="Minimum clip duration (seconds)"
    )
    spl.add_argument(
        "--max-dur", type=float, default=15.0, help="Maximum clip duration (seconds)"
    )
    spl.add_argument(
        "--asr-model",
        default="mlx-community/parakeet-tdt-0.6b-v3",
        help="ASR model for transcription",
    )
    spl.add_argument(
        "--pad",
        type=float,
        default=0.1,
        help="Padding around clip boundaries (seconds)",
    )

    # ── voices ──
    vcs = sub.add_parser("voices", help="List trained voice models in a directory")
    vcs.add_argument("directory", nargs="?", default=".", help="Directory to search")

    args = parser.parse_args()

    commands = {
        "generate": cmd_generate,
        "g": cmd_generate,
        "speakers": cmd_speakers,
        "split": cmd_split,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "check": cmd_check,
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
