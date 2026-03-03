"""
Prepare training data for Qwen3-TTS SFT.

Takes a directory of WAV files (or a JSONL manifest) and encodes each
clip through the speech tokenizer to produce audio_codes.

Input directory layout:
    clips/
        001.wav
        002.wav
        ...
    transcript.txt   # one line per wav: "001.wav|Hello world"

Or a pre-built JSONL with {audio, text, ref_audio} per line.

Output: JSONL with audio_codes added.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import cast

import librosa
import mlx.core as mx
import numpy as np
from mlx_audio.tts.models.qwen3_tts import Model as Qwen3TTSModel
from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
    Qwen3TTSSpeechTokenizer,
)

from .dataset import TrainingRecord


def load_model_for_encoding(model_id: str) -> Qwen3TTSModel:
    """Load a base model with the speech tokenizer encoder.

    The default mlx-audio post_load_hook skips building the encoder.
    We patch the speech tokenizer to include the encoder if the
    config and weights are available.
    """
    from mlx_audio.tts.utils import load_model
    from mlx_audio.utils import get_model_path

    model = cast(Qwen3TTSModel, load_model(model_id))

    # If encoder already loaded, great
    if (
        model.speech_tokenizer is not None
        and model.speech_tokenizer.has_encoder
    ):
        return model

    # Otherwise, try to load encoder manually
    model_path = get_model_path(model_id)
    speech_tokenizer_path = model_path / "speech_tokenizer"
    config_path = speech_tokenizer_path / "config.json"

    if not config_path.exists():
        raise RuntimeError(
            "No speech_tokenizer/config.json found in model"
        )

    with open(config_path) as f:
        st_config_dict = json.load(f)

    encoder_config_dict = st_config_dict.get("encoder_config")
    if encoder_config_dict is None:
        raise RuntimeError(
            "No encoder_config in speech tokenizer config"
        )

    print(
        "  Building speech tokenizer encoder "
        "(skipped by mlx-audio)..."
    )

    from mlx_audio.tts.models.qwen3_tts.config import (
        Qwen3TTSTokenizerConfig,
        Qwen3TTSTokenizerDecoderConfig,
        Qwen3TTSTokenizerEncoderConfig,
        filter_dict_for_dataclass,
    )
    from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
        Qwen3TTSSpeechTokenizer,
    )

    # Build full config with encoder
    enc_filtered = filter_dict_for_dataclass(
        Qwen3TTSTokenizerEncoderConfig, encoder_config_dict
    )
    # Handle _frame_rate -> frame_rate rename
    if (
        "frame_rate" not in enc_filtered
        and "_frame_rate" in encoder_config_dict
    ):
        enc_filtered["frame_rate"] = encoder_config_dict[
            "_frame_rate"
        ]
    encoder_config = Qwen3TTSTokenizerEncoderConfig(
        **enc_filtered
    )

    dec_filtered = filter_dict_for_dataclass(
        Qwen3TTSTokenizerDecoderConfig,
        st_config_dict.get("decoder_config", {}),
    )
    decoder_config = Qwen3TTSTokenizerDecoderConfig(
        **dec_filtered
    )

    tokenizer_config = Qwen3TTSTokenizerConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    for k, v in st_config_dict.items():
        is_sub_config = k in (
            "decoder_config",
            "encoder_config",
        )
        if not is_sub_config and hasattr(tokenizer_config, k):
            setattr(tokenizer_config, k, v)

    # Build new speech tokenizer with encoder
    speech_tokenizer = Qwen3TTSSpeechTokenizer(
        tokenizer_config
    )

    # Load weights
    tokenizer_weights: dict[str, mx.array] = {}
    for wf in speech_tokenizer_path.glob("*.safetensors"):
        loaded = mx.load(str(wf))
        tokenizer_weights.update(loaded)  # type: ignore[arg-type]

    if tokenizer_weights:
        sanitized = Qwen3TTSSpeechTokenizer.sanitize(
            tokenizer_weights
        )
        speech_tokenizer.load_weights(
            list(sanitized.items()), strict=False
        )
        mx.eval(speech_tokenizer.parameters())
        speech_tokenizer.eval()

        # Initialize encoder codebooks
        if speech_tokenizer.encoder_model is not None:
            quantizer = (
                speech_tokenizer.encoder_model.quantizer
            )
            for layer in quantizer.rvq_first.vq.layers:
                layer.codebook.update_in_place()
            for layer in quantizer.rvq_rest.vq.layers:
                layer.codebook.update_in_place()
            print("  Initialized encoder codebooks")

    model.load_speech_tokenizer(speech_tokenizer)

    loaded_speech_tokenizer = model.speech_tokenizer
    if loaded_speech_tokenizer is None or not loaded_speech_tokenizer.has_encoder:
        raise RuntimeError(
            "Failed to load speech tokenizer encoder"
        )

    print("  ✅ Encoder loaded successfully")
    return model


def load_audio_24k(path: str) -> np.ndarray:
    """Load audio file, resample to 24kHz mono."""
    audio, _sr = librosa.load(path, sr=24000, mono=True)
    return audio.astype(np.float32)


def encode_audio(
    speech_tokenizer: Qwen3TTSSpeechTokenizer, audio_np: np.ndarray
) -> list[list[int]]:
    """Encode audio through speech tokenizer."""
    audio_mx = mx.array(audio_np)[None, None, :]
    codes = speech_tokenizer.encode(audio_mx)
    mx.eval(codes)
    codes_np = np.array(codes[0]).T  # [time, 16]
    return cast(list[list[int]], codes_np.tolist())


def build_records_from_directory(
    data_dir: str,
    transcript_path: str,
    ref_audio: str,
) -> list[TrainingRecord]:
    """Build training records from a directory of WAVs."""
    dir_path = Path(data_dir)
    records: list[TrainingRecord] = []

    # Parse transcript: "filename|text" per line
    transcript_map: dict[str, str] = {}
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|", 1)
            fname = parts[0].strip()
            text = parts[1].strip()
            transcript_map[fname] = text

    # Build records
    for fname, text in transcript_map.items():
        audio_path = dir_path / fname
        if not audio_path.exists():
            # Try without extension
            for ext in [".wav", ".mp3", ".flac"]:
                candidate = dir_path / (fname + ext)
                if candidate.exists():
                    audio_path = candidate
                    break
        if not audio_path.exists():
            print(
                f"  Warning: skipping {fname} (file not found)"
            )
            continue
        records.append(
            TrainingRecord(
                audio=str(audio_path),
                text=text,
                ref_audio=ref_audio,
            )
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare training data for Qwen3-TTS SFT"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Directory of audio files OR input JSONL path",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        help=(
            "Transcript file (filename|text per line). "
            "Required if --data is a directory."
        ),
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help=(
            "Reference audio for speaker embedding "
            "(used for all samples). "
            "If not specified, uses the first audio clip."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output JSONL path with audio_codes",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=(
            "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        ),
        help="Base model to use for speech tokenizer",
    )
    args = parser.parse_args()

    # Build or load records
    data_path = Path(args.data)
    if data_path.is_dir():
        if args.transcript is None:
            default_transcript = data_path / "transcript.txt"
            if default_transcript.exists():
                args.transcript = str(default_transcript)
            else:
                raise ValueError(
                    "When --data is a directory, you must "
                    "provide --transcript or have a "
                    "transcript.txt in the directory."
                )
        ref_audio = args.ref_audio
        if ref_audio is None:
            wavs = sorted(data_path.glob("*.wav"))
            if not wavs:
                raise ValueError(
                    "No .wav files found in data directory"
                )
            ref_audio = str(wavs[0])
            print(
                "Using first clip as reference audio: "
                f"{ref_audio}"
            )
        records = build_records_from_directory(
            str(data_path), args.transcript, ref_audio
        )
    elif data_path.suffix in (".jsonl", ".json"):
        with open(data_path, encoding="utf-8") as f:
            records = [
                TrainingRecord(**json.loads(line.strip()))
                for line in f
                if line.strip()
            ]
    else:
        raise ValueError(
            "--data must be a directory or JSONL file, "
            f"got: {data_path}"
        )

    if not records:
        raise ValueError("No training records found!")

    print(f"Found {len(records)} training records")
    print("Loading model for speech tokenizer...")

    model = load_model_for_encoding(args.model)
    speech_tokenizer = model.speech_tokenizer
    if speech_tokenizer is None:
        raise RuntimeError("Model is missing speech_tokenizer")

    # Encode each audio
    output_records: list[TrainingRecord] = []
    for i, record in enumerate(records):
        basename = Path(record.audio).name
        print(
            f"  [{i + 1}/{len(records)}] "
            f"Encoding {basename}...",
            end=" ",
            flush=True,
        )

        audio_np = load_audio_24k(record.audio)
        record.audio_codes = encode_audio(
            speech_tokenizer, audio_np
        )

        print(f"{len(record.audio_codes)} frames")
        output_records.append(record)

        # Clear MLX cache periodically
        if (i + 1) % 10 == 0:
            mx.clear_cache()

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in output_records:
            row = dataclasses.asdict(record)
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    print(
        f"\nWrote {len(output_records)} records "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
