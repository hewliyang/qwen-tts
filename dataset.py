"""
Dataset and collation for Qwen3-TTS SFT on MLX.

Ported from the official PyTorch TTSDataset + collate_fn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import librosa
import mlx.core as mx
import numpy as np
from mlx_audio.tts.models.qwen3_tts import ModelConfig
from mlx_audio.tts.models.qwen3_tts.qwen3_tts import (
    mel_spectrogram,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass
class TrainingRecord:
    """A single training example from the JSONL manifest."""

    audio: str
    text: str
    ref_audio: str
    audio_codes: list[list[int]] | None = None


@dataclass
class DatasetItem:
    """Processed item returned by TTSDataset.__getitem__."""

    text_ids: np.ndarray  # [T] int64
    audio_codes: np.ndarray  # [time, 16] int64
    ref_mel: mx.array  # [1, frames, 128]


@dataclass
class Batch:
    """Collated batch of items for training."""

    input_ids: mx.array  # [B, T, 2]
    ref_mels: mx.array  # [B, frames, 128]
    attention_mask: mx.array  # [B, T]
    text_embedding_mask: mx.array  # [B, T, 1]
    codec_embedding_mask: mx.array  # [B, T, 1]
    codec_0_labels: mx.array  # [B, T]
    codec_ids: mx.array  # [B, T, 16]
    codec_mask: mx.array  # [B, T]


def load_audio_24k(path: str) -> tuple[np.ndarray, int]:
    """Load audio at 24kHz mono."""
    audio, _sr = librosa.load(path, sr=24000, mono=True)
    return audio.astype(np.float32), 24000


def extract_mels(audio: np.ndarray, sr: int = 24000) -> mx.array:
    """Extract mel spectrogram from audio waveform."""
    assert sr == 24000, "Only 24kHz audio supported"
    mels = mel_spectrogram(
        mx.array(audio),
        n_fft=1024,
        num_mels=128,
        sample_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    )  # [1, frames, 128]
    mx.eval(mels)
    return mels


def build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def load_jsonl(path: str) -> list[TrainingRecord]:
    """Load JSONL training data."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line.strip()) for line in f if line.strip()]
    return [TrainingRecord(**row) for row in rows]


class TTSDataset:
    """Dataset for Qwen3-TTS SFT training."""

    def __init__(
        self,
        data_list: list[TrainingRecord],
        tokenizer: PreTrainedTokenizerBase,
        config: ModelConfig,
    ):
        """
        Args:
            data_list: Training records with audio_codes.
            tokenizer: HuggingFace tokenizer.
            config: ModelConfig (from model.config).
        """
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.config = config
        self._ref_mel_cache: dict[str, mx.array] = {}

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> DatasetItem:
        item = self.data_list[idx]

        text = item.text
        codes = item.audio_codes
        ref_audio_path = item.ref_audio

        assert codes is not None, f"Record {idx} has no audio_codes — run prepare first"

        # Tokenize text
        chat_text = build_assistant_text(text)
        text_ids = self.tokenizer.encode(chat_text)
        text_ids = text_ids[:-5]  # Remove trailing tokens

        # Audio codes as numpy
        audio_codes = np.array(codes, dtype=np.int64)

        # Extract mel from reference audio (cached)
        if ref_audio_path not in self._ref_mel_cache:
            ref_audio, sr = load_audio_24k(ref_audio_path)
            self._ref_mel_cache[ref_audio_path] = extract_mels(ref_audio, sr)
        ref_mel = self._ref_mel_cache[ref_audio_path]

        return DatasetItem(
            text_ids=np.array(text_ids, dtype=np.int64),
            audio_codes=audio_codes,
            ref_mel=ref_mel,
        )

    def collate(self, batch: list[DatasetItem]) -> Batch:
        """Collate batch items into padded tensors.

        Replicates the exact index arithmetic from the
        PyTorch collate_fn.
        """
        config = self.config
        talker_config = config.talker_config
        if talker_config is None:
            raise ValueError("Missing talker_config in model config")

        # Compute max sequence length
        item_lengths = [len(b.text_ids) + b.audio_codes.shape[0] for b in batch]
        max_length = max(item_lengths) + 8
        b_size = len(batch)
        t = max_length

        # Initialize arrays (numpy first, convert to mx)
        input_ids = np.zeros((b_size, t, 2), dtype=np.int64)
        codec_ids = np.zeros((b_size, t, 16), dtype=np.int64)
        text_embedding_mask = np.zeros((b_size, t), dtype=bool)
        codec_embedding_mask = np.zeros((b_size, t), dtype=bool)
        codec_mask = np.zeros((b_size, t), dtype=bool)
        attention_mask = np.zeros((b_size, t), dtype=np.int64)
        codec_0_labels = np.full((b_size, t), -100, dtype=np.int64)

        for i, data in enumerate(batch):
            text_ids = data.text_ids
            audio_codes = data.audio_codes

            audio_codec_0 = audio_codes[:, 0]
            tlen = len(text_ids)
            clen = len(audio_codec_0)

            # Precompute common offsets
            text_end = 8 + tlen - 3
            codec_start = 8 + tlen - 1
            codec_end = codec_start + clen
            pad_start = 8 + tlen - 2
            total_len = 8 + tlen + clen

            # --- Text channel (input_ids[:, :, 0]) ---
            input_ids[i, :3, 0] = text_ids[:3]
            input_ids[i, 3:7, 0] = config.tts_pad_token_id
            input_ids[i, 7, 0] = config.tts_bos_token_id
            input_ids[i, 8:text_end, 0] = text_ids[3:]
            input_ids[i, text_end, 0] = config.tts_eos_token_id
            input_ids[i, pad_start:total_len, 0] = config.tts_pad_token_id
            text_embedding_mask[i, :total_len] = True

            # --- Codec channel (input_ids[:, :, 1]) ---
            input_ids[i, 3:8, 1] = [
                talker_config.codec_nothink_id,
                talker_config.codec_think_bos_id,
                talker_config.codec_think_eos_id,
                0,  # placeholder for speaker embedding
                talker_config.codec_pad_id,
            ]
            input_ids[i, 8:text_end, 1] = talker_config.codec_pad_id
            input_ids[i, text_end, 1] = talker_config.codec_pad_id
            input_ids[i, pad_start, 1] = talker_config.codec_bos_id
            input_ids[i, codec_start:codec_end, 1] = audio_codec_0
            input_ids[i, codec_end, 1] = talker_config.codec_eos_token_id

            # Labels for codec_0
            codec_0_labels[i, codec_start:codec_end] = audio_codec_0
            codec_0_labels[i, codec_end] = talker_config.codec_eos_token_id

            # Full codec ids (all 16 codebooks)
            codec_ids[i, codec_start:codec_end, :] = audio_codes

            # Masks
            codec_embedding_mask[i, 3:total_len] = True
            codec_embedding_mask[i, 6] = False

            codec_mask[i, codec_start:codec_end] = True
            attention_mask[i, :total_len] = True

        # Concatenate ref mels
        ref_mels = mx.concatenate([data.ref_mel for data in batch], axis=0)

        text_emb_mask = mx.expand_dims(mx.array(text_embedding_mask), axis=-1)
        codec_emb_mask = mx.expand_dims(mx.array(codec_embedding_mask), axis=-1)

        return Batch(
            input_ids=mx.array(input_ids),
            ref_mels=ref_mels,
            attention_mask=mx.array(attention_mask),
            text_embedding_mask=text_emb_mask,
            codec_embedding_mask=codec_emb_mask,
            codec_0_labels=mx.array(codec_0_labels),
            codec_ids=mx.array(codec_ids),
            codec_mask=mx.array(codec_mask),
        )
