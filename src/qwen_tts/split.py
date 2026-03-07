"""Split long audio into sentence-aligned clips using ASR (parakeet).

Runs parakeet on the full audio to get sentence boundaries, then merges
short sentences and splits the audio into clips suitable for TTS training.
Produces clips + transcript.txt in one step.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf


@dataclass
class SplitConfig:
    audio: str  # path to input audio file
    output: str = "./clips"  # output directory
    min_dur: float = 3.0  # minimum clip duration (seconds)
    max_dur: float = 15.0  # maximum clip duration (seconds)
    sample_rate: int = 24000
    asr_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    pad: float = 0.1  # padding around cuts (seconds)


@dataclass
class Segment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class _TokenSentence:
    """Minimal stand-in for an ASR sentence with token timestamps."""

    start: float
    end: float
    text: str
    tokens: list
    duration: float = 0.0

    def __post_init__(self) -> None:
        self.duration = self.end - self.start


def _split_long_sentence(sentence: Any, max_dur: float) -> list[Segment]:
    """Split an ASR sentence that exceeds max_dur at punctuation pauses.

    Uses token-level timestamps to find comma/semicolon boundaries,
    picking the split point closest to the middle.
    """
    tokens = sentence.tokens
    total_start = sentence.start
    total_end = sentence.end

    # Find punctuation tokens that could be split points
    split_candidates = []
    for i, tok in enumerate(tokens):
        if tok.text.strip() in (",", ";", ":") and 0 < i < len(tokens) - 1:
            split_candidates.append(i)

    if not split_candidates:
        # No good split points — return as-is
        return [Segment(start=total_start, end=total_end, text=sentence.text.strip())]

    # Pick split point closest to the middle by time
    mid_time = (total_start + total_end) / 2
    best_idx = min(split_candidates, key=lambda i: abs(tokens[i].end - mid_time))

    left_text = "".join(t.text for t in tokens[: best_idx + 1]).strip()
    right_text = "".join(t.text for t in tokens[best_idx + 1 :]).strip()
    split_time = tokens[best_idx].end

    results: list[Segment] = []

    # Left half
    if (split_time - total_start) > max_dur:
        left_fake = _TokenSentence(
            start=total_start,
            end=split_time,
            text=left_text,
            tokens=tokens[: best_idx + 1],
        )
        results.extend(_split_long_sentence(left_fake, max_dur))
    else:
        results.append(Segment(start=total_start, end=split_time, text=left_text))

    # Right half
    if (total_end - split_time) > max_dur:
        right_fake = _TokenSentence(
            start=split_time,
            end=total_end,
            text=right_text,
            tokens=tokens[best_idx + 1 :],
        )
        results.extend(_split_long_sentence(right_fake, max_dur))
    else:
        results.append(Segment(start=split_time, end=total_end, text=right_text))

    return results


def _transcribe(audio_path: str, model_id: str, max_dur: float = 15.0) -> list[Segment]:
    """Run parakeet ASR and return sentence segments.

    Sentences longer than max_dur are split at punctuation
    boundaries using token-level timestamps from parakeet.
    """
    import time

    from mlx_audio.stt import load as load_stt

    print(f"Loading ASR model {model_id}...", end=" ", flush=True)
    t0 = time.time()
    model = load_stt(model_id)
    print(f"done ({time.time() - t0:.1f}s)")

    print(f"Transcribing {audio_path}...", end=" ", flush=True)
    t0 = time.time()
    result = model.generate(audio_path)
    print(f"done ({time.time() - t0:.1f}s)")

    segments = []
    for s in result.sentences:
        text = s.text.strip()
        if not text:
            continue
        if s.duration > max_dur:
            segments.extend(_split_long_sentence(s, max_dur))
        else:
            segments.append(Segment(start=s.start, end=s.end, text=text))

    return segments


def _merge_segments(
    segments: list[Segment], min_dur: float, max_dur: float
) -> list[Segment]:
    """Merge short adjacent segments, respecting max duration."""
    if not segments:
        return []

    merged: list[Segment] = []
    cur = Segment(
        start=segments[0].start,
        end=segments[0].end,
        text=segments[0].text,
    )

    def _flush(seg: Segment) -> None:
        if seg.duration >= min_dur:
            merged.append(seg)

    for seg in segments[1:]:
        combined_dur = seg.end - cur.start
        if combined_dur <= max_dur:
            cur.end = seg.end
            cur.text = cur.text + " " + seg.text
        else:
            _flush(cur)
            cur = Segment(start=seg.start, end=seg.end, text=seg.text)

    _flush(cur)

    return merged


def run_split(config: SplitConfig) -> list[dict]:
    """Run the full split pipeline. Returns list of {file, text, duration}."""
    audio_path = Path(config.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. ASR → sentence segments (splits oversized sentences at punctuation)
    segments = _transcribe(str(audio_path), config.asr_model, config.max_dur)
    print(f"  Found {len(segments)} sentences")

    # 2. Merge short segments
    clips = _merge_segments(segments, config.min_dur, config.max_dur)
    print(f"  Merged into {len(clips)} clips")

    # 3. Load audio and slice
    audio, sr = librosa.load(str(audio_path), sr=config.sample_rate, mono=True)
    total_dur = len(audio) / sr

    transcript_lines = []
    results = []

    for i, clip in enumerate(clips):
        start_sample = max(0, int((clip.start - config.pad) * sr))
        end_sample = min(len(audio), int((clip.end + config.pad) * sr))
        clip_audio = audio[start_sample:end_sample]
        clip_dur = len(clip_audio) / sr

        filename = f"clip_{i:03d}.wav"
        out_path = output_dir / filename
        sf.write(str(out_path), clip_audio, sr)

        transcript_lines.append(f"{filename}|{clip.text}")
        results.append(
            {"file": filename, "text": clip.text, "duration": round(clip_dur, 1)}
        )
        print(f"  {filename}: {clip_dur:.1f}s  {clip.text[:60]}...")

    # 4. Write transcript
    transcript_path = output_dir / "transcript.txt"
    transcript_path.write_text("\n".join(transcript_lines) + "\n")

    print(f"\n✅ {len(results)} clips written to {output_dir}/")
    print(f"   Transcript: {transcript_path}")
    print(
        f"   Total: {sum(r['duration'] for r in results):.1f}s"
        f" from {total_dur:.1f}s source"
    )

    return results
