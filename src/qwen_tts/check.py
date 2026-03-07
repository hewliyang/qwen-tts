"""
Evaluate voice quality using ASR (local) and Gemini (remote).

Two complementary checks:
- ASR check: runs parakeet locally to verify intelligibility / word accuracy.
  No API calls, fast. Catches garbled speech, missing words, mispronunciations.
- Gemini check: uses Gemini as a judge to assess speaker similarity,
  naturalness, and audio quality via structured outputs.

API calls: up to (max_clips + pairs) for Gemini.  ASR check is fully local.
"""

import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from google import genai
from google.genai import types

# ── Pass/fail thresholds ─────────────────────────────────────────────

WER_THRESHOLD = 0.3  # WER below this = intelligible
INTELLIGIBILITY_THRESHOLD = 0.8  # fraction of clips that must be intelligible
SPEAKER_MATCH_THRESHOLD = 0.8  # fraction of clips matching target speaker
NATURALNESS_THRESHOLD = 0.8  # fraction of clips sounding natural
SIMILARITY_THRESHOLD = 6.0  # minimum avg similarity score (1-10)
MAX_TOP_ISSUES = 5  # number of most common issues to surface

# ── Structured output schemas ────────────────────────────────────────

_SINGLE_CLIP_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "transcription": types.Schema(
            type=types.Type.STRING,
            description="What is being said in the clip.",
        ),
        "speaker_match": types.Schema(
            type=types.Type.STRING,
            enum=["yes", "no", "uncertain"],
            description="Does the speaker sound like the target speaker?",
        ),
        "confidence": types.Schema(
            type=types.Type.STRING,
            enum=["low", "medium", "high"],
            description="Confidence that this is / sounds like the target speaker.",
        ),
        "natural": types.Schema(
            type=types.Type.BOOLEAN,
            description="Does the speech sound natural (not robotic or glitchy)?",
        ),
        "audio_quality": types.Schema(
            type=types.Type.STRING,
            enum=["poor", "fair", "good", "excellent"],
            description="Overall audio quality.",
        ),
        "issues": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description=(
                "List of specific issues (e.g. 'robotic tone', "
                "'wrong accent', 'audio glitch'). Empty if none."
            ),
        ),
    },
    required=[
        "transcription",
        "speaker_match",
        "confidence",
        "natural",
        "audio_quality",
        "issues",
    ],
)

_PAIR_CLIP_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "same_speaker": types.Schema(
            type=types.Type.STRING,
            enum=["yes", "no", "uncertain"],
            description="Do both clips sound like the same speaker?",
        ),
        "confidence": types.Schema(
            type=types.Type.STRING,
            enum=["low", "medium", "high"],
            description="Confidence in the same-speaker judgement.",
        ),
        "similarity_score": types.Schema(
            type=types.Type.INTEGER,
            description="Voice similarity 1-10 (10 = identical speaker).",
        ),
        "accent_match": types.Schema(type=types.Type.BOOLEAN),
        "cadence_match": types.Schema(type=types.Type.BOOLEAN),
        "tone_match": types.Schema(type=types.Type.BOOLEAN),
        "generated_natural": types.Schema(
            type=types.Type.BOOLEAN,
            description="Does the generated clip sound natural?",
        ),
        "issues": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Issues with the generated clip. Empty if none.",
        ),
    },
    required=[
        "same_speaker",
        "confidence",
        "similarity_score",
        "accent_match",
        "cadence_match",
        "tone_match",
        "generated_natural",
        "issues",
    ],
)


# ── Result dataclasses ───────────────────────────────────────────────


@dataclass
class ASRResult:
    """Result of ASR intelligibility check for a single clip."""

    file: str
    expected: str
    transcription: str | None
    wer: float
    intelligible: bool


@dataclass
class SingleEvalResult:
    """Gemini evaluation of a single generated clip."""

    file: str
    transcription: str
    speaker_match: str  # "yes" | "no" | "uncertain"
    confidence: str  # "low" | "medium" | "high"
    natural: bool
    audio_quality: str  # "poor" | "fair" | "good" | "excellent"
    issues: list[str]


@dataclass
class PairEvalResult:
    """Gemini comparison of a reference clip vs generated clip."""

    reference: str
    generated: str
    same_speaker: str  # "yes" | "no" | "uncertain"
    confidence: str  # "low" | "medium" | "high"
    similarity_score: int  # 1-10
    accent_match: bool
    cadence_match: bool
    tone_match: bool
    generated_natural: bool
    issues: list[str]


@dataclass
class CheckReport:
    """Aggregated metrics from all evaluations."""

    overall_pass: bool
    top_issues: list[str]
    avg_wer: float | None = None
    intelligibility_rate: float | None = None
    speaker_match_rate: float | None = None
    naturalness_rate: float | None = None
    avg_similarity: float | None = None


@dataclass
class CheckResult:
    """Complete result from a voice quality check run."""

    asr_results: list[ASRResult]
    single_results: list[SingleEvalResult]
    pair_results: list[PairEvalResult]
    report: CheckReport

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        d = asdict(self)
        # Strip None values from report
        d["report"] = {k: v for k, v in d["report"].items() if v is not None}
        return d


# ── Config ───────────────────────────────────────────────────────────


@dataclass
class CheckConfig:
    """Configuration for voice quality check."""

    generated: str = ""  # path to generated WAV or directory of WAVs
    reference: str = ""  # path to reference clips directory
    speaker: str = ""  # target speaker name (e.g. "Lee Kuan Yew")
    pairs: int = 3  # number of ref/gen pairs to compare
    max_clips: int = 5  # max generated clips to evaluate individually
    model: str = "gemini-3-flash-preview"
    api_key: str = ""  # Gemini API key (falls back to env)
    # ASR check options
    expected_texts: dict[str, str] = field(default_factory=dict)
    # mapping of filename -> expected text (e.g. {"gen_01.wav": "Hello world"})
    asr_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    skip_gemini: bool = False  # run ASR only, no Gemini calls


# ── Helpers ──────────────────────────────────────────────────────────


def _get_client(config: CheckConfig) -> genai.Client:
    key = config.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            from dotenv import load_dotenv

            load_dotenv()
            key = os.environ.get("GEMINI_API_KEY", "")
        except ImportError:
            pass
    if not key:
        raise RuntimeError(
            "No Gemini API key. Set GEMINI_API_KEY env var, put it in .env, "
            "or pass --api-key."
        )
    return genai.Client(api_key=key)


def _load_audio_part(path: Path) -> types.Part:
    with open(path, "rb") as f:
        data = f.read()
    suffix = path.suffix.lower()
    mime = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".aac": "audio/aac",
        ".aiff": "audio/aiff",
    }.get(suffix, "audio/wav")
    return types.Part.from_bytes(data=data, mime_type=mime)


def _collect_wavs(path: str, max_n: int) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        wavs = sorted(p.glob("*.wav"))
        if not wavs:
            wavs = sorted(p.glob("*.mp3")) or sorted(p.glob("*.flac"))
        if len(wavs) > max_n:
            wavs = random.sample(wavs, max_n)
        return wavs
    raise FileNotFoundError(f"Path not found: {path}")


# ── ASR evaluation (local, no API calls) ─────────────────────────────


def _normalize_text(text: str) -> str:
    """Normalize text for WER comparison."""
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word error rate using edit distance."""
    ref_words = _normalize_text(reference).split()
    hyp_words = _normalize_text(hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Levenshtein on words
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


_asr_model_cache: dict[str, object] = {}


def _get_asr_model(asr_model: str) -> object:
    """Load and cache the ASR model."""
    if asr_model not in _asr_model_cache:
        from mlx_audio.stt import load

        _asr_model_cache[asr_model] = load(asr_model)
    return _asr_model_cache[asr_model]


def transcribe_clip(clip: Path, asr_model: str) -> str | None:
    """Transcribe a single audio clip using mlx_audio.stt."""
    model = _get_asr_model(asr_model)
    result = model.generate(clip)  # type: ignore[union-attr]
    text: str | None = getattr(result, "text", None)
    return text.strip() if text else None


def eval_asr(
    clips: list[Path],
    expected_texts: dict[str, str],
    asr_model: str,
) -> list[ASRResult]:
    """Run ASR on generated clips, compare against expected text."""
    results = []
    for clip in clips:
        expected = expected_texts.get(clip.name)
        if expected is None:
            continue

        print(f"  ASR {clip.name}...", end=" ", flush=True)
        transcription = transcribe_clip(clip, asr_model)

        if transcription is None:
            print("⚠️  failed")
            results.append(
                ASRResult(
                    file=clip.name,
                    expected=expected,
                    transcription=None,
                    wer=1.0,
                    intelligible=False,
                )
            )
            continue

        wer = _word_error_rate(expected, transcription)
        intelligible = wer < WER_THRESHOLD
        icon = "✅" if intelligible else "❌"
        print(f'{icon} WER={wer:.0%} "{transcription[:60]}"')

        results.append(
            ASRResult(
                file=clip.name,
                expected=expected,
                transcription=transcription,
                wer=round(wer, 3),
                intelligible=intelligible,
            )
        )

    return results


# ── Gemini evaluation (remote API calls) ─────────────────────────────


def eval_single(
    client: genai.Client,
    model: str,
    clip: Path,
    speaker: str,
) -> SingleEvalResult:
    """Evaluate a single generated clip against a known speaker identity."""
    prompt = (
        f"Listen to this audio clip. The target speaker is {speaker}. "
        f"Assess whether this sounds like {speaker} and evaluate the audio quality. "
        f"Consider accent, cadence, tone, and naturalness."
    )
    response = client.models.generate_content(
        model=model,
        contents=[_load_audio_part(clip), prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_SINGLE_CLIP_SCHEMA,
        ),
    )
    assert response.text is not None
    data = json.loads(response.text)
    return SingleEvalResult(
        file=clip.name,
        transcription=data["transcription"],
        speaker_match=data["speaker_match"],
        confidence=data["confidence"],
        natural=data["natural"],
        audio_quality=data["audio_quality"],
        issues=data.get("issues", []),
    )


def eval_pair(
    client: genai.Client,
    model: str,
    ref_clip: Path,
    gen_clip: Path,
    speaker: str,
) -> PairEvalResult:
    """Compare a reference clip and a generated clip for speaker similarity."""
    prompt = (
        f"You are given two audio clips. Clip 1 is a real recording of {speaker}. "
        f"Clip 2 is AI-generated speech intended to sound like {speaker}. "
        f"Compare the two voices. Assess whether they sound like the same person "
        f"and rate the similarity."
    )
    response = client.models.generate_content(
        model=model,
        contents=[
            "Clip 1 (reference):",
            _load_audio_part(ref_clip),
            "Clip 2 (generated):",
            _load_audio_part(gen_clip),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_PAIR_CLIP_SCHEMA,
        ),
    )
    assert response.text is not None
    data = json.loads(response.text)
    return PairEvalResult(
        reference=ref_clip.name,
        generated=gen_clip.name,
        same_speaker=data["same_speaker"],
        confidence=data["confidence"],
        similarity_score=data["similarity_score"],
        accent_match=data["accent_match"],
        cadence_match=data["cadence_match"],
        tone_match=data["tone_match"],
        generated_natural=data["generated_natural"],
        issues=data.get("issues", []),
    )


def _compute_report(
    single_results: list[SingleEvalResult],
    pair_results: list[PairEvalResult],
    asr_results: list[ASRResult] | None = None,
) -> CheckReport:
    """Compute report metrics locally from structured eval results."""
    # Speaker match rate from single evals
    if single_results:
        matches = sum(1 for r in single_results if r.speaker_match == "yes")
        speaker_match_rate = matches / len(single_results)
        natural_count = sum(1 for r in single_results if r.natural)
        naturalness_rate = natural_count / len(single_results)
    else:
        speaker_match_rate = 0.0
        naturalness_rate = 0.0

    # Avg similarity from pair evals
    if pair_results:
        scores = [r.similarity_score for r in pair_results]
        avg_similarity = sum(scores) / len(scores) if scores else 0.0
        pair_natural = sum(1 for r in pair_results if r.generated_natural)
        pair_naturalness = pair_natural / len(pair_results)
        if single_results:
            naturalness_rate = (naturalness_rate + pair_naturalness) / 2
        else:
            naturalness_rate = pair_naturalness
    else:
        avg_similarity = 0.0

    # ASR metrics
    if asr_results:
        wers = [r.wer for r in asr_results]
        avg_wer = sum(wers) / len(wers)
        intelligibility_rate = sum(1 for r in asr_results if r.intelligible) / len(
            asr_results
        )
    else:
        avg_wer = 0.0
        intelligibility_rate = 1.0

    # Collect all issues
    all_issues: list[str] = []
    for r in single_results:
        all_issues.extend(r.issues)
    for r in pair_results:
        all_issues.extend(r.issues)
    if asr_results:
        for r in asr_results:
            if not r.intelligible:
                all_issues.append(f"unintelligible ({r.file})")
    top_issues = (
        [issue for issue, _ in Counter(all_issues).most_common(MAX_TOP_ISSUES)]
        if all_issues
        else []
    )

    # Pass criteria
    overall_pass = (
        intelligibility_rate >= INTELLIGIBILITY_THRESHOLD
        and (speaker_match_rate >= SPEAKER_MATCH_THRESHOLD or not single_results)
        and (naturalness_rate >= NATURALNESS_THRESHOLD or not single_results)
        and (avg_similarity >= SIMILARITY_THRESHOLD or not pair_results)
    )

    return CheckReport(
        overall_pass=overall_pass,
        top_issues=top_issues,
        avg_wer=round(avg_wer, 3) if asr_results else None,
        intelligibility_rate=round(intelligibility_rate, 2) if asr_results else None,
        speaker_match_rate=round(speaker_match_rate, 2) if single_results else None,
        naturalness_rate=round(naturalness_rate, 2) if single_results else None,
        avg_similarity=round(avg_similarity, 1) if pair_results else None,
    )


# ── Main entry point ─────────────────────────────────────────────────


def run_check(config: CheckConfig) -> CheckResult:
    """Run the full voice quality check and return structured results."""
    gen_clips = _collect_wavs(config.generated, config.max_clips)
    if not gen_clips:
        raise FileNotFoundError(f"No audio files found at: {config.generated}")

    # ── ASR check (local, no API calls) ──
    asr_results: list[ASRResult] = []
    if config.expected_texts:
        clips_with_text = [c for c in gen_clips if c.name in config.expected_texts]
        if clips_with_text:
            print(f"🗣️  ASR check: {len(clips_with_text)} clip(s) with expected text\n")
            asr_results = eval_asr(
                clips_with_text, config.expected_texts, config.asr_model
            )
            print()

    if config.skip_gemini:
        report = _compute_report([], [], asr_results or None)

        print(f"{'=' * 60}")
        print("  Voice Check (ASR only)")
        print(f"{'=' * 60}")
        passed = "✅ PASS" if report.overall_pass else "❌ FAIL"
        print(f"  Result:           {passed}")
        if report.avg_wer is not None:
            print(f"  Avg WER:          {report.avg_wer:.0%}")
            print(f"  Intelligibility:  {report.intelligibility_rate:.0%}")
        if report.top_issues:
            print(f"  Issues:           {', '.join(report.top_issues)}")
        print(f"{'=' * 60}\n")

        return CheckResult(
            asr_results=asr_results,
            single_results=[],
            pair_results=[],
            report=report,
        )

    # ── Gemini check (remote API calls) ──
    client = _get_client(config)

    n_pairs = 0
    if config.reference:
        ref_clips = _collect_wavs(config.reference, config.pairs * 2)
        n_pairs = min(config.pairs, len(ref_clips), len(gen_clips))

    total_calls = len(gen_clips) + n_pairs
    print(
        f"🔍 Gemini check: {len(gen_clips)} clip(s) + {n_pairs} pair(s) "
        f"= {total_calls} API call(s)"
    )
    print(f"   Speaker: {config.speaker}  Model: {config.model}\n")

    # Single clip evaluations
    single_results: list[SingleEvalResult] = []
    for clip in gen_clips:
        print(f"  Evaluating {clip.name}...", end=" ", flush=True)
        result = eval_single(client, config.model, clip, config.speaker)
        single_results.append(result)
        match = "✅" if result.speaker_match == "yes" else "❌"
        natural = "✅" if result.natural else "⚠️"
        print(
            f"{match} match={result.speaker_match} conf={result.confidence} "
            f"natural={natural} quality={result.audio_quality}"
        )

    # Pair comparisons
    pair_results: list[PairEvalResult] = []
    if n_pairs > 0:
        ref_sample = random.sample(ref_clips, n_pairs)
        gen_sample = (
            random.sample(gen_clips, n_pairs)
            if len(gen_clips) > n_pairs
            else gen_clips[:n_pairs]
        )
        print()
        for ref, gen in zip(ref_sample, gen_sample, strict=True):
            print(f"  Pair: {ref.name} ↔ {gen.name}...", end=" ", flush=True)
            result = eval_pair(client, config.model, ref, gen, config.speaker)
            pair_results.append(result)
            match = "✅" if result.same_speaker == "yes" else "❌"
            print(
                f"{match} same={result.same_speaker} "
                f"similarity={result.similarity_score}/10"
            )

    # Compute report locally
    report = _compute_report(single_results, pair_results, asr_results or None)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Voice Check: {config.speaker}")
    print(f"{'=' * 60}")
    passed = "✅ PASS" if report.overall_pass else "❌ FAIL"
    print(f"  Result:           {passed}")
    if report.avg_wer is not None:
        print(f"  Avg WER:          {report.avg_wer:.0%}")
        print(f"  Intelligibility:  {report.intelligibility_rate:.0%}")
    if report.speaker_match_rate is not None:
        print(f"  Speaker match:    {report.speaker_match_rate:.0%}")
        print(f"  Naturalness:      {report.naturalness_rate:.0%}")
    if report.avg_similarity is not None:
        print(f"  Avg similarity:   {report.avg_similarity:.1f}/10")
    if report.top_issues:
        print(f"  Issues:           {', '.join(report.top_issues)}")
    print(f"{'=' * 60}\n")

    return CheckResult(
        asr_results=asr_results,
        single_results=single_results,
        pair_results=pair_results,
        report=report,
    )
