#!/usr/bin/env python3
"""Transcribe UR_LYING video clips to text using Whisper.

For each clip in raw_clips/ and processed_clips/, this script:
  1. Converts the video to a 16 kHz mono .wav file using ffmpeg.
  2. Transcribes the .wav with OpenAI Whisper.
  3. Saves {key}.wav and {key}.txt to the output directory.

Outputs are separated by source:
  whisper_raw/       ← from raw_clips/
  whisper_processed/ ← from processed_clips/

Each output mirrors the Train/Test × Deceptive/Truthful split layout.

Usage:
    uv run --group dataset python dataset/transcribe_clips.py
    uv run --group dataset python dataset/transcribe_clips.py --model small --skip-existing
"""

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import whisper

from dataset.multimodal_dataset import openface_ur_lying_key, opensmile_ur_lying_key

BASE_DIR = Path(__file__).parent / "UR_LYING_Deception_Dataset"

DEFAULT_RAW_CLIPS = BASE_DIR / "raw_clips"
DEFAULT_PROCESSED_CLIPS = BASE_DIR / "processed_clips"
DEFAULT_WHISPER_RAW = BASE_DIR / "whisper_raw"
DEFAULT_WHISPER_PROCESSED = BASE_DIR / "whisper_processed"

SPLITS = ("Train", "Test")
CLASSES = ("Deceptive", "Truthful")


# ---------------------------------------------------------------------------
# ffmpeg helper
# ---------------------------------------------------------------------------


def _to_wav(video_path: Path, wav_path: Path) -> None:
    """Convert a video file to 16 kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-vn",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path.name}:\n{result.stderr.strip()}"
        )


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def _transcribe_source(
    clips_root: Path,
    output_root: Path,
    key_fn: Callable[[str], str],
    model: whisper.Whisper,
    language: str,
    skip_existing: bool,
    limit: int | None = None,
) -> None:
    """Transcribe all .mp4 files under clips_root into output_root."""
    total = 0
    skipped = 0
    errors = 0

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = clips_root / split / cls
            out_dir = output_root / split / cls
            if not src_dir.exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)

            videos = sorted(src_dir.glob("*.mp4"))
            for video in videos:
                if limit is not None and total >= limit:
                    break

                key = key_fn(video.stem)
                wav_path = out_dir / f"{key}.wav"
                txt_path = out_dir / f"{key}.txt"

                if skip_existing and txt_path.exists():
                    skipped += 1
                    continue

                print(f"  [{split}/{cls}] {video.name} → {key}")

                try:
                    _to_wav(video, wav_path)
                    result = model.transcribe(str(wav_path), language=language)
                    txt_path.write_text(result["text"].strip())
                    total += 1
                except Exception as exc:
                    print(f"    ERROR: {exc}", file=sys.stderr)
                    errors += 1

    print(f"  Done: {total} transcribed, {skipped} skipped, {errors} errors.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe UR_LYING clips with Whisper."
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language hint for Whisper (default: en)",
    )
    parser.add_argument(
        "--raw-clips-root",
        type=Path,
        default=DEFAULT_RAW_CLIPS,
        help="Path to raw_clips/ root",
    )
    parser.add_argument(
        "--processed-clips-root",
        type=Path,
        default=DEFAULT_PROCESSED_CLIPS,
        help="Path to processed_clips/ root",
    )
    parser.add_argument(
        "--whisper-raw-dir",
        type=Path,
        default=DEFAULT_WHISPER_RAW,
        help="Output directory for raw_clips transcriptions",
    )
    parser.add_argument(
        "--whisper-processed-dir",
        type=Path,
        default=DEFAULT_WHISPER_PROCESSED,
        help="Output directory for processed_clips transcriptions",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trials whose .txt file already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after transcribing N videos per source (useful for testing)",
    )
    args = parser.parse_args()

    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model)

    print(f"\n--- raw_clips → {args.whisper_raw_dir} ---")
    _transcribe_source(
        clips_root=args.raw_clips_root,
        output_root=args.whisper_raw_dir,
        key_fn=openface_ur_lying_key,
        model=model,
        language=args.language,
        skip_existing=args.skip_existing,
        limit=args.limit,
    )

    print(f"\n--- processed_clips → {args.whisper_processed_dir} ---")
    _transcribe_source(
        clips_root=args.processed_clips_root,
        output_root=args.whisper_processed_dir,
        key_fn=opensmile_ur_lying_key,
        model=model,
        language=args.language,
        skip_existing=args.skip_existing,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
