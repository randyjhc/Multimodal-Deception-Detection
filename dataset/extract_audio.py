#!/usr/bin/env python3
"""Extract pure audio (16 kHz mono WAV) from UR_LYING video clips.

For each .mp4 in clips_root/{Train,Test}/{Deceptive,Truthful}/, this script
extracts audio to audio_root/{Train,Test}/{Deceptive,Truthful}/{key}.wav
where key = openface_ur_lying_key(stem).

The canonical key matches the format used by opensmile_ur_lying_key, so the
resulting WAV files align with OpenSMILE CSVs for multimodal pairing.

Usage:
    uv run python dataset/extract_audio.py
    uv run python dataset/extract_audio.py --clips-root /alt/clips_raw --audio-root /alt/audio_raw
    uv run python dataset/extract_audio.py --no-skip-existing
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dataset.multimodal_dataset import openface_ur_lying_key

BASE_DIR = Path(__file__).parent / "UR_LYING_Deception_Dataset"
DEFAULT_CLIPS_ROOT = BASE_DIR / "clips_raw"
DEFAULT_AUDIO_ROOT = BASE_DIR / "audio_raw"

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
# Extraction
# ---------------------------------------------------------------------------


def extract_audio(
    clips_root: Path,
    audio_root: Path,
    skip_existing: bool = True,
) -> None:
    """Extract audio WAV files from all MP4 clips under clips_root.

    Output files are written to audio_root/{split}/{cls}/{key}.wav where
    key = openface_ur_lying_key(mp4_stem).  This canonical key matches the
    format produced by opensmile_ur_lying_key on OpenSMILE CSV stems, so the
    resulting WAV files are aligned for multimodal pairing.

    Args:
        clips_root:     Root directory containing {Train,Test}/{Deceptive,Truthful}/*.mp4
        audio_root:     Output root; mirrors the same split/class structure.
        skip_existing:  Skip conversion when the output .wav already exists.
    """
    total = 0
    skipped = 0
    errors = 0

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = clips_root / split / cls
            out_dir = audio_root / split / cls
            if not src_dir.exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)

            for video in sorted(src_dir.glob("*.mp4")):
                key = openface_ur_lying_key(video.stem)
                wav_path = out_dir / f"{key}.wav"

                if skip_existing and wav_path.exists():
                    skipped += 1
                    continue

                print(f"  [{split}/{cls}] {video.name} → {key}.wav")

                try:
                    _to_wav(video, wav_path)
                    total += 1
                except Exception as exc:
                    print(f"    ERROR: {exc}", file=sys.stderr)
                    errors += 1

    print(f"\nDone: {total} extracted, {skipped} skipped, {errors} errors.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 16 kHz mono WAV audio from UR_LYING video clips."
    )
    parser.add_argument(
        "--clips-root",
        type=Path,
        default=DEFAULT_CLIPS_ROOT,
        help=f"Root of clips_raw/ directory (default: {DEFAULT_CLIPS_ROOT})",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help=f"Output root for audio_raw/ directory (default: {DEFAULT_AUDIO_ROOT})",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-extract even when the output .wav already exists",
    )
    args = parser.parse_args()

    print(f"clips_root : {args.clips_root}")
    print(f"audio_root : {args.audio_root}")
    print(f"skip_existing: {not args.no_skip_existing}\n")

    extract_audio(
        clips_root=args.clips_root,
        audio_root=args.audio_root,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
