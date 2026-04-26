#!/usr/bin/env python3
"""Copy UR_LYING raw video clips from clips_raw_bk into clips_raw.

clips_raw_bk contains symlinks to the original videos in v1.0/raw_data/.
This script resolves each symlink and copies the real MP4 file to the
mirrored path under clips_raw/, preserving the split/class structure.

Usage:
    uv run python dataset/copy_clips_raw.py
    uv run python dataset/copy_clips_raw.py --no-skip-existing
    uv run python dataset/copy_clips_raw.py --src clips_raw_bk --dst clips_raw
"""

import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent / "UR_LYING_Deception_Dataset"
DEFAULT_SRC = BASE_DIR / "clips_raw_bk"
DEFAULT_DST = BASE_DIR / "clips_raw"

SPLITS = ("Train", "Test")
CLASSES = ("Deceptive", "Truthful")


def copy_clips(
    src_root: Path,
    dst_root: Path,
    skip_existing: bool = True,
) -> None:
    """Copy resolved MP4 files from src_root into dst_root.

    Args:
        src_root:      Source root containing {Train,Test}/{Deceptive,Truthful}/*.mp4
                       (may be symlinks — they are resolved before copying).
        dst_root:      Destination root; mirrors the same split/class structure.
        skip_existing: Skip copy when the destination file already exists.
    """
    total = 0
    skipped = 0
    errors = 0

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = src_root / split / cls
            dst_dir = dst_root / split / cls
            if not src_dir.exists():
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)

            for src_file in sorted(src_dir.glob("*.mp4")):
                dst_file = dst_dir / src_file.name

                if skip_existing and dst_file.exists():
                    skipped += 1
                    continue

                resolved = src_file.resolve()
                print(f"  [{split}/{cls}] {src_file.name}")

                try:
                    shutil.copy2(resolved, dst_file)
                    total += 1
                except Exception as exc:
                    print(f"    ERROR: {exc}", file=sys.stderr)
                    errors += 1

    print(f"\nDone: {total} copied, {skipped} skipped, {errors} errors.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy UR_LYING raw video clips from clips_raw_bk into clips_raw."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"Source root with symlinked MP4s (default: {DEFAULT_SRC})",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help=f"Destination root for actual copies (default: {DEFAULT_DST})",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Overwrite destination files that already exist",
    )
    args = parser.parse_args()

    print(f"src : {args.src}")
    print(f"dst : {args.dst}")
    print(f"skip_existing: {not args.no_skip_existing}\n")

    copy_clips(
        src_root=args.src,
        dst_root=args.dst,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
