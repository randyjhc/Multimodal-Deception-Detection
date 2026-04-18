#!/usr/bin/env python3
"""Check cross-directory consistency for UR_LYING dataset."""

import argparse
import sys
from pathlib import Path
from typing import Callable

from multimodal_dataset import openface_ur_lying_key, opensmile_ur_lying_key

DEFAULT_ROOT = "dataset/UR_LYING_Deception_Dataset"
DEFAULT_OPENFACE_ROOT = f"{DEFAULT_ROOT}/openface"
DEFAULT_OPENSMILE_ROOT = f"{DEFAULT_ROOT}/opensmile"
DEFAULT_RAW_CLIPS_ROOT = f"{DEFAULT_ROOT}/raw_clips"
DEFAULT_PROCESSED_CLIPS_ROOT = f"{DEFAULT_ROOT}/processed_clips"
DEFAULT_WHISPER_RAW_ROOT = f"{DEFAULT_ROOT}/whisper_raw"
DEFAULT_WHISPER_PROCESSED_ROOT = f"{DEFAULT_ROOT}/whisper_processed"


# ---------------------------------------------------------------------------
# Cross-directory consistency check
# ---------------------------------------------------------------------------


def _collect_keys(
    directory: Path, key_fn: Callable[[str], str], extensions: tuple[str, ...]
) -> set[str]:
    """Return canonical keys for all matching files in a directory."""
    keys: set[str] = set()
    for ext in extensions:
        for f in directory.glob(ext):
            keys.add(key_fn(f.stem))
    return keys


def check_consistency(
    openface_root: Path,
    opensmile_root: Path,
    raw_clips_root: Path,
    processed_clips_root: Path,
    whisper_raw_root: Path,
    whisper_processed_root: Path,
) -> bool:
    """Check that all six directories contain the same trials per (split, class).

    Each directory uses a different filename convention, but all reduce to the
    same canonical trial key via the appropriate key function:
      - openface / raw_clips / whisper_raw / whisper_processed:
            openface_ur_lying_key  (strips -W-B/T-userNN suffix; no-op for already-canonical stems)
      - opensmile / processed_clips:
            opensmile_ur_lying_key (drops leading HH- segment)

    Returns True if all checks pass, False if any mismatch is found.
    """
    dir_configs: dict[str, tuple[Path, Callable[[str], str], tuple[str, ...]]] = {
        "openface": (openface_root, openface_ur_lying_key, ("*.csv",)),
        "opensmile": (opensmile_root, opensmile_ur_lying_key, ("*.csv",)),
        "raw_clips": (raw_clips_root, openface_ur_lying_key, ("*.mp4",)),
        "processed_clips": (processed_clips_root, opensmile_ur_lying_key, ("*.mp4",)),
        "whisper_raw": (whisper_raw_root, openface_ur_lying_key, ("*.txt",)),
        "whisper_processed": (
            whisper_processed_root,
            openface_ur_lying_key,
            ("*.txt",),
        ),
    }

    all_ok = True

    for split in ("Train", "Test"):
        for cls in ("Deceptive", "Truthful"):
            key_sets: dict[str, set[str]] = {}
            for name, (root, key_fn, exts) in dir_configs.items():
                subdir = root / split / cls
                if not subdir.exists():
                    print(f"  MISSING dir: {subdir}")
                    all_ok = False
                    key_sets[name] = set()
                else:
                    key_sets[name] = _collect_keys(subdir, key_fn, exts)

            counts = {n: len(k) for n, k in key_sets.items()}
            count_str = "  ".join(f"{n}={c}" for n, c in counts.items())
            counts_match = len(set(counts.values())) == 1

            status = "OK" if counts_match else "COUNT MISMATCH"
            print(f"  {split}/{cls}: {count_str}  {status}")

            if not counts_match:
                all_ok = False

            # Report key-level mismatches between each pair of directories
            names = list(key_sets.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    n1, n2 = names[i], names[j]
                    only_1 = sorted(key_sets[n1] - key_sets[n2])
                    only_2 = sorted(key_sets[n2] - key_sets[n1])
                    if only_1:
                        preview = only_1[:5]
                        ellipsis = "..." if len(only_1) > 5 else ""
                        print(
                            f"    In {n1} not {n2} ({len(only_1)}): {preview}{ellipsis}"
                        )
                        all_ok = False
                    if only_2:
                        preview = only_2[:5]
                        ellipsis = "..." if len(only_2) > 5 else ""
                        print(
                            f"    In {n2} not {n1} ({len(only_2)}): {preview}{ellipsis}"
                        )
                        all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openface-root",
        default=DEFAULT_OPENFACE_ROOT,
        help="Path to OpenFace features root",
    )
    parser.add_argument(
        "--opensmile-root",
        default=DEFAULT_OPENSMILE_ROOT,
        help="Path to OpenSMILE features root",
    )
    parser.add_argument(
        "--raw-clips-root",
        default=DEFAULT_RAW_CLIPS_ROOT,
        help="Path to raw_clips root",
    )
    parser.add_argument(
        "--processed-clips-root",
        default=DEFAULT_PROCESSED_CLIPS_ROOT,
        help="Path to processed_clips root",
    )
    parser.add_argument(
        "--whisper-raw-root",
        default=DEFAULT_WHISPER_RAW_ROOT,
        help="Path to whisper_raw root",
    )
    parser.add_argument(
        "--whisper-processed-root",
        default=DEFAULT_WHISPER_PROCESSED_ROOT,
        help="Path to whisper_processed root",
    )
    args = parser.parse_args()

    print(f"OpenFace root:          {args.openface_root}")
    print(f"OpenSMILE root:         {args.opensmile_root}")
    print(f"raw_clips root:         {args.raw_clips_root}")
    print(f"processed_clips root:   {args.processed_clips_root}")
    print(f"whisper_raw root:       {args.whisper_raw_root}")
    print(f"whisper_processed root: {args.whisper_processed_root}\n")

    print("--- Cross-directory consistency ---")
    consistent = check_consistency(
        openface_root=Path(args.openface_root),
        opensmile_root=Path(args.opensmile_root),
        raw_clips_root=Path(args.raw_clips_root),
        processed_clips_root=Path(args.processed_clips_root),
        whisper_raw_root=Path(args.whisper_raw_root),
        whisper_processed_root=Path(args.whisper_processed_root),
    )
    if consistent:
        print("All directories are consistent.")
    else:
        print("WARNING: directories are NOT fully consistent (see above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
