"""Transcription length statistics for whisper_raw .txt files.

Reports avg/min/max word count and character count, broken down by
split (Train/Test) and class (Deceptive/Truthful).

Usage::

    uv run python dataset/whisper_stats.py
    uv run python dataset/whisper_stats.py --root dataset/UR_LYING_Deception_Dataset/whisper_raw
"""

import argparse
from pathlib import Path
from typing import List


DEFAULT_ROOT = "dataset/UR_LYING_Deception_Dataset/whisper_raw"


def _stats(values: List[int]) -> str:
    if not values:
        return "n=0"
    avg = sum(values) / len(values)
    return f"n={len(values)}  avg={avg:.1f}  min={min(values)}  max={max(values)}"


def compute_and_print(root: str) -> None:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"whisper_raw root not found: {root_path}")

    for split in ("Train", "Test"):
        split_words: List[int] = []
        split_chars: List[int] = []
        print(f"\n=== {split} ===")
        for label in ("Deceptive", "Truthful"):
            label_dir = root_path / split / label
            if not label_dir.exists():
                continue
            words: List[int] = []
            chars: List[int] = []
            for txt in sorted(label_dir.glob("*.txt")):
                text = txt.read_text(encoding="utf-8").strip()
                words.append(len(text.split()))
                chars.append(len(text))
            print(f"  {label}")
            print(f"    words: {_stats(words)}")
            print(f"    chars: {_stats(chars)}")
            split_words.extend(words)
            split_chars.extend(chars)
        print(f"  {split} overall")
        print(f"    words: {_stats(split_words)}")
        print(f"    chars: {_stats(split_chars)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=DEFAULT_ROOT, help="Path to whisper_raw directory"
    )
    args = parser.parse_args()
    compute_and_print(args.root)
