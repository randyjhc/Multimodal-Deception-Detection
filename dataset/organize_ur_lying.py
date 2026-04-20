#!/usr/bin/env python3
"""Organize UR_LYING dataset into openface_original/, clips_raw/, and clips_processed/ with train/test splits.

Split assignments are read from openface_bk/ (the gold source) and written to four
text files under splits/:

    splits/
      train_deceptive.txt  ← one canonical trial key per line
      train_truthful.txt
      test_deceptive.txt
      test_truthful.txt

organize_ur_lying.py then reads those lists to drive all output:

    openface_original/
    ├── Train/
    │   ├── Deceptive/   ← per-trial OpenFace feature CSVs (W-B trials)
    │   └── Truthful/    ← per-trial OpenFace feature CSVs (W-T trials)
    └── Test/  (same layout)

    clips_raw/
    ├── Train/
    │   ├── Deceptive/   → symlinks to raw W-B videos in v1.0/raw_data/
    │   └── Truthful/    → symlinks to raw W-T videos in v1.0/raw_data/
    └── Test/  (same layout)

    clips_processed/
    ├── Train/
    │   ├── Deceptive/   → symlinks to merged videos in v1.0/processed_data/merged_videos/
    │   └── Truthful/
    └── Test/  (same layout)

Classification: W-B (Bluff) → Deceptive, W-T (Truth) → Truthful.
Interrogator (I-file) rows/videos are excluded.
~21 trials have no processed/merged video (raw video exists but was not merged);
these will appear in openface_original/ and clips_raw/ but not in clips_processed/.
"""

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent / "UR_LYING_Deception_Dataset"
PROCESSED_DIR = BASE_DIR / "v1.0" / "processed_data" / "merged_videos"
RAW_DIR = BASE_DIR / "v1.0" / "raw_data"
OPENFACE_SRC = BASE_DIR / "v1.0" / "processed_data" / "openface_features"

OPENFACE_BK_DIR = BASE_DIR / "openface_original"
SPLITS_LIST_DIR = BASE_DIR / "splits"

OPENFACE_DIR = BASE_DIR / "openface_original"
RAW_CLIPS_DIR = BASE_DIR / "clips_raw"
PROCESSED_CLIPS_DIR = BASE_DIR / "clips_processed"

CONDITIONS = ["commanded_low_stakes", "commanded_med_stakes", "voluntary_med_stakes"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _openface_key(stem: str) -> str:
    """Canonical trial key from an OpenFace filename stem (strips -W-B/T-... suffix)."""
    return stem.split("-W-B-")[0].split("-W-T-")[0]


def get_processed_file_id(stem: str, condition: str) -> str:
    """Extract the canonical trial key from a processed video stem.

    commanded_low_stakes uses HH-MM-SS-ms naming; the key is the last 3 segments
    (MM-SS-ms). All other conditions use the full timestamp stem as the key.
    """
    if condition == "commanded_low_stakes":
        parts = stem.split("-")
        return "-".join(parts[1:])  # drop leading HH segment
    return stem


def _extract_trial_id(filename: str) -> str | None:
    """Extract trial_id = everything before the -I-/-W- role marker."""
    m = re.search(r"-(I|W)-(B|T)-", filename)
    return filename[: m.start()] if m else None


# ---------------------------------------------------------------------------
# Step 1: dump split key lists from openface_bk
# ---------------------------------------------------------------------------

_LIST_NAMES: dict[tuple[str, str], str] = {
    ("Train", "Deceptive"): "train_deceptive.txt",
    ("Train", "Truthful"): "train_truthful.txt",
    ("Test", "Deceptive"): "test_deceptive.txt",
    ("Test", "Truthful"): "test_truthful.txt",
}


def dump_split_lists() -> None:
    """Write 4 text files to splits/ with one canonical key per line, derived from openface_original."""
    SPLITS_LIST_DIR.mkdir(parents=True, exist_ok=True)
    for (split_name, cls), fname in _LIST_NAMES.items():
        src_dir = OPENFACE_BK_DIR / split_name / cls
        if not src_dir.exists():
            raise FileNotFoundError(f"openface_bk source dir not found: {src_dir}")
        keys = sorted(_openface_key(p.stem) for p in src_dir.glob("*.csv"))
        out = SPLITS_LIST_DIR / fname
        out.write_text("\n".join(keys) + "\n")
        print(f"  Wrote {len(keys)} keys → {out.name}")


# ---------------------------------------------------------------------------
# Step 2: load split from lists and build path lookups
# ---------------------------------------------------------------------------


def _load_split_from_lists() -> tuple[dict, dict, dict]:
    """Read the 4 split text files and build path dicts.

    Returns:
        split      : {cls: {"train": set((condition, trial_id)), "test": {...}}}
        proc_paths : {(condition, trial_id): Path}  — processed/merged video
        raw_paths  : {(condition, trial_id): Path}  — raw W-file
    """
    # Build key → condition from raw W-files
    key_to_condition: dict[str, str] = {}
    for condition in CONDITIONS:
        for f in (RAW_DIR / condition / "videos").glob("*.mp4"):
            m = re.search(r"-W-(B|T)-", f.name)
            if m:
                trial_id = f.name[: f.name.index(f"-W-{m.group(1)}-")]
                key_to_condition[trial_id] = condition

    # Build key → processed clip path
    key_to_proc: dict[str, Path] = {}
    for condition in CONDITIONS:
        for video in (PROCESSED_DIR / condition).rglob("*.mp4"):
            key = get_processed_file_id(video.stem, condition)
            key_to_proc[key] = video

    # Build key → raw W-file path
    key_to_raw: dict[str, Path] = {}
    for condition in CONDITIONS:
        for f in (RAW_DIR / condition / "videos").glob("*.mp4"):
            m = re.search(r"-W-(B|T)-", f.name)
            if m:
                trial_id = f.name[: f.name.index(f"-W-{m.group(1)}-")]
                key_to_raw[trial_id] = f

    # Read split lists
    split: dict[str, dict[str, set]] = {
        "Deceptive": {"train": set(), "test": set()},
        "Truthful": {"train": set(), "test": set()},
    }
    proc_paths: dict[tuple[str, str], Path] = {}
    raw_paths: dict[tuple[str, str], Path] = {}
    missing_raw = 0
    missing_proc = 0

    for (split_name, cls), fname in _LIST_NAMES.items():
        keys = (SPLITS_LIST_DIR / fname).read_text().splitlines()
        for key in keys:
            key = key.strip()
            if not key:
                continue
            if key not in key_to_condition:
                print(f"  WARNING: key {key!r} has no matching raw W-file — skipping")
                missing_raw += 1
                continue
            condition = key_to_condition[key]
            split[cls][split_name.lower()].add((condition, key))
            if key in key_to_proc:
                proc_paths[(condition, key)] = key_to_proc[key]
            else:
                missing_proc += 1
            if key in key_to_raw:
                raw_paths[(condition, key)] = key_to_raw[key]

    if missing_proc:
        print(
            f"  Note: {missing_proc} trials have no processed/merged video (expected ~21)"
        )

    return split, proc_paths, raw_paths


# ---------------------------------------------------------------------------
# Write OpenFace CSVs
# ---------------------------------------------------------------------------


def _write_openface_csvs(split: dict) -> None:
    """Write per-trial OpenFace CSVs to openface/{Train|Test}/{Deceptive|Truthful}/."""
    lookup: dict[tuple[str, str], tuple[str, str]] = {}
    for cls in ("Deceptive", "Truthful"):
        for split_name in ("train", "test"):
            for cond, tid in split[cls][split_name]:
                lookup[(cond, tid)] = (split_name.capitalize(), cls)

    csv_paths: list[tuple[Path, str]] = []
    for condition in CONDITIONS:
        p = OPENFACE_SRC / f"{condition}_openface.csv.bz2"
        if not p.exists():
            p = OPENFACE_SRC / f"{condition}_openface.csv"
        csv_paths.append((p, condition))

    trial_buffers: dict[tuple[str, str, str], list[pd.DataFrame]] = {}
    for csv_path, condition in csv_paths:
        print(f"  Streaming {condition}...")
        for chunk in pd.read_csv(csv_path, chunksize=50_000):
            w_mask = chunk["filename"].str.contains(r"-W-[BT]-", regex=True)
            chunk = chunk[w_mask].copy()
            chunk["_tid"] = chunk["filename"].apply(_extract_trial_id)
            for tid, grp in chunk.groupby("_tid"):
                key = (condition, tid)
                if key not in lookup:
                    continue
                split_name, cls = lookup[key]
                # e.g. "36-04-870-W-B-user98_openface.csv" → "36-04-870-W-B-user98.csv"
                raw_fname = grp["filename"].iloc[0]
                out_name = raw_fname.replace("_openface.csv", ".csv")
                buf_key = (split_name, cls, out_name)
                trial_buffers.setdefault(buf_key, []).append(
                    grp.drop(columns=["_tid", "filename"])
                )

    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for (split_name, cls, out_name), frames in trial_buffers.items():
        out_path = OPENFACE_DIR / split_name / cls / out_name
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
        counts[(split_name, cls)] += 1

    for (split_name, cls), n in sorted(counts.items()):
        print(f"  Wrote {n} trial CSVs to openface_original/{split_name}/{cls}/")


# ---------------------------------------------------------------------------
# Create clip symlinks
# ---------------------------------------------------------------------------


def _create_clip_symlinks(
    split: dict,
    proc_paths: dict[tuple[str, str], Path],
    raw_paths: dict[tuple[str, str], Path],
) -> None:
    """Create symlinks in processed_clips/ and raw_clips/ using direct path lookups."""
    proc_count: defaultdict[tuple[str, str], int] = defaultdict(int)
    raw_count: defaultdict[tuple[str, str], int] = defaultdict(int)
    proc_skip = 0

    for cls in ("Deceptive", "Truthful"):
        for split_name_lower in ("train", "test"):
            split_name = split_name_lower.capitalize()
            for condition, trial_id in split[cls][split_name_lower]:
                key = (condition, trial_id)

                # Processed clip symlink
                if key in proc_paths:
                    src = proc_paths[key]
                    dst = PROCESSED_CLIPS_DIR / split_name / cls / src.name
                    if not dst.exists():
                        dst.symlink_to(src.resolve())
                    proc_count[(split_name, cls)] += 1
                else:
                    proc_skip += 1

                # Raw clip symlink
                if key in raw_paths:
                    src = raw_paths[key]
                    dst = RAW_CLIPS_DIR / split_name / cls / src.name
                    if not dst.exists():
                        dst.symlink_to(src.resolve())
                    raw_count[(split_name, cls)] += 1

    for (split_name, cls), n in sorted(proc_count.items()):
        print(f"  clips_processed/{split_name}/{cls}/: {n} symlinks")
    for (split_name, cls), n in sorted(raw_count.items()):
        print(f"  clips_raw/{split_name}/{cls}/: {n} symlinks")
    if proc_skip:
        print(f"  Skipped {proc_skip} trials with no processed clip")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Organize dataset, generating split list files from openface_bk if missing."""
    # Generate split lists if they don't exist yet
    if not all((SPLITS_LIST_DIR / fname).exists() for fname in _LIST_NAMES.values()):
        print("Generating split lists from openface_bk...")
        dump_split_lists()

    # Create output directories
    for split_name in ("Train", "Test"):
        for category in ("Deceptive", "Truthful"):
            (OPENFACE_DIR / split_name / category).mkdir(parents=True, exist_ok=True)
            (RAW_CLIPS_DIR / split_name / category).mkdir(parents=True, exist_ok=True)
            (PROCESSED_CLIPS_DIR / split_name / category).mkdir(
                parents=True, exist_ok=True
            )

    print("Step 1: Loading split and building path lookups...")
    split, proc_paths, raw_paths = _load_split_from_lists()
    for cls in ("Deceptive", "Truthful"):
        print(
            f"  {cls}: {len(split[cls]['train'])} train, {len(split[cls]['test'])} test trials"
        )

    print("\nStep 2: Writing OpenFace CSVs...")
    _write_openface_csvs(split)

    print("\nStep 3: Creating clip symlinks...")
    _create_clip_symlinks(split, proc_paths, raw_paths)

    print("\nDone.")


if __name__ == "__main__":
    main()
