import warnings
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

from dataset.openface_dataset import (
    DEFAULT_FEATURE_COLS as VISUAL_FEATURE_COLS,
    OpenFaceDataset,
)
from dataset.opensmile_dataset import AUDIO_FEATURE_COLS, OpenSmileDataset

# ---------------------------------------------------------------------------
# Dataset-specific key extraction for UR-LYING
# ---------------------------------------------------------------------------
# OpenFace stems:   "MM-SS-MSS-W-B-userNN"  or  "2018-..._HH-MM-SS-MSS-W-B-name"
# OpenSMILE stems:  "HH-MM-SS-MSS"          or  "2018-..._HH-MM-SS-MSS"
# Canonical key:    "MM-SS-MSS"             or  "2018-..._HH-MM-SS-MSS"


def openface_ur_lying_key(stem: str) -> str:
    """Strip the '-W-B-...' suffix from a UR-LYING OpenFace filename stem."""
    return stem.split("-W-B-")[0].split("-W-T-")[0]


def opensmile_ur_lying_key(stem: str) -> str:
    """Drop the leading HH- component from old-format UR-LYING OpenSMILE stems.

    New date-based stems (containing '_') are returned unchanged.
    """
    if "_" in stem:
        return stem
    # old format: HH-MM-SS-MSS  →  MM-SS-MSS
    return "-".join(stem.split("-")[1:])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MultimodalDataset(Dataset):
    """PyTorch Dataset pairing OpenFace (visual) and OpenSMILE (audio) features.

    Samples are aligned by a canonical key derived from each file's stem.
    Only clips present in **both** directories are included; a warning is
    emitted for any unmatched keys.

    Each ``__getitem__`` returns ``(visual_seq, audio_seq, label)`` where:
      - ``visual_seq`` : float32 tensor ``(T_v, 48)``  — variable length
      - ``audio_seq``  : float32 tensor ``(T_a, 88)``  — variable length
      - ``label``      : int, 0 (Truthful) or 1 (Deceptive)

    Args:
        visual_key_fn: Maps an OpenFace filename stem to a canonical match key.
                       Defaults to identity (exact stem match).
        audio_key_fn:  Maps an OpenSMILE filename stem to a canonical match key.
                       Defaults to identity (exact stem match).
    """

    def __init__(
        self,
        openface_root: str,
        opensmile_root: str,
        split: str = "Train",
        visual_feature_cols: Optional[List[str]] = None,
        audio_feature_cols: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        visual_key_fn: Optional[Callable[[str], str]] = None,
        audio_key_fn: Optional[Callable[[str], str]] = None,
        audio_subsample_k: int = 1,
        visual_subsample_k: int = 1,
        audio_motion_method: Literal["none", "feature_diff"] = "none",
        audio_motion_low: float = 0.0,
        audio_motion_high: float = float("inf"),
        visual_motion_method: Literal["none", "feature_diff"] = "none",
        visual_motion_low: float = 0.0,
        visual_motion_high: float = float("inf"),
    ) -> None:
        self.visual_feature_cols = visual_feature_cols or VISUAL_FEATURE_COLS
        self.audio_feature_cols = audio_feature_cols or AUDIO_FEATURE_COLS

        _vkey = visual_key_fn or (lambda s: s)
        _akey = audio_key_fn or (lambda s: s)

        # Visual loader — owns confidence filtering, normalization, and subsampling
        self._visual_loader = OpenFaceDataset(
            openface_root,
            split,
            visual_feature_cols,
            min_confidence,
            visual_subsample_k,
            visual_motion_method,
            visual_motion_low,
            visual_motion_high,
        )
        self._visual_path_to_idx: Dict[Path, int] = {
            p: i for i, (p, _) in enumerate(self._visual_loader.samples)
        }

        # Audio loader — owns subsampling logic
        self._audio_loader = OpenSmileDataset(
            opensmile_root,
            split,
            audio_feature_cols,
            audio_subsample_k,
            audio_motion_method,
            audio_motion_low,
            audio_motion_high,
        )
        self._audio_path_to_idx: Dict[Path, int] = {
            p: i for i, (p, _) in enumerate(self._audio_loader.samples)
        }

        # Build canonical_key → (path, label) maps for each modality
        visual_map = self._build_stem_map(Path(openface_root) / split, _vkey)
        audio_map = self._build_stem_map(Path(opensmile_root) / split, _akey)

        # Warn about unmatched keys
        visual_only = set(visual_map) - set(audio_map)
        audio_only = set(audio_map) - set(visual_map)
        if visual_only:
            warnings.warn(
                f"[MultimodalDataset] {len(visual_only)} OpenFace keys have no "
                f"matching OpenSMILE file in split='{split}': {sorted(visual_only)}"
            )
        if audio_only:
            warnings.warn(
                f"[MultimodalDataset] {len(audio_only)} OpenSMILE keys have no "
                f"matching OpenFace file in split='{split}': {sorted(audio_only)}"
            )

        # Inner join on canonical keys
        common_keys = sorted(set(visual_map) & set(audio_map))
        self.samples: List[Tuple[Path, Path, int]] = [
            (visual_map[k][0], audio_map[k][0], visual_map[k][1]) for k in common_keys
        ]

    @staticmethod
    def _build_stem_map(
        split_dir: Path, key_fn: Callable[[str], str]
    ) -> Dict[str, Tuple[Path, int]]:
        """Return {canonical_key: (csv_path, label)} for all CSVs under split_dir."""
        stem_map: Dict[str, Tuple[Path, int]] = {}
        for label_name, label in [("Truthful", 0), ("Deceptive", 1)]:
            for csv_path in sorted((split_dir / label_name).glob("*.csv")):
                key = key_fn(csv_path.stem)
                stem_map[key] = (csv_path, label)
        return stem_map

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        visual_path, audio_path, label = self.samples[idx]

        # --- Visual (OpenFace) — filtering, normalization, subsampling handled by _visual_loader ---
        visual_seq, _ = self._visual_loader[self._visual_path_to_idx[visual_path]]

        # --- Audio (OpenSMILE) — subsampling handled by _audio_loader ---
        audio_seq, _ = self._audio_loader[self._audio_path_to_idx[audio_path]]

        return visual_seq, audio_seq, label

    @property
    def visual_d_in(self) -> int:
        return len(self.visual_feature_cols)

    @property
    def audio_d_in(self) -> int:
        return len(self.audio_feature_cols)


def _multimodal_collate(
    batch: list,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences into a batch.

    Returns:
        visual_x       : (B, T_v_max, 48)
        visual_lengths : (B,)
        audio_x        : (B, T_a_max, 88)
        audio_lengths  : (B,)
        y              : (B,)  float labels 0/1
    """
    visual_seqs, audio_seqs, ys = zip(*batch)
    visual_lengths = torch.tensor([v.shape[0] for v in visual_seqs], dtype=torch.long)
    audio_lengths = torch.tensor([a.shape[0] for a in audio_seqs], dtype=torch.long)
    visual_x = pad_sequence(visual_seqs, batch_first=True)
    audio_x = pad_sequence(audio_seqs, batch_first=True)
    y = torch.tensor(ys, dtype=torch.float32)
    return visual_x, visual_lengths, audio_x, audio_lengths, y


def make_multimodal_loaders(
    openface_root: str,
    opensmile_root: str,
    val_frac: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
    visual_feature_cols: Optional[List[str]] = None,
    audio_feature_cols: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    visual_key_fn: Optional[Callable[[str], str]] = None,
    audio_key_fn: Optional[Callable[[str], str]] = None,
    audio_subsample_k: int = 1,
    visual_subsample_k: int = 1,
    audio_motion_method: Literal["none", "feature_diff"] = "none",
    audio_motion_low: float = 0.0,
    audio_motion_high: float = float("inf"),
    visual_motion_method: Literal["none", "feature_diff"] = "none",
    visual_motion_low: float = 0.0,
    visual_motion_high: float = float("inf"),
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Build train, val, and test DataLoaders with paired visual+audio features.

    Returns:
        train_loader, val_loader, test_loader, visual_d_in, audio_d_in
    """
    kwargs: dict = dict(
        visual_feature_cols=visual_feature_cols,
        audio_feature_cols=audio_feature_cols,
        min_confidence=min_confidence,
        visual_key_fn=visual_key_fn,
        audio_key_fn=audio_key_fn,
        audio_subsample_k=audio_subsample_k,
        visual_subsample_k=visual_subsample_k,
        audio_motion_method=audio_motion_method,
        audio_motion_low=audio_motion_low,
        audio_motion_high=audio_motion_high,
        visual_motion_method=visual_motion_method,
        visual_motion_low=visual_motion_low,
        visual_motion_high=visual_motion_high,
    )

    full_train = MultimodalDataset(openface_root, opensmile_root, "Train", **kwargs)
    n_val = max(1, int(len(full_train) * val_frac))
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=generator)

    test_ds = MultimodalDataset(openface_root, opensmile_root, "Test", **kwargs)

    loader_kwargs: dict = dict(
        batch_size=batch_size,
        collate_fn=_multimodal_collate,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return (
        train_loader,
        val_loader,
        test_loader,
        full_train.visual_d_in,
        full_train.audio_d_in,
    )
