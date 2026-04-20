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
from dataset.whisper_dataset import WhisperDataset

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
# Shared stem-map builder
# ---------------------------------------------------------------------------


def _build_stem_map(
    split_dir: Path,
    key_fn: Callable[[str], str],
    glob: str = "*.csv",
) -> Dict[str, Tuple[Path, int]]:
    """Return ``{canonical_key: (file_path, label)}`` for all files matching
    *glob* under ``split_dir/{Truthful,Deceptive}/``.
    """
    stem_map: Dict[str, Tuple[Path, int]] = {}
    for label_name, label in [("Truthful", 0), ("Deceptive", 1)]:
        label_dir = split_dir / label_name
        if label_dir.exists():
            for path in sorted(label_dir.glob(glob)):
                key = key_fn(path.stem)
                stem_map[key] = (path, label)
    return stem_map


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
        visual_map = _build_stem_map(Path(openface_root) / split, _vkey)
        audio_map = _build_stem_map(Path(opensmile_root) / split, _akey)

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


# ---------------------------------------------------------------------------
# Audio + Visual + Text (AVT) dataset
# ---------------------------------------------------------------------------


class MultimodalDatasetAVT(Dataset):
    """PyTorch Dataset pairing OpenFace (visual), OpenSMILE (audio), and
    Whisper transcription (text) features.

    Samples are aligned by a canonical key derived from each file's stem.
    Only clips present in **all three** directories are included; warnings are
    emitted for any unmatched keys.

    Each ``__getitem__`` returns ``(visual_seq, audio_seq, text_seq, label)``
    where:
      - ``visual_seq`` : float32 tensor ``(T_v, 48)``  — variable length
      - ``audio_seq``  : float32 tensor ``(T_a, 88)``  — variable length
      - ``text_seq``   : float32 tensor ``(T_t, 768)`` — variable length
      - ``label``      : int, 0 (Truthful) or 1 (Deceptive)
    """

    def __init__(
        self,
        openface_root: str,
        opensmile_root: str,
        whisper_root: str,
        split: str = "Train",
        visual_feature_cols: Optional[List[str]] = None,
        audio_feature_cols: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        visual_key_fn: Optional[Callable[[str], str]] = None,
        audio_key_fn: Optional[Callable[[str], str]] = None,
        text_key_fn: Optional[Callable[[str], str]] = None,
        audio_subsample_k: int = 1,
        visual_subsample_k: int = 1,
        audio_motion_method: Literal["none", "feature_diff"] = "none",
        audio_motion_low: float = 0.0,
        audio_motion_high: float = float("inf"),
        visual_motion_method: Literal["none", "feature_diff"] = "none",
        visual_motion_low: float = 0.0,
        visual_motion_high: float = float("inf"),
        roberta_model: str = "roberta-base",
        roberta_max_length: int = 512,
    ) -> None:
        self.visual_feature_cols = visual_feature_cols or VISUAL_FEATURE_COLS
        self.audio_feature_cols = audio_feature_cols or AUDIO_FEATURE_COLS

        _vkey = visual_key_fn or (lambda s: s)
        _akey = audio_key_fn or (lambda s: s)
        _tkey = text_key_fn or (lambda s: s)

        # Visual loader
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

        # Audio loader
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

        # Text loader (WhisperDataset; key_fn passed through)
        self._text_loader = WhisperDataset(
            whisper_root,
            split,
            model_name=roberta_model,
            max_length=roberta_max_length,
            key_fn=_tkey,
        )
        self._text_path_to_idx: Dict[Path, int] = {
            p: i for i, (p, _) in enumerate(self._text_loader.samples)
        }

        # Build canonical_key → (path, label) maps
        visual_map = _build_stem_map(Path(openface_root) / split, _vkey)
        audio_map = _build_stem_map(Path(opensmile_root) / split, _akey)
        text_map = _build_stem_map(
            Path(whisper_root) / split,
            _tkey,
            glob="*.txt",
        )

        # Warn about unmatched keys
        all_keys = set(visual_map) | set(audio_map) | set(text_map)
        for name, mod_map in [
            ("OpenFace", visual_map),
            ("OpenSMILE", audio_map),
            ("Whisper", text_map),
        ]:
            missing = all_keys - set(mod_map)
            if missing:
                warnings.warn(
                    f"[MultimodalDatasetAVT] {len(missing)} keys missing from "
                    f"{name} in split='{split}': {sorted(missing)}"
                )

        # Triple inner join
        common_keys = sorted(set(visual_map) & set(audio_map) & set(text_map))
        self.samples: List[Tuple[Path, Path, Path, int]] = [
            (visual_map[k][0], audio_map[k][0], text_map[k][0], visual_map[k][1])
            for k in common_keys
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        visual_path, audio_path, text_path, label = self.samples[idx]

        visual_seq, _ = self._visual_loader[self._visual_path_to_idx[visual_path]]
        audio_seq, _ = self._audio_loader[self._audio_path_to_idx[audio_path]]
        text_seq, _ = self._text_loader[self._text_path_to_idx[text_path]]

        return visual_seq, audio_seq, text_seq, label

    @property
    def visual_d_in(self) -> int:
        return len(self.visual_feature_cols)

    @property
    def audio_d_in(self) -> int:
        return len(self.audio_feature_cols)

    @property
    def text_d_in(self) -> int:
        return self._text_loader.d_in


def _avt_collate(
    batch: list,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Pad variable-length sequences into an AVT batch.

    Returns:
        visual_x       : (B, T_v_max, 48)
        visual_lengths : (B,)
        audio_x        : (B, T_a_max, 88)
        audio_lengths  : (B,)
        text_x         : (B, T_t_max, 768)
        text_lengths   : (B,)
        y              : (B,) float labels 0/1
    """
    visual_seqs, audio_seqs, text_seqs, ys = zip(*batch)
    visual_lengths = torch.tensor([v.shape[0] for v in visual_seqs], dtype=torch.long)
    audio_lengths = torch.tensor([a.shape[0] for a in audio_seqs], dtype=torch.long)
    text_lengths = torch.tensor([t.shape[0] for t in text_seqs], dtype=torch.long)
    visual_x = pad_sequence(list(visual_seqs), batch_first=True)
    audio_x = pad_sequence(list(audio_seqs), batch_first=True)
    text_x = pad_sequence(list(text_seqs), batch_first=True)
    y = torch.tensor(ys, dtype=torch.float32)
    return visual_x, visual_lengths, audio_x, audio_lengths, text_x, text_lengths, y


def make_avt_loaders(
    openface_root: str,
    opensmile_root: str,
    whisper_root: str,
    val_frac: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
    visual_feature_cols: Optional[List[str]] = None,
    audio_feature_cols: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    visual_key_fn: Optional[Callable[[str], str]] = None,
    audio_key_fn: Optional[Callable[[str], str]] = None,
    text_key_fn: Optional[Callable[[str], str]] = None,
    audio_subsample_k: int = 1,
    visual_subsample_k: int = 1,
    audio_motion_method: Literal["none", "feature_diff"] = "none",
    audio_motion_low: float = 0.0,
    audio_motion_high: float = float("inf"),
    visual_motion_method: Literal["none", "feature_diff"] = "none",
    visual_motion_low: float = 0.0,
    visual_motion_high: float = float("inf"),
    roberta_model: str = "roberta-base",
    roberta_max_length: int = 512,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    """Build train, val, and test DataLoaders with visual + audio + text features.

    Returns:
        train_loader, val_loader, test_loader, visual_d_in, audio_d_in, text_d_in
    """
    kwargs: dict = dict(
        visual_feature_cols=visual_feature_cols,
        audio_feature_cols=audio_feature_cols,
        min_confidence=min_confidence,
        visual_key_fn=visual_key_fn,
        audio_key_fn=audio_key_fn,
        text_key_fn=text_key_fn,
        audio_subsample_k=audio_subsample_k,
        visual_subsample_k=visual_subsample_k,
        audio_motion_method=audio_motion_method,
        audio_motion_low=audio_motion_low,
        audio_motion_high=audio_motion_high,
        visual_motion_method=visual_motion_method,
        visual_motion_low=visual_motion_low,
        visual_motion_high=visual_motion_high,
        roberta_model=roberta_model,
        roberta_max_length=roberta_max_length,
    )

    full_train = MultimodalDatasetAVT(
        openface_root, opensmile_root, whisper_root, "Train", **kwargs
    )
    n_val = max(1, int(len(full_train) * val_frac))
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=generator)

    test_ds = MultimodalDatasetAVT(
        openface_root, opensmile_root, whisper_root, "Test", **kwargs
    )

    loader_kwargs: dict = dict(
        batch_size=batch_size,
        collate_fn=_avt_collate,
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
        full_train.text_d_in,
    )
