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

# ==== Dataset-specific key extraction for UR-LYING ====
# OpenFace stems:   "MM-SS-MSS-W-B-userNN"  or  "2018-..._HH-MM-SS-MSS-W-B-name"
# OpenSMILE stems:  "HH-MM-SS-MSS"          or  "2018-..._HH-MM-SS-MSS"
# Canonical key:    "MM-SS-MSS"             or  "2018-..._HH-MM-SS-MSS"


def openface_ur_lying_key(stem):
    """Strip the '-W-B-...' suffix from a UR-LYING OpenFace filename stem."""
    return stem.split("-W-B-")[0].split("-W-T-")[0]


def opensmile_ur_lying_key(stem):
    """Drop the leading HH- component from old-format UR-LYING OpenSMILE stems."""
    if "_" in stem:
        return stem
    return "-".join(stem.split("-")[1:])


# ==== Shared stem-map builder ====


def _build_stem_map(
    split_dir: Path,
    key_fn: Callable[[str], str],
    glob="*.csv",
):
    """Return {canonical_key: (file_path, label)} for all files matching."""
    stem_map: Dict[str, Tuple[Path, int]] = {}
    for label_name, label in [("Truthful", 0), ("Deceptive", 1)]:
        label_dir = split_dir / label_name
        if label_dir.exists():
            for path in sorted(label_dir.glob(glob)):
                key = key_fn(path.stem)
                stem_map[key] = (path, label)
    return stem_map


# ==== Audio + Visual + Text (AVT) dataset ====


class MultimodalDatasetAVT(Dataset):
    """Dataset pairing OpenFace (visual), OpenSMILE (audio), and
    Whisper transcription (text) features.

    Returns (visual_seq, audio_seq, text_seq, label)
      visual_seq : (T_v, 18)
      audio_seq  : (T_a, 9)
      text_seq   : (T_t, 768)
      label      : 0 (Truthful) or 1 (Deceptive)
    """

    def __init__(
        self,
        openface_root: Optional[str],
        opensmile_root: Optional[str],
        whisper_root: Optional[str],
        split="Train",
        visual_feature_cols: Optional[List[str]] = None,
        audio_feature_cols: Optional[List[str]] = None,
        min_confidence=0.5,
        visual_key_fn: Optional[Callable[[str], str]] = None,
        audio_key_fn: Optional[Callable[[str], str]] = None,
        text_key_fn: Optional[Callable[[str], str]] = None,
        audio_subsample_k=1,
        visual_subsample_k=1,
        audio_motion_method: Literal["none", "feature_diff"] = "none",
        audio_motion_low=0.0,
        audio_motion_high=float("inf"),
        visual_motion_method: Literal["none", "feature_diff"] = "none",
        visual_motion_low=0.0,
        visual_motion_high=float("inf"),
        roberta_model="roberta-base",
        roberta_max_length=512,
    ) -> None:
        self.visual_feature_cols = visual_feature_cols or VISUAL_FEATURE_COLS
        self.audio_feature_cols = audio_feature_cols or AUDIO_FEATURE_COLS

        _vkey = visual_key_fn or (lambda s: s)
        _akey = audio_key_fn or (lambda s: s)
        _tkey = text_key_fn or (lambda s: s)

        # Visual loader
        if openface_root is not None:
            self._visual_loader: Optional[OpenFaceDataset] = OpenFaceDataset(
                openface_root,
                split,
                visual_feature_cols,
                min_confidence,
                visual_subsample_k,
                visual_motion_method,
                visual_motion_low,
                visual_motion_high,
            )
            self._visual_path_to_idx: Optional[Dict[Path, int]] = {
                p: i for i, (p, _) in enumerate(self._visual_loader.samples)
            }
        else:
            self._visual_loader = None
            self._visual_path_to_idx = None

        # Audio loader
        if opensmile_root is not None:
            self._audio_loader: Optional[OpenSmileDataset] = OpenSmileDataset(
                opensmile_root,
                split,
                audio_feature_cols,
                audio_subsample_k,
                audio_motion_method,
                audio_motion_low,
                audio_motion_high,
            )
            self._audio_path_to_idx: Optional[Dict[Path, int]] = {
                p: i for i, (p, _) in enumerate(self._audio_loader.samples)
            }
        else:
            self._audio_loader = None
            self._audio_path_to_idx = None

        # Text loader
        if whisper_root is not None:
            self._text_loader: Optional[WhisperDataset] = WhisperDataset(
                whisper_root,
                split,
                model_name=roberta_model,
                max_length=roberta_max_length,
            )
            self._text_path_to_idx: Optional[Dict[Path, int]] = {
                p: i for i, (p, _) in enumerate(self._text_loader.samples)
            }
        else:
            self._text_loader = None
            self._text_path_to_idx = None

        # Build canonical_key into (path, label) maps
        visual_map = (
            _build_stem_map(Path(openface_root) / split, _vkey)
            if openface_root is not None
            else {}
        )
        audio_map = (
            _build_stem_map(Path(opensmile_root) / split, _akey)
            if opensmile_root is not None
            else {}
        )
        text_map = (
            _build_stem_map(Path(whisper_root) / split, _tkey, glob="*.txt")
            if whisper_root is not None
            else {}
        )

        # Warn about unmatched keys across active modalities
        active_maps = []
        if openface_root is not None:
            active_maps.append(("OpenFace", visual_map))
        if opensmile_root is not None:
            active_maps.append(("OpenSMILE", audio_map))
        if whisper_root is not None:
            active_maps.append(("Whisper", text_map))
        all_keys = set().union(*(m for _, m in active_maps))
        for name, mod_map in active_maps:
            missing = all_keys - set(mod_map)
            if missing:
                warnings.warn(
                    f"[MultimodalDatasetAVT] {len(missing)} keys missing from "
                    f"{name} in split='{split}': {sorted(missing)}"
                )

        # Inner join over enabled modalities
        nonempty_maps = [m for m in [visual_map, audio_map, text_map] if m]
        if not nonempty_maps:
            raise ValueError(
                "At least one of openface_root, opensmile_root, whisper_root must be set."
            )
        common_keys_set = set(nonempty_maps[0])
        for m in nonempty_maps[1:]:
            common_keys_set &= set(m)
        common_keys = sorted(common_keys_set)

        label_map = nonempty_maps[0]
        self.samples = [
            (
                visual_map[k][0] if openface_root is not None else None,
                audio_map[k][0] if opensmile_root is not None else None,
                text_map[k][0] if whisper_root is not None else None,
                label_map[k][1],
            )
            for k in common_keys
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        visual_path, audio_path, text_path, label = self.samples[idx]

        visual_seq = None
        if (
            visual_path is not None
            and self._visual_loader is not None
            and self._visual_path_to_idx is not None
        ):
            visual_seq, _ = self._visual_loader[self._visual_path_to_idx[visual_path]]

        audio_seq = None
        if (
            audio_path is not None
            and self._audio_loader is not None
            and self._audio_path_to_idx is not None
        ):
            audio_seq, _ = self._audio_loader[self._audio_path_to_idx[audio_path]]

        text_seq = None
        if (
            text_path is not None
            and self._text_loader is not None
            and self._text_path_to_idx is not None
        ):
            text_seq, _ = self._text_loader[self._text_path_to_idx[text_path]]

        return visual_seq, audio_seq, text_seq, label

    @property
    def visual_d_in(self):
        return (
            len(self.visual_feature_cols) if self._visual_loader is not None else None
        )

    @property
    def audio_d_in(self):
        return len(self.audio_feature_cols) if self._audio_loader is not None else None

    @property
    def text_d_in(self):
        return self._text_loader.d_in if self._text_loader is not None else None


def _avt_collate(batch):
    """
    Pad variable-length sequences into an AVT batch.
    Returns:
        visual_x       : (B, T_v_max, 18)
        visual_lengths : (B,)
        audio_x        : (B, T_a_max, 9)
        audio_lengths  : (B,)
        text_x         : (B, T_t_max, 768)
        text_lengths   : (B,)
        y              : (B,) float labels 0/1
    """
    visual_seqs, audio_seqs, text_seqs, ys = zip(*batch)
    if visual_seqs[0] is not None:
        visual_lengths = torch.tensor(
            [v.shape[0] for v in visual_seqs], dtype=torch.long
        )
        visual_x = pad_sequence(list(visual_seqs), batch_first=True)
    else:
        visual_x = None
        visual_lengths = None
    if audio_seqs[0] is not None:
        audio_lengths: Optional[torch.Tensor] = torch.tensor(
            [a.shape[0] for a in audio_seqs], dtype=torch.long
        )
        audio_x = pad_sequence(list(audio_seqs), batch_first=True)
    else:
        audio_x = None
        audio_lengths = None
    if text_seqs[0] is not None:
        text_lengths: Optional[torch.Tensor] = torch.tensor(
            [t.shape[0] for t in text_seqs], dtype=torch.long
        )
        text_x = pad_sequence(list(text_seqs), batch_first=True)
    else:
        text_x = None
        text_lengths = None
    y = torch.tensor(ys, dtype=torch.float32)
    return visual_x, visual_lengths, audio_x, audio_lengths, text_x, text_lengths, y


def make_avt_loaders(
    openface_root: Optional[str],
    opensmile_root: Optional[str],
    whisper_root: Optional[str],
    val_frac=0.2,
    batch_size=16,
    num_workers=0,
    seed=42,
    visual_feature_cols: Optional[List[str]] = None,
    audio_feature_cols: Optional[List[str]] = None,
    min_confidence=0.5,
    visual_key_fn: Optional[Callable[[str], str]] = None,
    audio_key_fn: Optional[Callable[[str], str]] = None,
    text_key_fn: Optional[Callable[[str], str]] = None,
    audio_subsample_k=1,
    visual_subsample_k=1,
    audio_motion_method: Literal["none", "feature_diff"] = "none",
    audio_motion_low=0.0,
    audio_motion_high=float("inf"),
    visual_motion_method: Literal["none", "feature_diff"] = "none",
    visual_motion_low=0.0,
    visual_motion_high=float("inf"),
    roberta_model="roberta-base",
    roberta_max_length=512,
):
    """
    Build train, val, and test DataLoaders for any subset of modalities.
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
