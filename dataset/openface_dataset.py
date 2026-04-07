from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Feature column definitions (48 features total)
# ---------------------------------------------------------------------------

GAZE_COLS: List[str] = [
    "gaze_0_x", "gaze_0_y", "gaze_0_z",
    "gaze_1_x", "gaze_1_y", "gaze_1_z",
    "gaze_angle_x", "gaze_angle_y",
]

POSE_COLS: List[str] = [
    "pose_Tx", "pose_Ty", "pose_Tz",
    "pose_Rx", "pose_Ry", "pose_Rz",
]

AU_R_COLS: List[str] = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

AU_C_COLS: List[str] = [col.replace("_r", "_c") for col in AU_R_COLS]

DEFAULT_FEATURE_COLS: List[str] = GAZE_COLS + POSE_COLS + AU_R_COLS + AU_C_COLS  # 48


class OpenFaceDataset(Dataset):
    """PyTorch Dataset for OpenFace facial feature CSVs.

    Directory layout expected::

        root_dir/
          Train/
            Truthful/   *.csv   → label 0
            Deceptive/  *.csv   → label 1
          Test/
            Truthful/   *.csv   → label 0
            Deceptive/  *.csv   → label 1

    Each ``__getitem__`` returns ``(seq, label)`` where:
      - ``seq``   : float32 tensor of shape ``(T, D)`` — T varies per sample
      - ``label`` : int, 0 (Truthful) or 1 (Deceptive)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        feature_cols: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        subsample_k: int = 1,
    ) -> None:
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.min_confidence = min_confidence
        self.subsample_k = subsample_k

        split_dir = Path(root_dir) / split
        self.samples: List[Tuple[Path, int]] = []
        for label_name, label in [("Truthful", 0), ("Deceptive", 1)]:
            for csv_path in sorted((split_dir / label_name).glob("*.csv")):
                self.samples.append((csv_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        csv_path, label = self.samples[idx]

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Drop frames where OpenFace tracking failed or was low-confidence.
        # Fall back to all frames if every row would be filtered out.
        filtered = df[(df["success"] == 1) & (df["confidence"] >= self.min_confidence)]
        df = filtered if len(filtered) > 0 else df

        seq = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)
        
        # per-sample normalization
        mean = seq.mean(dim=0, keepdim=True)
        std = seq.std(dim=0, keepdim=True).clamp_min(1e-6)
        seq = (seq - mean) / std

        if self.subsample_k > 1:
            seq = seq[:: self.subsample_k]
        return seq, label



def _collate(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences into a batch.

    Returns:
        x_padded : (B, T_max, D)
        lengths  : (B,)  actual sequence lengths
        y        : (B,)  float labels 0/1
    """
    seqs, ys = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    x_padded = pad_sequence(seqs, batch_first=True)  # zero-padded
    y = torch.tensor(ys, dtype=torch.float32)
    return x_padded, lengths, y


def make_loaders(
    root_dir: str,
    val_frac: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
    feature_cols: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    subsample_k: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Build train, val, and test DataLoaders from the OpenFace feature directory.

    Args:
        root_dir:       Path to the ``OpenFace_features/`` directory.
        val_frac:       Fraction of Train/ to reserve for validation (default 0.2).
        batch_size:     Samples per batch.
        num_workers:    DataLoader worker processes.
        seed:           Random seed for the train/val split.
        feature_cols:   Override the default 48-column feature set.
        min_confidence: Minimum OpenFace confidence to keep a frame.

    Returns:
        train_loader, val_loader, test_loader, d_in

        Pass ``d_in`` directly to ``BiLSTMClassifier(d_in=d_in, ...)`` or
        ``run(..., d_in=d_in)``.
    """
    feature_cols = feature_cols or DEFAULT_FEATURE_COLS

    full_train = OpenFaceDataset(root_dir, "Train", feature_cols, min_confidence, subsample_k)
    n_val = max(1, int(len(full_train) * val_frac))
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=generator)

    test_ds = OpenFaceDataset(root_dir, "Test", feature_cols, min_confidence, subsample_k)

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, shuffle=True, generator=train_generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, len(feature_cols)
