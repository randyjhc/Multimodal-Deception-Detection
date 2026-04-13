from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Feature column definitions (25 per-frame features — excludes time_start/time_end)
# ---------------------------------------------------------------------------

AUDIO_FEATURE_COLS: List[str] = [
    # Loudness / spectral
    "Loudness_sma3",
    "alphaRatio_sma3",
    "hammarbergIndex_sma3",
    "slope0-500_sma3",
    "slope500-1500_sma3",
    "spectralFlux_sma3",
    # MFCCs
    "mfcc1_sma3",
    "mfcc2_sma3",
    "mfcc3_sma3",
    "mfcc4_sma3",
    # Fundamental frequency / voice quality
    "F0semitoneFrom27.5Hz_sma3nz",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
    "logRelF0-H1-H2_sma3nz",
    "logRelF0-H1-A3_sma3nz",
    # Formants
    "F1frequency_sma3nz",
    "F1bandwidth_sma3nz",
    "F1amplitudeLogRelF0_sma3nz",
    "F2frequency_sma3nz",
    "F2bandwidth_sma3nz",
    "F2amplitudeLogRelF0_sma3nz",
    "F3frequency_sma3nz",
    "F3bandwidth_sma3nz",
    "F3amplitudeLogRelF0_sma3nz",
]


class OpenSmileDataset(Dataset):
    """PyTorch Dataset for OpenSMILE audio feature CSVs.

    Directory layout expected::

        root_dir/
          Train/
            Truthful/   *.csv   → label 0
            Deceptive/  *.csv   → label 1
          Test/
            Truthful/   *.csv   → label 0
            Deceptive/  *.csv   → label 1

    Each CSV has ``time_start`` / ``time_end`` columns followed by 25 per-frame
    audio feature columns.  Each file contains one row per analysis frame
    (10 ms hop), giving variable-length sequences across clips.

    Each ``__getitem__`` returns ``(seq, label)`` where:
      - ``seq``   : float32 tensor of shape ``(T, D)`` — T rows in the CSV
      - ``label`` : int, 0 (Truthful) or 1 (Deceptive)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        feature_cols: Optional[List[str]] = None,
        subsample_k: int = 1,
        motion_method: Literal["none", "feature_diff"] = "none",
        motion_low: float = 0.0,
        motion_high: float = float("inf"),
    ) -> None:
        self.feature_cols = feature_cols or AUDIO_FEATURE_COLS
        self.subsample_k = subsample_k
        self.motion_method = motion_method
        self.motion_low = motion_low
        self.motion_high = motion_high

        split_dir = Path(root_dir) / split
        self.samples: List[Tuple[Path, int]] = []
        for label_name, label in [("Truthful", 0), ("Deceptive", 1)]:
            for csv_path in sorted((split_dir / label_name).glob("*.csv")):
                self.samples.append((csv_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def _motion_mask(self, seq: torch.Tensor) -> np.ndarray:
        T = seq.shape[0]
        scores = np.zeros(T, dtype=np.float32)
        if self.motion_method == "feature_diff":
            diff = np.abs(np.diff(seq.numpy(), axis=0)).mean(axis=1)
            scores[1:] = diff
        return (scores >= self.motion_low) & (scores <= self.motion_high)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        csv_path, label = self.samples[idx]

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        seq = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)

        # Per-sample z-score normalization before motion filtering.
        seq = (seq - seq.mean(dim=0, keepdim=True)) / seq.std(
            dim=0, keepdim=True
        ).clamp_min(1e-6)

        if self.motion_method != "none":
            mask = self._motion_mask(seq)
            if mask.any():
                seq = seq[mask]

        if self.subsample_k > 1:
            seq = seq[:: self.subsample_k]
        return seq, label
