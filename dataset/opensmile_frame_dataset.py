from typing import List, Optional
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

from dataset.opensmile_dataset import OpenSmileDataset

LLD_FEATURE_COLS: List[str] = [
    "Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3",
    "slope0-500_sma3", "slope500-1500_sma3", "spectralFlux_sma3",
    "mfcc1_sma3", "mfcc2_sma3", "mfcc3_sma3", "mfcc4_sma3",
    "F0semitoneFrom27.5Hz_sma3nz", "jitterLocal_sma3nz", "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz", "logRelF0-H1-H2_sma3nz", "logRelF0-H1-A3_sma3nz",
    "F1frequency_sma3nz", "F1bandwidth_sma3nz", "F1amplitudeLogRelF0_sma3nz",
    "F2frequency_sma3nz", "F2bandwidth_sma3nz", "F2amplitudeLogRelF0_sma3nz",
    "F3frequency_sma3nz", "F3bandwidth_sma3nz", "F3amplitudeLogRelF0_sma3nz",
]


class FrameDataset(OpenSmileDataset):
    """OpenSmileDataset extended with max_len truncation and optional global norm."""

    def __init__(self, *args, max_len: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_len = max_len

    def __getitem__(self, idx):
        seq, label = super().__getitem__(idx)
        if self.max_len is not None:
            seq = seq[: self.max_len]
        return seq, label


class _NormalizedDataset(Dataset):
    def __init__(self, base: Dataset, indices, mean: torch.Tensor, std: torch.Tensor):
        self.base = base
        self.indices = indices
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        seq, label = self.base[self.indices[i]]
        return (seq - self.mean) / self.std, label


class _NormalizedTestDataset(Dataset):
    def __init__(self, base: Dataset, mean: torch.Tensor, std: torch.Tensor):
        self.base = base
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        seq, label = self.base[i]
        return (seq - self.mean) / self.std, label


def _collate_fn(batch):
    seqs, ys = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    x_padded = pad_sequence(seqs, batch_first=True)
    y = torch.tensor(ys, dtype=torch.float32)
    return x_padded, lengths, y


def make_frame_loaders(
    root_dir: str,
    val_frac: float = 0.2,
    batch_size: int = 16,
    stride: int = 1,
    max_len: Optional[int] = None,
    normalize: bool = True,
    feature_cols: Optional[List[str]] = None,
    seed: int = 42,
):
    cols = feature_cols or LLD_FEATURE_COLS

    train_full = FrameDataset(
        root_dir=root_dir, split="Train",
        feature_cols=cols, subsample_k=stride, max_len=max_len,
    )
    test_base = FrameDataset(
        root_dir=root_dir, split="Test",
        feature_cols=cols, subsample_k=stride, max_len=max_len,
    )

    n_val = int(len(train_full) * val_frac)
    n_train = len(train_full) - n_val
    train_subset, val_subset = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    d_in = len(cols)

    if normalize:
        # Fit global mean/std on training frames (on top of per-sample norm)
        all_frames = torch.cat(
            [train_full[i][0] for i in train_subset.indices], dim=0
        )
        mean = all_frames.mean(dim=0)
        std = all_frames.std(dim=0).clamp_min(1e-6)

        train_ds = _NormalizedDataset(train_full, train_subset.indices, mean, std)
        val_ds = _NormalizedDataset(train_full, val_subset.indices, mean, std)
        test_ds = _NormalizedTestDataset(test_base, mean, std)
    else:
        train_ds = train_subset
        val_ds = val_subset
        test_ds = test_base

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)

    return train_loader, val_loader, test_loader, d_in
