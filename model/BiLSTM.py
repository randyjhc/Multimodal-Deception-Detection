import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        d_in: int,          # feature dim D (e.g., #FAUs)
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        pooling: str = "mean",  # "mean" | "max" | "last"
    ):
        super().__init__()
        self.pooling = pooling

        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # BiLSTM => output dim = 2*hidden
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # logits
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: (B, T, D) padded
        lengths: (B,) actual lengths of the input(int), sorted or unsorted OK
        """
        # pack -> LSTM -> unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )  # packed: (B, T_max, d_in), input can be unsorted
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # out: (B, T_max, 2H)

        # build mask: (B, T_max)
        B, T_max, _ = out.shape
        device = out.device
        mask = torch.arange(T_max, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # bool, [0<3, 1<3, 2<3, 3<3, 4<3] -> [T, T, T, F, F]

        if self.pooling == "mean":
            mask_f = mask.unsqueeze(-1).float()
            pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)  # (B, 2H)

        elif self.pooling == "max":
            # set padded positions to very negative so they won't win max
            out_masked = out.masked_fill(~mask.unsqueeze(-1), -1e9)
            pooled = out_masked.max(dim=1).values  # (B, 2H)

        elif self.pooling == "last":
            # last valid timestep per sequence
            idx = (lengths - 1).clamp_min(0)  # (B,)
            pooled = out[torch.arange(B, device=device), idx]  # (B, 2H)

        else:
            raise ValueError("pooling must be one of: mean | max | last")

        logits = self.classifier(pooled).squeeze(-1)  # (B,)
        return logits