import torch
import torch.nn as nn


class BiGRUClassifier(nn.Module):
    def __init__(
        self,
        d_in: int,               # feature dim D
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        pooling: str = "attention",   # "mean" | "max" | "last" | "topk_mean" | "attn"
        top_k: int = 5
    ):
        super().__init__()
        self.pooling = pooling
        self.top_k = top_k

        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # BiGRU output dim = 2 * hidden
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)   # binary classification logits
        )

        if pooling == "topk_mean":
            self.score_layer = nn.Linear(2 * hidden, 1)

        if pooling == "attention":
            self.attn = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: (B, T, D) padded
        lengths: (B,) actual sequence lengths
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.gru(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # out: (B, T_max, 2H)

        B, T_max, _ = out.shape
        device = out.device

        mask = torch.arange(T_max, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        if self.pooling == "mean":
            mask_f = mask.unsqueeze(-1).float()
            pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

        elif self.pooling == "max":
            out_masked = out.masked_fill(~mask.unsqueeze(-1), -1e9)
            pooled = out_masked.max(dim=1).values

        elif self.pooling == "last":
            idx = (lengths - 1).clamp_min(0)
            pooled = out[torch.arange(B, device=device), idx]

        elif self.pooling == "topk_mean":
            scores = self.score_layer(out).squeeze(-1)   # (B, T)
            scores = scores.masked_fill(~mask, -1e9)

            k = min(self.top_k, T_max)
            topk_idx = scores.topk(k=k, dim=1).indices
            topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, out.size(-1))
            topk_feats = torch.gather(out, dim=1, index=topk_idx_expanded)

            pooled = topk_feats.mean(dim=1)

        elif self.pooling == "attention":
            attn_scores = self.attn(out).squeeze(-1)   # (B, T)
            attn_scores = attn_scores.masked_fill(~mask, -1e9)

            attn_weights = torch.softmax(attn_scores, dim=1)
            pooled = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)

        else:
            raise ValueError("pooling must be one of: mean | max | last")

        logits = self.classifier(pooled).squeeze(-1)  # (B,)
        return logits