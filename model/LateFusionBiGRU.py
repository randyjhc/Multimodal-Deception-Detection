from typing import Optional

import torch
import torch.nn as nn


class BiGRUEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        pooling: str = "attention",
        top_k: int = 5,
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

        if pooling == "topk_mean":
            self.score_layer = nn.Linear(2 * hidden, 1)

        if pooling == "attention":
            self.attn = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        batch_size, max_steps, _ = out.shape
        device = out.device
        mask = torch.arange(max_steps, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        if self.pooling == "mean":
            mask_f = mask.unsqueeze(-1).float()
            pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

        elif self.pooling == "max":
            out_masked = out.masked_fill(~mask.unsqueeze(-1), -1e9)
            pooled = out_masked.max(dim=1).values

        elif self.pooling == "last":
            idx = (lengths - 1).clamp_min(0)
            pooled = out[torch.arange(batch_size, device=device), idx]

        elif self.pooling == "topk_mean":
            scores = self.score_layer(out).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)

            k = min(self.top_k, max_steps)
            topk_idx = scores.topk(k=k, dim=1).indices
            topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, out.size(-1))
            topk_feats = torch.gather(out, dim=1, index=topk_idx_expanded)
            pooled = topk_feats.mean(dim=1)

        elif self.pooling == "attention":
            attn_scores = self.attn(out).squeeze(-1)
            attn_scores = attn_scores.masked_fill(~mask, -1e9)

            attn_weights = torch.softmax(attn_scores, dim=1)
            pooled = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)

        else:
            raise ValueError("pooling must be one of: mean | max | last | topk_mean | attention")

        return pooled


class LateFusionBiGRUClassifier(nn.Module):
    def __init__(
        self,
        visual_d_in: Optional[int] = None,
        audio_d_in: Optional[int] = None,
        text_d_in: Optional[int] = None,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        pooling: str = "attention",
        top_k: int = 5,
        fusion_hidden: int = 128,
        use_visual: bool = True,
        use_audio: bool = True,
        use_text: bool = True,
    ):
        super().__init__()

        self.use_visual = use_visual
        self.use_audio = use_audio
        self.use_text = use_text

        if not any([self.use_visual, self.use_audio, self.use_text]):
            raise ValueError("At least one encoder must be enabled.")

        if self.use_visual:
            if visual_d_in is None:
                raise ValueError("visual_d_in must be provided when use_visual=True.")
            self.visual_encoder = BiGRUEncoder(
                d_in=visual_d_in,
                hidden=hidden,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
                top_k=top_k,
            )
        else:
            self.visual_encoder = None

        if self.use_audio:
            if audio_d_in is None:
                raise ValueError("audio_d_in must be provided when use_audio=True.")
            self.audio_encoder = BiGRUEncoder(
                d_in=audio_d_in,
                hidden=hidden,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
                top_k=top_k,
            )
        else:
            self.audio_encoder = None

        if self.use_text:
            if text_d_in is None:
                raise ValueError("text_d_in must be provided when use_text=True.")
            self.text_encoder = BiGRUEncoder(
                d_in=text_d_in,
                hidden=hidden,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
                top_k=top_k,
            )
        else:
            self.text_encoder = None

        num_active_encoders = sum([self.use_visual, self.use_audio, self.use_text])
        fused_dim = num_active_encoders * (2 * hidden)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(
        self,
        visual_x: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None,
        audio_x: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        text_x: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = []

        if self.use_visual:
            if visual_x is None or visual_lengths is None:
                raise ValueError("visual_x and visual_lengths are required when use_visual=True.")
            features.append(self.visual_encoder(visual_x, visual_lengths))

        if self.use_audio:
            if audio_x is None or audio_lengths is None:
                raise ValueError("audio_x and audio_lengths are required when use_audio=True.")
            features.append(self.audio_encoder(audio_x, audio_lengths))

        if self.use_text:
            if text_x is None or text_lengths is None:
                raise ValueError("text_x and text_lengths are required when use_text=True.")
            features.append(self.text_encoder(text_x, text_lengths))

        fused = torch.cat(features, dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        return logits
