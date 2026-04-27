"""Late-fusion classifier using LoRA-wrapped pretrained encoders.

``PretrainedLateFusionClassifier`` composes up to three pretrained encoders
(VideoMAE, Wav2Vec2, RoBERTa) with optional LoRA adapters and a 2-layer MLP
classification head.  Each active encoder produces a pooled ``(B, 768)`` feature
vector; these are concatenated and passed through the classifier.

This model corresponds to ``model_type = "pretrained_avt"`` in checkpoints.

Usage::

    from model.pretrained_late_fusion import PretrainedLateFusionClassifier
    from model.pretrained_encoders import LoRAEncoderConfig

    lora_cfg = LoRAEncoderConfig(r=8, lora_alpha=16)
    model = PretrainedLateFusionClassifier(
        use_visual=True, use_audio=True, use_text=True,
        lora_cfg=lora_cfg,
    )
    logits = model(
        pixel_values=...,   # (B, T, 3, 224, 224)
        waveforms=...,      # (B, T_audio)
        input_ids=...,      # (B, L)
        text_attention_mask=...,  # (B, L)
    )
"""

from typing import Optional

import torch
import torch.nn as nn

from model.LateFusionBiGRU import BiGRUEncoder
from model.pretrained_encoders import (
    LoRAEncoderConfig,
    RoBERTaEncoder,
    VideoMAEEncoder,
    Wav2Vec2Encoder,
)


class PretrainedLateFusionClassifier(nn.Module):
    """Late-fusion classifier with LoRA-wrapped pretrained backbones and BiGRU temporal encoders.

    Each active modality encoder extracts a token/frame/patch sequence from the
    pretrained backbone.  Each sequence is then passed through a per-modality
    BiGRU (attention-pooled) before concatenation and MLP classification.

    Args:
        use_visual:        Enable VideoMAE encoder.
        use_audio:         Enable Wav2Vec2 encoder.
        use_text:          Enable RoBERTa encoder.
        video_model_name:  HuggingFace VideoMAE model identifier.
        audio_model_name:  HuggingFace Wav2Vec2 model identifier.
        text_model_name:   HuggingFace RoBERTa model identifier.
        lora_cfg:          Shared :class:`~model.pretrained_encoders.LoRAEncoderConfig`
                           applied to all active encoders.  ``None`` freezes all
                           backbones (inference / feature extraction only).
        gru_hidden:        Hidden size for each per-modality BiGRU encoder.
        gru_pooling:       Pooling strategy for BiGRU output (``"attention"``,
                           ``"mean"``, ``"max"``, ``"last"``).
        fusion_hidden:     Hidden size of the MLP classifier head.
        dropout:           Dropout applied inside the classifier head and BiGRUs.
        text_pool:         Unused (kept for API compatibility); pooling is now
                           handled by the BiGRU encoder.
    """

    def __init__(
        self,
        use_visual: bool = True,
        use_audio: bool = True,
        use_text: bool = True,
        video_model_name: str = "MCG-NJU/videomae-base",
        audio_model_name: str = "facebook/wav2vec2-base-960h",
        text_model_name: str = "roberta-base",
        lora_cfg: Optional[LoRAEncoderConfig] = None,
        gru_hidden: int = 128,
        gru_pooling: str = "attention",
        fusion_hidden: int = 256,
        dropout: float = 0.3,
        text_pool: str = "cls",
    ) -> None:
        super().__init__()

        if not any([use_visual, use_audio, use_text]):
            raise ValueError("At least one encoder must be enabled.")

        self.use_visual = use_visual
        self.use_audio = use_audio
        self.use_text = use_text

        if use_visual:
            self.visual_encoder: Optional[VideoMAEEncoder] = VideoMAEEncoder(
                model_name=video_model_name,
                lora_cfg=lora_cfg,
            )
            self.visual_gru: Optional[BiGRUEncoder] = BiGRUEncoder(
                d_in=768, hidden=gru_hidden, pooling=gru_pooling, dropout=dropout
            )
        else:
            self.visual_encoder = None
            self.visual_gru = None

        if use_audio:
            self.audio_encoder: Optional[Wav2Vec2Encoder] = Wav2Vec2Encoder(
                model_name=audio_model_name,
                lora_cfg=lora_cfg,
            )
            self.audio_gru: Optional[BiGRUEncoder] = BiGRUEncoder(
                d_in=768, hidden=gru_hidden, pooling=gru_pooling, dropout=dropout
            )
        else:
            self.audio_encoder = None
            self.audio_gru = None

        if use_text:
            self.text_encoder: Optional[RoBERTaEncoder] = RoBERTaEncoder(
                model_name=text_model_name,
                lora_cfg=lora_cfg,
                pool=text_pool,
            )
            self.text_gru: Optional[BiGRUEncoder] = BiGRUEncoder(
                d_in=768, hidden=gru_hidden, pooling=gru_pooling, dropout=dropout
            )
        else:
            self.text_encoder = None
            self.text_gru = None

        n_active = sum([use_visual, use_audio, use_text])
        fused_dim = n_active * (2 * gru_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        waveforms: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        video_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute binary classification logits.

        Args:
            pixel_values:        ``(B, N_windows, T, C, H, W)`` — required when
                                 ``use_visual=True``.  N_windows may be padded;
                                 pass ``video_lengths`` for accurate masking.
            waveforms:           ``(B, T_audio)`` — required when ``use_audio=True``.
            audio_attention_mask:``(B, T_audio)`` — padding mask for variable-length audio.
            input_ids:           ``(B, L)`` — required when ``use_text=True``.
            text_attention_mask: ``(B, L)`` — padding mask for text.
            video_lengths:       ``(B,)`` long tensor — actual number of windows per
                                 sample.  ``None`` treats all windows as real.

        Returns:
            ``(B,)`` raw logits (before sigmoid / BCEWithLogitsLoss).
        """
        features = []

        if self.use_visual:
            if pixel_values is None:
                raise ValueError("pixel_values required when use_visual=True.")
            assert self.visual_encoder is not None and self.visual_gru is not None
            # (B, N_windows, 768) — padded windows are zeroed; real count in video_lengths
            seq = self.visual_encoder(pixel_values, return_sequence=True)
            B = seq.shape[0]
            lengths = (
                video_lengths.to(seq.device)
                if video_lengths is not None
                else torch.full((B,), seq.shape[1], dtype=torch.long, device=seq.device)
            )
            features.append(self.visual_gru(seq, lengths))

        if self.use_audio:
            if waveforms is None:
                raise ValueError("waveforms required when use_audio=True.")
            assert self.audio_encoder is not None and self.audio_gru is not None
            # (B, T_frames, 768) — backbone handles internal padding
            seq = self.audio_encoder(
                waveforms, audio_attention_mask, return_sequence=True
            )
            B, T_a, _ = seq.shape
            lengths = torch.full((B,), T_a, dtype=torch.long, device=seq.device)
            features.append(self.audio_gru(seq, lengths))

        if self.use_text:
            if input_ids is None or text_attention_mask is None:
                raise ValueError(
                    "input_ids and text_attention_mask required when use_text=True."
                )
            assert self.text_encoder is not None and self.text_gru is not None
            # (B, L, 768) — use attention mask to get real token counts
            seq = self.text_encoder(
                input_ids, text_attention_mask, return_sequence=True
            )
            lengths = text_attention_mask.sum(dim=1).long()
            features.append(self.text_gru(seq, lengths))

        fused = torch.cat(features, dim=-1)
        return self.classifier(fused).squeeze(-1)
