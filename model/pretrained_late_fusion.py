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

from model.pretrained_encoders import (
    LoRAEncoderConfig,
    RoBERTaEncoder,
    VideoMAEEncoder,
    Wav2Vec2Encoder,
)


class PretrainedLateFusionClassifier(nn.Module):
    """Late-fusion classifier with LoRA-wrapped pretrained backbones.

    Each active modality encoder pools the pretrained model's output into a
    ``(B, 768)`` vector.  Active encoder outputs are concatenated and passed
    through a two-layer MLP classifier.

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
        fusion_hidden:     Hidden size of the MLP classifier head.
        dropout:           Dropout applied inside the classifier head.
        text_pool:         Pooling for RoBERTa: ``"cls"`` or ``"mean"``.
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
        else:
            self.visual_encoder = None

        if use_audio:
            self.audio_encoder: Optional[Wav2Vec2Encoder] = Wav2Vec2Encoder(
                model_name=audio_model_name,
                lora_cfg=lora_cfg,
            )
        else:
            self.audio_encoder = None

        if use_text:
            self.text_encoder: Optional[RoBERTaEncoder] = RoBERTaEncoder(
                model_name=text_model_name,
                lora_cfg=lora_cfg,
                pool=text_pool,
            )
        else:
            self.text_encoder = None

        n_active = sum([use_visual, use_audio, use_text])
        fused_dim = n_active * 768
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
    ) -> torch.Tensor:
        """Compute binary classification logits.

        Args:
            pixel_values:        ``(B, T, C, H, W)`` — required when ``use_visual=True``.
            waveforms:           ``(B, T_audio)`` — required when ``use_audio=True``.
            audio_attention_mask:``(B, T_audio)`` — padding mask for variable-length audio.
            input_ids:           ``(B, L)`` — required when ``use_text=True``.
            text_attention_mask: ``(B, L)`` — padding mask for text.

        Returns:
            ``(B,)`` raw logits (before sigmoid / BCEWithLogitsLoss).
        """
        features = []

        if self.use_visual:
            if pixel_values is None:
                raise ValueError("pixel_values required when use_visual=True.")
            assert self.visual_encoder is not None
            features.append(self.visual_encoder(pixel_values))

        if self.use_audio:
            if waveforms is None:
                raise ValueError("waveforms required when use_audio=True.")
            assert self.audio_encoder is not None
            features.append(self.audio_encoder(waveforms, audio_attention_mask))

        if self.use_text:
            if input_ids is None or text_attention_mask is None:
                raise ValueError(
                    "input_ids and text_attention_mask required when use_text=True."
                )
            assert self.text_encoder is not None
            features.append(self.text_encoder(input_ids, text_attention_mask))

        fused = torch.cat(features, dim=-1)
        return self.classifier(fused).squeeze(-1)
