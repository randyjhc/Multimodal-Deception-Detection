"""LoRA-wrapped pretrained encoder nn.Modules for VideoMAE, Wav2Vec2, and RoBERTa.

Each encoder wraps a HuggingFace pretrained backbone with optional PEFT LoRA
adapters.  When ``lora_cfg`` is provided the backbone is wrapped with
``get_peft_model()`` and only the LoRA adapter parameters are trainable.
When ``lora_cfg`` is ``None`` the backbone is fully frozen.

All encoders pool the backbone's last hidden state into a fixed-size vector
``(B, d_out)`` suitable for late fusion.

Usage::

    from model.pretrained_encoders import VideoMAEEncoder, Wav2Vec2Encoder, RoBERTaEncoder, LoRAEncoderConfig

    lora_cfg = LoRAEncoderConfig(r=8, lora_alpha=16)

    video_enc = VideoMAEEncoder(lora_cfg=lora_cfg)
    audio_enc = Wav2Vec2Encoder(lora_cfg=lora_cfg)
    text_enc  = RoBERTaEncoder(lora_cfg=lora_cfg)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LoRAEncoderConfig:
    """Configuration for LoRA adapters on a pretrained encoder.

    Args:
        r:            LoRA rank (number of low-rank matrices).
        lora_alpha:   LoRA scaling factor (effective scale = lora_alpha / r).
        lora_dropout: Dropout applied to LoRA adapter activations.
        bias:         Which biases to train: ``"none"``, ``"all"``, or ``"lora_only"``.
    """

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: str = "none"


# ---------------------------------------------------------------------------
# VideoMAE encoder
# ---------------------------------------------------------------------------


class VideoMAEEncoder(nn.Module):
    """VideoMAE backbone + optional LoRA → mean-pool patches → ``(B, 768)``.

    Args:
        model_name: HuggingFace identifier for the VideoMAE model.
        lora_cfg:   LoRA configuration.  ``None`` freezes the backbone.
    """

    d_out: int = 768

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        lora_cfg: Optional[LoRAEncoderConfig] = None,
    ) -> None:
        super().__init__()

        try:
            from transformers import VideoMAEModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("transformers is required for VideoMAEEncoder.") from exc

        backbone = VideoMAEModel.from_pretrained(model_name)

        if lora_cfg is not None:
            try:
                from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "peft is required for LoRA fine-tuning. " "Install it with: uv sync"
                ) from exc
            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=lora_cfg.lora_dropout,
                bias=lora_cfg.bias,
            )
            self.backbone = get_peft_model(backbone, peft_config)
        else:
            self.backbone = backbone
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of video clips.

        Args:
            pixel_values: ``(B, T, C, H, W)`` float32 tensor — preprocessed
                          frames in VideoMAE format (T=16, C=3, H=W=224).

        Returns:
            ``(B, 768)`` float32 feature vector (mean pool of patch tokens).
        """
        outputs = self.backbone(pixel_values=pixel_values)
        # last_hidden_state: (B, num_patches, 768)
        return outputs.last_hidden_state.mean(dim=1)


# ---------------------------------------------------------------------------
# Wav2Vec2 encoder
# ---------------------------------------------------------------------------


class Wav2Vec2Encoder(nn.Module):
    """Wav2Vec2 backbone + optional LoRA → mean-pool hidden states → ``(B, 768)``.

    Args:
        model_name: HuggingFace identifier for the Wav2Vec2 model.
        lora_cfg:   LoRA configuration.  ``None`` freezes the backbone.
    """

    d_out: int = 768

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        lora_cfg: Optional[LoRAEncoderConfig] = None,
    ) -> None:
        super().__init__()

        try:
            from transformers import Wav2Vec2Model  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("transformers is required for Wav2Vec2Encoder.") from exc

        backbone = Wav2Vec2Model.from_pretrained(model_name)

        if lora_cfg is not None:
            try:
                from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError("peft is required for LoRA fine-tuning.") from exc
            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_cfg.lora_dropout,
                bias=lora_cfg.bias,
            )
            self.backbone = get_peft_model(backbone, peft_config)
        else:
            self.backbone = backbone
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of raw audio waveforms.

        Args:
            waveforms:      ``(B, T_audio)`` float32 tensor — raw 16 kHz PCM.
            attention_mask: ``(B, T_audio)`` long tensor — 1 for real samples,
                            0 for padding.  May be ``None`` for same-length inputs.

        Returns:
            ``(B, 768)`` float32 feature vector (mean pool of frame hidden states).
        """
        outputs = self.backbone(
            input_values=waveforms,
            attention_mask=attention_mask,
        )
        # last_hidden_state: (B, T_frames, 768)
        return outputs.last_hidden_state.mean(dim=1)


# ---------------------------------------------------------------------------
# RoBERTa encoder
# ---------------------------------------------------------------------------


class RoBERTaEncoder(nn.Module):
    """RoBERTa backbone + optional LoRA → pooled feature → ``(B, 768)``.

    Args:
        model_name: HuggingFace identifier for the RoBERTa model.
        lora_cfg:   LoRA configuration.  ``None`` freezes the backbone.
        pool:       Pooling strategy: ``"cls"`` (first token) or ``"mean"``
                    (attention-masked mean over all tokens).
    """

    d_out: int = 768

    def __init__(
        self,
        model_name: str = "roberta-base",
        lora_cfg: Optional[LoRAEncoderConfig] = None,
        pool: str = "cls",
    ) -> None:
        super().__init__()

        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}")
        self.pool = pool

        try:
            from transformers import AutoModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("transformers is required for RoBERTaEncoder.") from exc

        backbone = AutoModel.from_pretrained(model_name)

        if lora_cfg is not None:
            try:
                from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError("peft is required for LoRA fine-tuning.") from exc
            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=lora_cfg.lora_dropout,
                bias=lora_cfg.bias,
            )
            self.backbone = get_peft_model(backbone, peft_config)
        else:
            self.backbone = backbone
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of tokenized text sequences.

        Args:
            input_ids:      ``(B, L)`` long tensor — RoBERTa token IDs.
            attention_mask: ``(B, L)`` long tensor — 1 for real tokens, 0 for padding.

        Returns:
            ``(B, 768)`` float32 feature vector.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (B, L, 768)
        hidden = outputs.last_hidden_state
        if self.pool == "cls":
            return hidden[:, 0, :]
        # mean pool over real tokens
        mask_f = attention_mask.unsqueeze(-1).float()
        return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
