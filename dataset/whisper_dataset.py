"""PyTorch Dataset for Whisper transcription files with RoBERTa token embeddings.

Each sample's text is tokenized and encoded with a frozen RoBERTa model.
Embeddings are cached in memory after the first forward pass.

Directory layout expected::

    {root_dir}/whisper_{source}/{split}/{Truthful,Deceptive}/*.txt

Usage::

    from dataset.whisper_dataset import WhisperDataset

    ds = WhisperDataset("dataset/UR_LYING_Deception_Dataset", split="Train")
    token_embs, label = ds[0]   # token_embs: (T, 768), label: 0 or 1
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

ROBERTA_D_IN = 768  # roberta-base hidden size


class WhisperDataset(Dataset):
    """Dataset that reads Whisper ``.txt`` transcription files and returns
    per-token RoBERTa embeddings (last hidden state, special tokens stripped).

    Args:
        root_dir:   Root directory containing ``whisper_raw/`` and
                    ``whisper_processed/`` sub-trees.
        split:      ``"Train"`` or ``"Test"``.
        source:     ``"raw"`` → reads from ``whisper_raw/``;
                    ``"processed"`` → reads from ``whisper_processed/``.
        model_name: HuggingFace model identifier (default ``"roberta-base"``).
        max_length: Maximum tokenizer sequence length including special tokens.
        key_fn:     Optional function mapping a filename stem to a canonical key.
                    Defaults to identity (stem is already the canonical key).
    """

    # Class-level tokenizer / model cache so multiple dataset instances
    # (e.g. Train + Test) share a single copy in memory.
    _tokenizer_cache: Dict[str, Any] = {}
    _model_cache: Dict[str, Any] = {}

    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        source: Literal["raw", "processed"] = "raw",
        model_name: str = "roberta-base",
        max_length: int = 512,
        key_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.source = source
        self.model_name = model_name
        self.max_length = max_length
        self._key_fn = key_fn or (lambda s: s)

        split_dir = self.root_dir / f"whisper_{source}" / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Whisper split directory not found: {split_dir}")

        self.samples: List[Tuple[Path, int]] = []
        for label_name, label in [("Truthful", 0), ("Deceptive", 1)]:
            label_dir = split_dir / label_name
            if label_dir.exists():
                for txt_path in sorted(label_dir.glob("*.txt")):
                    self.samples.append((txt_path, label))

        # In-memory embedding cache: index → tensor (T, 768)
        self._cache: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def d_in(self) -> int:
        return ROBERTA_D_IN

    # ------------------------------------------------------------------
    # Lazy tokenizer / model loading
    # ------------------------------------------------------------------

    def _get_tokenizer(self) -> Any:
        if self.model_name not in WhisperDataset._tokenizer_cache:
            WhisperDataset._tokenizer_cache[self.model_name] = (
                AutoTokenizer.from_pretrained(self.model_name)
            )
        return WhisperDataset._tokenizer_cache[self.model_name]

    def _get_model(self) -> torch.nn.Module:
        if self.model_name not in WhisperDataset._model_cache:
            model = AutoModel.from_pretrained(self.model_name)
            model.eval()
            WhisperDataset._model_cache[self.model_name] = model
        return WhisperDataset._model_cache[self.model_name]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx in self._cache:
            return self._cache[idx], self.samples[idx][1]

        txt_path, label = self.samples[idx]
        text = txt_path.read_text(encoding="utf-8").strip()

        tokenizer = self._get_tokenizer()
        model = self._get_model()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # last_hidden_state: (1, T, 768) — strip batch dim
        hidden: torch.Tensor = outputs.last_hidden_state.squeeze(0)  # (T, 768)

        # Strip leading <s> and trailing </s> special tokens
        if hidden.shape[0] >= 2:
            hidden = hidden[1:-1]

        hidden = hidden.float().cpu()
        self._cache[idx] = hidden
        return hidden, label

    def key_for(self, idx: int) -> str:
        """Return the canonical key for sample ``idx``."""
        stem = self.samples[idx][0].stem
        return self._key_fn(stem)
