"""PyTorch Dataset and DataLoaders for the pretrained-feature pipeline.

``PretrainedMultimodalDataset`` aligns raw video clips (MP4), audio waveforms
(WAV), and Whisper transcription text files by canonical UR-LYING key and
returns per-sample raw inputs for the ``PretrainedLateFusionClassifier``.

Unlike ``MultimodalDatasetAVT``, **no** pretrained model inference is run in
the dataset — the returned tensors are the raw model inputs:

* ``pixel_values``   : ``(T, 3, 224, 224)`` — preprocessed frames (VideoMAE)
* ``waveform``       : ``(T_audio,)``        — raw 16 kHz PCM (Wav2Vec2)
* ``input_ids``      : ``(L,)``              — RoBERTa token IDs
* ``attention_mask`` : ``(L,)``              — RoBERTa token attention mask
* ``label``          : ``int``               — 0 (Truthful) / 1 (Deceptive)

Any modality root can be ``None`` to disable that modality.

Directory layout expected::

    clips_root/  {Train,Test}/{Deceptive,Truthful}/*.mp4
    audio_root/  {Train,Test}/{Deceptive,Truthful}/*.wav
    whisper_root/{Train,Test}/{Deceptive,Truthful}/*.txt

Usage::

    from dataset.pretrained_multimodal_dataset import (
        PretrainedMultimodalDataset,
        _pretrained_collate,
        make_pretrained_loaders,
    )
    from torch.utils.data import DataLoader

    ds = PretrainedMultimodalDataset(
        clips_root="dataset/UR_LYING_Deception_Dataset/clips_raw",
        audio_root="dataset/UR_LYING_Deception_Dataset/audio_raw",
        whisper_root="dataset/UR_LYING_Deception_Dataset/whisper_raw",
        split="Train",
    )
    loader = DataLoader(ds, batch_size=4, collate_fn=_pretrained_collate)
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

from dataset.multimodal_dataset import (
    _build_stem_map,
    openface_ur_lying_key,
)
from dataset.video_clip_dataset import VideoClipDataset
from dataset.audio_wave_dataset import AudioWaveDataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PretrainedMultimodalDataset(Dataset):
    """Aligns clips_raw (video), audio_raw (WAV), and whisper_raw (text) by
    canonical UR-LYING key and returns raw pretrained-model inputs.

    Key alignment:
    * Video (MP4)  : ``openface_ur_lying_key(stem)``
    * Audio (WAV)  : identity (stems are already canonical keys, matching
                     the output of ``openface_ur_lying_key`` on the MP4 stems)
    * Text (TXT)   : identity (stems are already canonical keys)

    Only clips present in **all** enabled modalities are included; warnings are
    emitted for unmatched keys.

    Args:
        clips_root:          Root for MP4 clips.  ``None`` disables video.
        audio_root:          Root for WAV files.  ``None`` disables audio.
        whisper_root:        Root for ``.txt`` transcription files.
                             ``None`` disables text.
        split:               ``"Train"`` or ``"Test"``.
        num_frames:          Frames to sample per clip for VideoMAE.
        video_processor_name:HuggingFace VideoMAE processor name.
        roberta_model_name:  HuggingFace tokenizer name for text.
        roberta_max_length:  Maximum token sequence length (incl. special tokens).
        max_audio_seconds:   Maximum audio duration in seconds.
    """

    # Tokenizer cache shared across instances.
    _tokenizer_cache: Dict[str, Any] = {}

    def __init__(
        self,
        clips_root: Optional[str],
        audio_root: Optional[str],
        whisper_root: Optional[str],
        split: str = "Train",
        num_frames: int = 16,
        video_stride: Optional[int] = None,
        video_processor_name: str = "MCG-NJU/videomae-base",
        roberta_model_name: str = "roberta-base",
        roberta_max_length: int = 512,
        max_audio_seconds: float = 60.0,
    ) -> None:
        if clips_root is None and audio_root is None and whisper_root is None:
            raise ValueError(
                "At least one of clips_root, audio_root, whisper_root must be set."
            )

        self.split = split
        self.roberta_model_name = roberta_model_name
        self.roberta_max_length = roberta_max_length

        # ----------------------------------------------------------------
        # Sub-dataset loaders
        # ----------------------------------------------------------------
        if clips_root is not None:
            self._video_loader: Optional[VideoClipDataset] = VideoClipDataset(
                clips_root,
                split,
                num_frames=num_frames,
                stride=video_stride,
                processor_name=video_processor_name,
                key_fn=openface_ur_lying_key,
            )
            self._video_path_to_idx: Optional[Dict[Path, int]] = {
                p: i for i, (p, _) in enumerate(self._video_loader.samples)
            }
        else:
            self._video_loader = None
            self._video_path_to_idx = None

        if audio_root is not None:
            self._audio_loader: Optional[AudioWaveDataset] = AudioWaveDataset(
                audio_root,
                split,
                max_seconds=max_audio_seconds,
            )
            self._audio_path_to_idx: Optional[Dict[Path, int]] = {
                p: i for i, (p, _) in enumerate(self._audio_loader.samples)
            }
        else:
            self._audio_loader = None
            self._audio_path_to_idx = None

        # ----------------------------------------------------------------
        # Key → (path, label) maps for alignment
        # ----------------------------------------------------------------
        video_map = (
            _build_stem_map(
                Path(clips_root) / split, openface_ur_lying_key, glob="*.mp4"
            )
            if clips_root is not None
            else {}
        )
        audio_map = (
            _build_stem_map(Path(audio_root) / split, lambda s: s, glob="*.wav")
            if audio_root is not None
            else {}
        )
        text_map = (
            _build_stem_map(Path(whisper_root) / split, lambda s: s, glob="*.txt")
            if whisper_root is not None
            else {}
        )

        # Warn about unmatched keys across active modalities
        active: List[Tuple[str, Dict[str, Tuple[Path, int]]]] = []
        if clips_root is not None:
            active.append(("video", video_map))
        if audio_root is not None:
            active.append(("audio", audio_map))
        if whisper_root is not None:
            active.append(("text", text_map))

        all_keys = set().union(*(m for _, m in active))
        for name, mod_map in active:
            missing = all_keys - set(mod_map)
            if missing:
                warnings.warn(
                    f"[PretrainedMultimodalDataset] {len(missing)} keys missing "
                    f"from {name} in split='{split}': {sorted(missing)}"
                )

        # Inner join over enabled modalities
        nonempty = [m for m in [video_map, audio_map, text_map] if m]
        common: set[str] = set(nonempty[0])
        for m in nonempty[1:]:
            common &= set(m)
        common_keys = sorted(common)

        label_map = nonempty[0]
        self.samples: List[
            Tuple[Optional[Path], Optional[Path], Optional[Path], int]
        ] = [
            (
                video_map[k][0] if clips_root is not None else None,
                audio_map[k][0] if audio_root is not None else None,
                text_map[k][0] if whisper_root is not None else None,
                label_map[k][1],
            )
            for k in common_keys
        ]

        # Store roots for reference
        self._clips_root = clips_root
        self._audio_root = audio_root
        self._whisper_root = whisper_root

    # ------------------------------------------------------------------
    # Lazy tokenizer loading
    # ------------------------------------------------------------------

    def _get_tokenizer(self) -> Any:
        if self.roberta_model_name not in PretrainedMultimodalDataset._tokenizer_cache:
            try:
                from transformers import AutoTokenizer  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for text tokenization."
                ) from exc
            PretrainedMultimodalDataset._tokenizer_cache[self.roberta_model_name] = (
                AutoTokenizer.from_pretrained(self.roberta_model_name)
            )
        return PretrainedMultimodalDataset._tokenizer_cache[self.roberta_model_name]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        Optional[torch.Tensor],  # pixel_values (T, 3, H, W)
        Optional[torch.Tensor],  # waveform (T_audio,)
        Optional[torch.Tensor],  # input_ids (L,)
        Optional[torch.Tensor],  # attention_mask (L,)
        int,  # label
    ]:
        video_path, audio_path, text_path, label = self.samples[idx]

        # Video
        pixel_values: Optional[torch.Tensor] = None
        if (
            video_path is not None
            and self._video_loader is not None
            and self._video_path_to_idx is not None
        ):
            pixel_values, _ = self._video_loader[self._video_path_to_idx[video_path]]

        # Audio
        waveform: Optional[torch.Tensor] = None
        if (
            audio_path is not None
            and self._audio_loader is not None
            and self._audio_path_to_idx is not None
        ):
            waveform, _ = self._audio_loader[self._audio_path_to_idx[audio_path]]

        # Text — tokenize .txt on the fly (no model inference)
        input_ids: Optional[torch.Tensor] = None
        attention_mask: Optional[torch.Tensor] = None
        if text_path is not None:
            text = text_path.read_text(encoding="utf-8").strip()
            tokenizer = self._get_tokenizer()
            enc = tokenizer(
                text,
                return_tensors="pt",
                max_length=self.roberta_max_length,
                truncation=True,
                padding="max_length",
            )
            input_ids = enc["input_ids"].squeeze(0)  # (L,)
            attention_mask = enc["attention_mask"].squeeze(0)  # (L,)

        return pixel_values, waveform, input_ids, attention_mask, label


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def _pretrained_collate(
    batch: list,
) -> Tuple[
    Optional[torch.Tensor],  # pixel_values  (B, N_max, T, 3, H, W)
    Optional[torch.Tensor],  # waveforms     (B, T_max)
    Optional[torch.Tensor],  # audio_attn    (B, T_max)
    Optional[torch.Tensor],  # input_ids     (B, L)
    Optional[torch.Tensor],  # text_attn     (B, L)
    Optional[torch.Tensor],  # video_lengths (B,)  — actual N_windows per sample
    torch.Tensor,  # y             (B,)
]:
    """Collate a list of ``PretrainedMultimodalDataset`` samples into a batch.

    * ``pixel_values``: padded to ``N_max`` windows → ``(B, N_max, T, 3, H, W)``
    * ``video_lengths``: ``(B,)`` long tensor with actual N_windows per sample.
    * ``waveforms``   : right-padded to ``T_max`` → ``(B, T_max)``; matching
                        ``audio_attn`` mask computed from original lengths.
    * ``input_ids``   : already padded to ``max_length`` by the tokenizer → stacked.
    * ``text_attn``   : same — stacked.
    """
    pixel_values_list, waveform_list, input_ids_list, attn_mask_list, ys = zip(*batch)

    # --- pixel_values (variable N_windows: pad to N_max) ---
    pixel_values: Optional[torch.Tensor] = None
    video_lengths: Optional[torch.Tensor] = None
    if pixel_values_list[0] is not None:
        video_lengths = torch.tensor(
            [pv.shape[0] for pv in pixel_values_list], dtype=torch.long
        )
        N_max = int(video_lengths.max().item())
        padded = []
        for pv in pixel_values_list:
            pad_n = N_max - pv.shape[0]
            if pad_n > 0:
                pv = torch.cat([pv, pv.new_zeros(pad_n, *pv.shape[1:])], dim=0)
            padded.append(pv)
        pixel_values = torch.stack(padded)  # (B, N_max, T, 3, H, W)

    # --- waveforms (variable length: pad + mask) ---
    waveforms: Optional[torch.Tensor] = None
    audio_attn: Optional[torch.Tensor] = None
    if waveform_list[0] is not None:
        lengths = torch.tensor([w.shape[0] for w in waveform_list], dtype=torch.long)
        waveforms = pad_sequence(list(waveform_list), batch_first=True)  # (B, T_max)
        t_max = waveforms.shape[1]
        audio_attn = (torch.arange(t_max).unsqueeze(0) < lengths.unsqueeze(1)).long()

    # --- input_ids / text attention mask (fixed length: stack directly) ---
    input_ids: Optional[torch.Tensor] = None
    text_attn: Optional[torch.Tensor] = None
    if input_ids_list[0] is not None:
        input_ids = torch.stack(list(input_ids_list))
        text_attn = torch.stack(list(attn_mask_list))

    y = torch.tensor(ys, dtype=torch.float32)
    return pixel_values, waveforms, audio_attn, input_ids, text_attn, video_lengths, y


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def make_pretrained_loaders(
    clips_root: Optional[str],
    audio_root: Optional[str],
    whisper_root: Optional[str],
    val_frac: float = 0.2,
    batch_size: int = 4,
    seed: int = 42,
    num_frames: int = 16,
    video_stride: Optional[int] = None,
    video_processor_name: str = "MCG-NJU/videomae-base",
    roberta_model_name: str = "roberta-base",
    roberta_max_length: int = 512,
    max_audio_seconds: float = 60.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders for the pretrained pipeline.

    .. note::
        ``num_workers`` defaults to 0 because PyAV (libav) is not fork-safe.
        Increase it only if you are not using video clips (``clips_root=None``).

    Returns:
        ``train_loader, val_loader, test_loader``
    """
    ds_kwargs: dict = dict(
        num_frames=num_frames,
        video_stride=video_stride,
        video_processor_name=video_processor_name,
        roberta_model_name=roberta_model_name,
        roberta_max_length=roberta_max_length,
        max_audio_seconds=max_audio_seconds,
    )

    full_train = PretrainedMultimodalDataset(
        clips_root, audio_root, whisper_root, "Train", **ds_kwargs
    )
    n_val = max(1, int(len(full_train) * val_frac))
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=generator)

    test_ds = PretrainedMultimodalDataset(
        clips_root, audio_root, whisper_root, "Test", **ds_kwargs
    )

    loader_kwargs: dict = dict(
        batch_size=batch_size,
        collate_fn=_pretrained_collate,
        num_workers=0,  # PyAV is not fork-safe
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
