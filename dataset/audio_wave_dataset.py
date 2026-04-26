"""PyTorch Dataset for 16 kHz mono WAV audio files (Wav2Vec2 input).

Each sample's raw waveform is read with soundfile and returned as a 1-D
float32 tensor ready to be fed into a Wav2Vec2 model.

Directory layout expected::

    {root_dir}/{split}/{Truthful,Deceptive}/*.wav

Usage::

    from dataset.audio_wave_dataset import AudioWaveDataset

    ds = AudioWaveDataset("dataset/UR_LYING_Deception_Dataset/audio_raw", split="Train")
    waveform, label = ds[0]   # waveform: (T_audio,), label: 0 or 1
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class AudioWaveDataset(Dataset):
    """Dataset that reads 16 kHz mono WAV files and returns raw waveforms.

    Each ``__getitem__`` returns ``(waveform, label)`` where:

    * ``waveform`` : float32 tensor ``(T_audio,)`` — raw PCM samples in [-1, 1],
      clipped to at most ``max_seconds`` of audio.
    * ``label``    : int, 0 (Truthful) or 1 (Deceptive).

    Args:
        root_dir:    Root directory containing ``{split}/{Truthful,Deceptive}/``
                     sub-trees of ``.wav`` files.
        split:       ``"Train"`` or ``"Test"``.
        target_sr:   Expected sample rate of the WAV files (default 16000).
                     Files at a different rate will raise a RuntimeError.
        max_seconds: Maximum audio duration to return.  Longer clips are
                     silently truncated to ``max_seconds * target_sr`` samples.
        key_fn:      Optional function mapping a filename stem to a canonical
                     key used by :meth:`key_for`.  Defaults to identity.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        target_sr: int = 16000,
        max_seconds: float = 60.0,
        key_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.target_sr = target_sr
        self.max_samples = int(max_seconds * target_sr)
        self._key_fn = key_fn or (lambda s: s)

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Audio split directory not found: {split_dir}")

        self.samples: List[Tuple[Path, int]] = []
        for label_name, label in [("Truthful", 0), ("Deceptive", 1)]:
            label_dir = split_dir / label_name
            if label_dir.exists():
                for wav_path in sorted(label_dir.glob("*.wav")):
                    self.samples.append((wav_path, label))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        wav_path, label = self.samples[idx]

        try:
            import soundfile as sf  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "soundfile is required for AudioWaveDataset. "
                "Install it with: uv sync --group pretrained"
            ) from exc

        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

        if sr != self.target_sr:
            raise RuntimeError(
                f"Expected {self.target_sr} Hz but got {sr} Hz for {wav_path.name}. "
                "Re-extract audio with dataset/extract_audio.py."
            )

        # data is (T,) for mono; silently truncate to max_samples
        waveform = torch.from_numpy(data[: self.max_samples])
        return waveform, label

    # ------------------------------------------------------------------
    # Key helper
    # ------------------------------------------------------------------

    def key_for(self, idx: int) -> str:
        """Return the canonical key for sample ``idx``."""
        stem = self.samples[idx][0].stem
        return self._key_fn(stem)
