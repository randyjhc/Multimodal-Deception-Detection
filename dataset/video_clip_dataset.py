"""PyTorch Dataset for raw MP4 video clips (VideoMAE input).

Each sample's video is decoded with PyAV, split into non-overlapping windows of
``num_frames`` frames each, and each window is preprocessed with
``VideoMAEImageProcessor`` into a float32 pixel_values tensor ready for VideoMAE.

Directory layout expected::

    {root_dir}/{split}/{Truthful,Deceptive}/*.mp4

Usage::

    from dataset.video_clip_dataset import VideoClipDataset
    from dataset.multimodal_dataset import openface_ur_lying_key

    ds = VideoClipDataset(
        "dataset/UR_LYING_Deception_Dataset/clips_raw",
        split="Train",
        key_fn=openface_ur_lying_key,
    )
    pixel_values, label = ds[0]   # pixel_values: (N_windows, T, 3, 224, 224), label: 0 or 1
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoClipDataset(Dataset):
    """Dataset that reads MP4 clips and returns non-overlapping sliding windows.

    Each ``__getitem__`` returns ``(pixel_values, label)`` where:

    * ``pixel_values`` : float32 tensor ``(N_windows, T, 3, H, W)`` — per-window
      preprocessed frames in VideoMAE format (T = num_frames, H = W = 224).
    * ``label``        : int, 0 (Truthful) or 1 (Deceptive).

    All frames are decoded, then split into windows of ``num_frames`` with a
    configurable ``stride``.  When ``stride == num_frames`` (default) windows
    are non-overlapping; smaller strides produce overlapping windows.
    Incomplete trailing frames are discarded.  Videos shorter than one window
    are padded with the last frame.

    Args:
        root_dir:        Root directory containing ``{split}/{Truthful,Deceptive}/``
                         sub-trees of ``.mp4`` files.
        split:           ``"Train"`` or ``"Test"``.
        num_frames:      Window size — number of frames per VideoMAE input clip.
        stride:          Step between window start frames.  Defaults to
                         ``num_frames`` (non-overlapping).
        processor_name:  HuggingFace model/processor identifier used to build
                         ``VideoMAEImageProcessor`` (default ``"MCG-NJU/videomae-base"``).
        key_fn:          Optional function mapping a filename stem to a canonical
                         key used by :meth:`key_for`.  Defaults to identity.
    """

    # Class-level processor cache so multiple dataset instances share one copy.
    _processor_cache: Dict[str, Any] = {}

    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        num_frames: int = 16,
        stride: Optional[int] = None,
        processor_name: str = "MCG-NJU/videomae-base",
        key_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.stride = stride if stride is not None else num_frames
        self.processor_name = processor_name
        self._key_fn = key_fn or (lambda s: s)

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Video clip split directory not found: {split_dir}"
            )

        self.samples: List[Tuple[Path, int]] = []
        for label_name, label in [("Truthful", 0), ("Deceptive", 1)]:
            label_dir = split_dir / label_name
            if label_dir.exists():
                for mp4_path in sorted(label_dir.glob("*.mp4")):
                    self.samples.append((mp4_path, label))

    # ------------------------------------------------------------------
    # Lazy processor loading
    # ------------------------------------------------------------------

    def _get_processor(self) -> Any:
        if self.processor_name not in VideoClipDataset._processor_cache:
            try:
                from transformers import VideoMAEImageProcessor  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for VideoClipDataset."
                ) from exc
            VideoClipDataset._processor_cache[self.processor_name] = (
                VideoMAEImageProcessor.from_pretrained(self.processor_name)
            )
        return VideoClipDataset._processor_cache[self.processor_name]

    # ------------------------------------------------------------------
    # Frame decoding
    # ------------------------------------------------------------------

    def _decode_all_frames(self, path: Path) -> List[np.ndarray]:
        """Decode all frames of a video and return HWC uint8 RGB arrays."""
        try:
            import av  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "av (PyAV) is required for VideoClipDataset. "
                "Install it with: uv sync --group pretrained"
            ) from exc

        container = av.open(str(path))
        video_stream = container.streams.video[0]

        all_frames: List[np.ndarray] = []
        for frame in container.decode(video_stream):
            all_frames.append(frame.to_ndarray(format="rgb24"))
        container.close()

        if len(all_frames) == 0:
            h = video_stream.height or 224
            w = video_stream.width or 224
            all_frames = [np.zeros((h, w, 3), dtype=np.uint8)]

        return all_frames

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        mp4_path, label = self.samples[idx]

        all_frames = self._decode_all_frames(mp4_path)
        window_size = self.num_frames

        # Pad short videos to at least one full window
        if len(all_frames) < window_size:
            all_frames = all_frames + [all_frames[-1]] * (window_size - len(all_frames))

        processor = self._get_processor()
        n = len(all_frames)

        windows: List[torch.Tensor] = []
        for start in range(0, n - window_size + 1, self.stride):
            w_frames = all_frames[start : start + window_size]
            # processor returns {"pixel_values": (1, T, 3, H, W)}
            inp = processor(w_frames, return_tensors="pt")
            windows.append(inp["pixel_values"].squeeze(0))  # (T, 3, H, W)

        pixel_values: torch.Tensor = torch.stack(windows)  # (N_windows, T, 3, H, W)
        return pixel_values, label

    # ------------------------------------------------------------------
    # Key helper
    # ------------------------------------------------------------------

    def key_for(self, idx: int) -> str:
        """Return the canonical key for sample ``idx``."""
        stem = self.samples[idx][0].stem
        return self._key_fn(stem)
