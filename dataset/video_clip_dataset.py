"""PyTorch Dataset for raw MP4 video clips (VideoMAE input).

Each sample's video is decoded with PyAV, num_frames evenly-spaced frames are
sampled, and the frames are preprocessed with VideoMAEImageProcessor into a
float32 pixel_values tensor ready for a VideoMAE model.

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
    pixel_values, label = ds[0]   # pixel_values: (T, 3, 224, 224), label: 0 or 1
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoClipDataset(Dataset):
    """Dataset that reads MP4 clips, samples frames, and returns pixel_values.

    Each ``__getitem__`` returns ``(pixel_values, label)`` where:

    * ``pixel_values`` : float32 tensor ``(T, 3, H, W)`` — preprocessed frames
      in VideoMAE format (T = num_frames, H = W = 224 by default).
    * ``label``        : int, 0 (Truthful) or 1 (Deceptive).

    The returned tensor has shape ``(num_frames, 3, H, W)`` (T-first), which
    matches what ``VideoMAEImageProcessor`` produces.  The collate function in
    ``pretrained_multimodal_dataset`` stacks samples into ``(B, T, 3, H, W)``,
    the format expected by ``VideoMAEModel``.

    Args:
        root_dir:        Root directory containing ``{split}/{Truthful,Deceptive}/``
                         sub-trees of ``.mp4`` files.
        split:           ``"Train"`` or ``"Test"``.
        num_frames:      Number of frames to sample uniformly from each clip.
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
        processor_name: str = "MCG-NJU/videomae-base",
        key_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
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
    # Frame sampling
    # ------------------------------------------------------------------

    def _sample_frames(self, path: Path) -> List[np.ndarray]:
        """Decode video and return num_frames evenly-spaced HWC uint8 arrays."""
        try:
            import av  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "av (PyAV) is required for VideoClipDataset. "
                "Install it with: uv sync --group pretrained"
            ) from exc

        container = av.open(str(path))
        video_stream = container.streams.video[0]

        # Collect all frames (decoded as RGB numpy arrays)
        all_frames: List[np.ndarray] = []
        for frame in container.decode(video_stream):
            all_frames.append(frame.to_ndarray(format="rgb24"))
        container.close()

        n = len(all_frames)
        if n == 0:
            # Fallback: return black frames
            h = video_stream.height or 224
            w = video_stream.width or 224
            return [np.zeros((h, w, 3), dtype=np.uint8)] * self.num_frames

        if n < self.num_frames:
            # Repeat last frame to reach num_frames
            all_frames = all_frames + [all_frames[-1]] * (self.num_frames - n)
            n = self.num_frames

        # Sample num_frames evenly spaced indices
        indices = np.linspace(0, n - 1, self.num_frames, dtype=int)
        return [all_frames[i] for i in indices]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        mp4_path, label = self.samples[idx]

        frames = self._sample_frames(mp4_path)
        processor = self._get_processor()

        # processor accepts list of HWC numpy arrays or PIL images.
        # Returns dict with "pixel_values" of shape (1, T, 3, H, W).
        inputs = processor(frames, return_tensors="pt")
        # Squeeze batch dim: (T, 3, H, W)
        pixel_values: torch.Tensor = inputs["pixel_values"].squeeze(0)

        return pixel_values, label

    # ------------------------------------------------------------------
    # Key helper
    # ------------------------------------------------------------------

    def key_for(self, idx: int) -> str:
        """Return the canonical key for sample ``idx``."""
        stem = self.samples[idx][0].stem
        return self._key_fn(stem)
