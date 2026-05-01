import argparse
from pathlib import Path

import av


def main(root: str) -> None:
    paths = sorted(Path(root).rglob("*.mp4"))
    if not paths:
        print(f"No .mp4 files found in {root}")
        return

    durations: list[float] = []
    fpss: list[float] = []
    for p in paths:
        with av.open(str(p)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            duration = container.duration / 1e6  # microseconds -> seconds
            durations.append(duration)
            fpss.append(fps)

    print(f"Found {len(paths)} videos\n")
    print(
        f"Duration (s): avg={sum(durations)/len(durations):.2f}  "
        f"min={min(durations):.2f}  max={max(durations):.2f}"
    )
    print(
        f"FPS:          avg={sum(fpss)/len(fpss):.2f}  "
        f"min={min(fpss):.2f}  max={max(fpss):.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="dataset/UR_LYING_Deception_Dataset/clips_raw"
    )
    main(parser.parse_args().root)
