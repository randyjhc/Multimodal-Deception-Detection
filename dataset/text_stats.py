"""
Compute text statistics for Deceptive vs. Truthful transcriptions.

Metrics per clip, then averaged by class:
  - Word Count
  - Words Per Sentence (WPS)
  - Type-Token Ratio (TTR = unique tokens / total tokens)
  - Filler Count (uh, um, ah, er, hmm and extended variants)
  - Ellipsis Count (..., ...., unicode …)
"""

import re
from pathlib import Path

from scipy import stats

TRANSCRIPTION_DIR = Path("dataset/Real-life_Deception_Detection_2016/Transcription")

# Spoken fillers: uh/uhhh, um/umm, ah/ahhh, eh/ehh, er/err, hm/hmm, amm/ammm
_FILLER_RE = re.compile(r"\b(uh+|um+|ah+|eh+|er+|hm+|am{2,})\b", re.IGNORECASE)
# Ellipses: two-or-more dots, or unicode ellipsis character
_ELLIPSIS_RE = re.compile(r"\.{2,}|…")


def tokenize(text: str) -> list[str]:
    """Lowercase and extract word tokens, stripping surrounding punctuation."""
    raw = re.findall(r"\b[\w'-]+\b", text.lower())
    # Filter out tokens that are purely punctuation/digits after stripping
    return [t for t in raw if re.search(r"[a-z]", t)]


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation (. ! ?), treating ... as one boundary."""
    # Collapse ellipses and multiple dots into a single period first
    text = re.sub(r"\.{2,}", ".", text)
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def compute_stats(filepath: Path) -> dict:
    text = filepath.read_text(encoding="utf-8").strip()

    words = tokenize(text)
    sentences = split_sentences(text)

    word_count = len(words)
    num_sentences = max(len(sentences), 1)  # avoid division by zero
    wps = word_count / num_sentences
    ttr = len(set(words)) / word_count if word_count > 0 else 0.0
    fillers = len(_FILLER_RE.findall(text))
    ellipses = len(_ELLIPSIS_RE.findall(text))

    return {
        "word_count": word_count,
        "wps": wps,
        "ttr": ttr,
        "fillers": fillers,
        "ellipses": ellipses,
    }


def summarize(stats_list: list[dict]) -> dict:
    keys = stats_list[0].keys()
    summary = {}
    for k in keys:
        values = [s[k] for s in stats_list]
        summary[k] = {
            "mean": sum(values) / len(values),
            "std": (
                sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)
            )
            ** 0.5,
        }
    return summary


def normalize_for_radar(
    deceptive_stats: list[dict],
    truthful_stats: list[dict],
    metric_keys: list[str],
) -> dict[str, dict[str, float]]:
    """Min-max normalize class means over the combined per-clip value range.

    Returns a dict mapping each metric key to {"deceptive": float, "truthful": float}
    where each value is in [0, 1] relative to the observed min/max across all clips.
    """
    all_stats = deceptive_stats + truthful_stats
    result = {}
    for key in metric_keys:
        values = [s[key] for s in all_stats]
        lo, hi = min(values), max(values)
        span = hi - lo if hi != lo else 1.0
        d_mean = sum(s[key] for s in deceptive_stats) / len(deceptive_stats)
        t_mean = sum(s[key] for s in truthful_stats) / len(truthful_stats)
        result[key] = {
            "deceptive": (d_mean - lo) / span,
            "truthful": (t_mean - lo) / span,
        }
    return result


def main():
    deceptive_dir = TRANSCRIPTION_DIR / "Deceptive"
    truthful_dir = TRANSCRIPTION_DIR / "Truthful"

    deceptive_files = sorted(deceptive_dir.glob("*.txt"))
    truthful_files = sorted(truthful_dir.glob("*.txt"))

    deceptive_stats = [compute_stats(f) for f in deceptive_files]
    truthful_stats = [compute_stats(f) for f in truthful_files]

    d_summary = summarize(deceptive_stats)
    t_summary = summarize(truthful_stats)

    print("\nTranscription Text Statistics")
    print(
        f"Deceptive clips: {len(deceptive_files)}  |  Truthful clips: {len(truthful_files)}"
    )

    metrics = [
        ("word_count", "Word Count"),
        ("wps", "Words Per Sentence"),
        ("ttr", "TTR (unique/total words)"),
        ("fillers", "Filler Count"),
        ("ellipses", "Ellipsis Count"),
    ]

    print(f"\n  {'=' * 75}")
    print(
        f"  {'Metric':<25} {'D Mean':>8}  {'D Std':>8}  {'T Mean':>8}  {'T Std':>8}  {'p-value':>9}"
    )
    print(f"  {'-'*75}")
    for metric, label in metrics:
        dm, ds = d_summary[metric]["mean"], d_summary[metric]["std"]
        tm, ts = t_summary[metric]["mean"], t_summary[metric]["std"]
        d_vals = [s[metric] for s in deceptive_stats]
        t_vals = [s[metric] for s in truthful_stats]
        _, p = stats.ttest_ind(d_vals, t_vals, equal_var=False)
        print(
            f"  {label:<25} {dm:>8.2f}  {ds:>8.2f}  {tm:>8.2f}  {ts:>8.2f}  {p:>9.4f}"
        )

    metric_keys = [m for m, _ in metrics]
    normalized = normalize_for_radar(deceptive_stats, truthful_stats, metric_keys)

    print("\n  Normalized means for radar chart (min-max over all clips)")
    print(f"  {'=' * 49}")
    print(f"  {'Metric':<25} {'Deceptive':>10}  {'Truthful':>10}")
    print(f"  {'='*49}")
    for metric, label in metrics:
        d_norm = normalized[metric]["deceptive"]
        t_norm = normalized[metric]["truthful"]
        print(f"  {label:<25} {d_norm:>10.4f}  {t_norm:>10.4f}")
    print()


if __name__ == "__main__":
    main()
