"""Figures for the report, generated only from CSV files in results/raw (so `make plots`
reproduces them exactly without re-running any benchmark)."""

from __future__ import annotations

import csv
from pathlib import Path

from bench import paths, results

# Reference palette (light mode): surface, ink, and categorical slot 1.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def stream_bandwidth(csv_path: Path, out: Path, peak_gbs: float | None = None) -> Path:
    """Best Triad bandwidth per thread count, with the spread across repeats."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_threads: dict[int, list[float]] = {}
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["kernel"] == "Triad":
                by_threads.setdefault(int(row["threads"]), []).append(float(row["best_rate_gbs"]))
    threads = sorted(by_threads)
    best = [max(by_threads[t]) for t in threads]
    low = [b - min(by_threads[t]) for t, b in zip(threads, best)]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.errorbar(threads, best, yerr=[low, [0] * len(best)], color=SERIES_1, linewidth=2,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                ecolor=SERIES_1, elinewidth=1, capsize=0)
    i_max = max(range(len(best)), key=best.__getitem__)
    ax.annotate(f"{best[i_max]:.1f} GB/s", (threads[i_max], best[i_max]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color=TEXT_PRIMARY)
    if peak_gbs:
        ax.axhline(peak_gbs, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"theoretical peak {peak_gbs:.0f} GB/s", (threads[0], peak_gbs),
                    textcoords="offset points", xytext=(0, 5), fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(threads)
    ax.set_ylim(0, max(best + [peak_gbs or 0]) * 1.18)
    ax.set_xlabel("OpenMP threads", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("GB/s (10^9 bytes/s)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title("STREAM Triad memory bandwidth (best of repeats; bar = spread)",
                 color=TEXT_PRIMARY, fontsize=11, loc="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def make_all(peak_gbs: float | None = None) -> list[Path]:
    made = []
    src = results.latest("stream_*.csv")
    if src:
        made.append(stream_bandwidth(src, paths.PLOTS_DIR / "stream_bandwidth.png", peak_gbs))
    return made
