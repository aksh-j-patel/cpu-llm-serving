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
# Categorical slots 1-3 (validated all-pairs for colour-vision deficiency); more series are faceted.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")


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


def _summary_medians(csv_path: Path, metric: str) -> dict[tuple[str, str], dict[int, tuple[float, float, float]]]:
    from bench.analysis.metrics import medians

    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["users"] = int(r["users"])
    series: dict[tuple[str, str], dict[int, tuple[float, float, float]]] = {}
    for m in medians(rows, [metric]):
        if m[metric] is None:
            continue
        series.setdefault((m["server"], m["prompt_class"]), {})[m["users"]] = (
            m[metric], m[f"{metric}_min"], m[f"{metric}_max"])
    return series


def _line_chart(series, title: str, ylabel: str, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank = {"short": 0, "medium": 1, "long": 2}
    keys = sorted(series, key=lambda k: (k[0], rank.get(k[1], 9), k[1]))
    groups = [keys[i:i + len(SERIES)] for i in range(0, len(keys), len(SERIES))]   # facet beyond 3
    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 4), dpi=150, facecolor=SURFACE,
                             squeeze=False)
    all_users = sorted({u for v in series.values() for u in v})
    for ax, group in zip(axes[0], groups):
        _style(ax)
        for i, key in enumerate(group):
            pts = series[key]
            us = sorted(pts)
            y = [pts[u][0] for u in us]
            lo = [pts[u][0] - pts[u][1] for u in us]
            hi = [pts[u][2] - pts[u][0] for u in us]
            label = key[1] if len({k[0] for k in keys}) == 1 else f"{key[0]} / {key[1]}"
            ax.errorbar(us, y, yerr=[lo, hi], color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=1, capsize=0, label=label)
            ax.annotate(label, (us[-1], y[-1]), textcoords="offset points", xytext=(6, 0), va="center",
                        fontsize=8, color=TEXT_SECONDARY)
        ax.set_xticks(all_users)
        ax.margins(x=0.12)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("concurrent users", color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
        if len(group) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="best")
    names = sorted({k[0] for k in keys})
    fig.suptitle(f"{title} - {', '.join(names)} (median of repeats; bar = min-max)", color=TEXT_PRIMARY,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def concurrency_plots(summary_csv: Path) -> list[Path]:
    """Throughput and p95 latency against concurrent users, one line per server/prompt class.
    Two separate charts: the measures have different units (never a dual axis)."""
    tag = summary_csv.name.split("_summary_")[0]
    made = []
    for metric, title, ylabel, suffix in (
            ("throughput_tok_s", "Output throughput", "output tokens / s (all users)", "throughput"),
            ("e2e_p95_s", "Request latency, p95", "seconds per request", "latency_p95")):
        series = _summary_medians(summary_csv, metric)
        if series:
            made.append(_line_chart(series, title, ylabel, paths.PLOTS_DIR / f"{tag}_{suffix}.png"))
    return made


def make_all(peak_gbs: float | None = None) -> list[Path]:
    made = []
    src = results.latest("stream_*.csv")
    if src:
        made.append(stream_bandwidth(src, paths.PLOTS_DIR / "stream_bandwidth.png", peak_gbs))
    latest_by_tag: dict[str, Path] = {}
    for f in sorted(paths.RAW_DIR.glob("*_summary_*.csv")):
        latest_by_tag[f.name.split("_summary_")[0]] = f
    for f in latest_by_tag.values():
        made += concurrency_plots(f)
    return made
