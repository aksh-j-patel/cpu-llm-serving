"""Per-run summary statistics from per-request load-generator results."""

from __future__ import annotations

from statistics import fmean, median
from typing import Any, Iterable

import numpy as np


def pct(values: Iterable[float | None], q: float) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.percentile(v, q)) if v else None


def _mean(values: Iterable[float | None]) -> float | None:
    v = [x for x in values if x is not None]
    return fmean(v) if v else None


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def summarize(rows: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    """rows: RequestResult.row() dicts from one closed-loop run."""
    ok = [r for r in rows if r["ok"]]
    out_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    tpots = [r["tpot_s"] for r in ok]
    return {
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "wall_s": _r(wall_s, 3),
        "output_tokens": out_tokens,
        "throughput_tok_s": _r(out_tokens / wall_s if wall_s else None, 3),
        "requests_per_s": _r(len(ok) / wall_s if wall_s else None, 4),
        "prompt_tokens_mean": _r(_mean(r["prompt_tokens"] for r in ok), 1),
        "ttft_p50_s": _r(pct((r["ttft_s"] for r in ok), 50)),
        "ttft_p95_s": _r(pct((r["ttft_s"] for r in ok), 95)),
        "e2e_p50_s": _r(pct((r["e2e_s"] for r in ok), 50)),
        "e2e_p95_s": _r(pct((r["e2e_s"] for r in ok), 95)),
        "e2e_p99_s": _r(pct((r["e2e_s"] for r in ok), 99)),
        "tpot_mean_ms": _r(None if not any(tpots) else 1000 * _mean(tpots), 2),
        "tpot_p95_ms": _r(None if not any(tpots) else 1000 * pct(tpots, 95), 2),
        "decode_tok_s_per_request": _r(_mean(1 / t for t in tpots if t), 3),
        "server_prompt_tok_s": _r(_mean(r.get("server_prompt_per_second") for r in ok), 2),
        "server_decode_tok_s": _r(_mean(r.get("server_predicted_per_second") for r in ok), 2),
    }


GROUP_KEYS = ("server", "prompt_class", "users")


def medians(summary_rows: list[dict[str, Any]], metrics: Iterable[str]) -> list[dict[str, Any]]:
    """Median (and min/max) of each metric across repetitions, per server/prompt class/users."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in summary_rows:
        groups.setdefault(tuple(r[k] for k in GROUP_KEYS), []).append(r)
    out = []
    for key, rs in groups.items():
        row = dict(zip(GROUP_KEYS, key))
        row["repetitions"] = len(rs)
        for m in metrics:
            vals = [float(r[m]) for r in rs if r.get(m) not in (None, "")]
            row[m] = median(vals) if vals else None
            row[f"{m}_min"] = min(vals) if vals else None
            row[f"{m}_max"] = max(vals) if vals else None
        out.append(row)
    return out
