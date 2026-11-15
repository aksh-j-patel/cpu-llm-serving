"""Writing result files. Every run produces <name>_<UTC timestamp>.<ext> under results/raw,
with the environment metadata (see bench.sysinfo) stored alongside the measurements."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bench import paths


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _target(name: str, ts: str, ext: str) -> Path:
    paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
    return paths.RAW_DIR / f"{name}_{ts}.{ext}"


def write_json(name: str, payload: dict[str, Any], ts: str | None = None) -> Path:
    out = _target(name, ts or timestamp(), "json")
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def write_csv(name: str, rows: Iterable[dict[str, Any]], ts: str | None = None) -> Path:
    rows = list(rows)
    out = _target(name, ts or timestamp(), "csv")
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return out


def latest(pattern: str) -> Path | None:
    """Most recent file in results/raw matching a glob such as 'stream_*.csv'."""
    files = sorted(paths.RAW_DIR.glob(pattern))
    return files[-1] if files else None
