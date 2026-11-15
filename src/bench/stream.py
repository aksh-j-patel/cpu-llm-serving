"""STREAM memory-bandwidth measurement (McCalpin's STREAM, built in the Docker image).

Single-stream LLM decoding reads roughly the whole model from RAM for every generated token,
so sustained memory bandwidth sets an upper bound on tokens per second. This module measures
that bandwidth across thread counts so later results can be compared against it.

STREAM reports MB/s with MB = 10^6 bytes; values here are converted to GB/s (10^9 bytes/s).
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

import psutil

from bench import paths

KERNELS = ("Copy", "Scale", "Add", "Triad")
_LINE = re.compile(r"^(Copy|Scale|Add|Triad):\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", re.M)
_ARRAY = re.compile(r"Array size = (\d+)")
_COUNTED = re.compile(r"Number of Threads counted = (\d+)")


class StreamError(RuntimeError):
    pass


def parse(output: str) -> dict[str, Any]:
    kernels = {}
    for m in _LINE.finditer(output):
        kernels[m.group(1)] = {
            "best_rate_gbs": float(m.group(2)) / 1000.0,
            "avg_time_s": float(m.group(3)),
            "min_time_s": float(m.group(4)),
            "max_time_s": float(m.group(5)),
        }
    if set(kernels) != set(KERNELS):
        raise StreamError("could not find all four STREAM kernels in the output")
    array = _ARRAY.search(output)
    counted = _COUNTED.search(output)
    return {
        "kernels": kernels,
        "array_elements": int(array.group(1)) if array else None,
        "threads_counted": int(counted.group(1)) if counted else None,
        "validates": "Solution Validates" in output,
    }


def default_threads() -> list[int]:
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False) or logical
    counts = {1, physical, logical}
    t = 2
    while t < logical:
        counts.add(t)
        t *= 2
    return sorted(counts)


def run_once(threads: int, timeout: float = 300) -> dict[str, Any]:
    binary = paths.stream_bin()
    if binary is None:
        raise StreamError("STREAM binary not found (set STREAM_BIN or run inside the Docker image)")
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    p = subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=timeout)
    if p.returncode != 0:
        raise StreamError(f"STREAM exited with {p.returncode}: {p.stderr.strip()[:300]}")
    return parse(p.stdout)


def run(threads: list[int] | None = None, repeats: int = 3) -> list[dict[str, Any]]:
    rows = []
    for t in threads or default_threads():
        for r in range(1, repeats + 1):
            res = run_once(t)
            for k in KERNELS:
                rows.append({"threads": t, "repeat": r, "kernel": k,
                             **res["kernels"][k],
                             "threads_counted": res["threads_counted"],
                             "array_elements": res["array_elements"],
                             "validates": res["validates"]})
            triad = res["kernels"]["Triad"]["best_rate_gbs"]
            print(f"  threads={t:<3d} repeat={r}  Triad {triad:7.2f} GB/s"
                  f"{'' if res['validates'] else '  (VALIDATION FAILED)'}")
    return rows


def best_by_threads(rows: list[dict[str, Any]], kernel: str = "Triad") -> dict[int, float]:
    best: dict[int, float] = {}
    for r in rows:
        if r["kernel"] == kernel:
            best[r["threads"]] = max(best.get(r["threads"], 0.0), float(r["best_rate_gbs"]))
    return dict(sorted(best.items()))
