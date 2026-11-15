"""Thin wrappers around llama.cpp command-line tools."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from bench import paths

BENCH_FIELDS = ("model_type", "model_size", "model_n_params", "n_threads", "n_batch", "n_ubatch",
                "type_k", "type_v", "n_prompt", "n_gen", "avg_ns", "stddev_ns", "avg_ts",
                "stddev_ts", "build_commit", "build_number", "cpu_info")


class LlamaCppError(RuntimeError):
    pass


def require(name: str) -> Path:
    p = paths.llama_bin(name)
    if p is None:
        raise LlamaCppError(f"{name} not found (set LLAMA_CPP_BIN_DIR or run inside the Docker image)")
    return p


def parse_bench_json(text: str) -> list[dict[str, Any]]:
    """llama-bench -o json prints a JSON array (one object per test) on stdout."""
    start = text.find("[")
    if start < 0:
        raise LlamaCppError("no JSON array in llama-bench output")
    data = json.loads(text[start:])
    rows = []
    for item in data:
        row = {k: item.get(k) for k in BENCH_FIELDS if k in item}
        n_prompt, n_gen = item.get("n_prompt", 0), item.get("n_gen", 0)
        row["test"] = f"pp{n_prompt}" if n_prompt and not n_gen else (
            f"tg{n_gen}" if n_gen and not n_prompt else f"pp{n_prompt}+tg{n_gen}")
        rows.append(row)
    return rows


def bench(model: Path, n_prompt: int = 64, n_gen: int = 32, repetitions: int = 2,
          threads: int | None = None, extra: list[str] | None = None,
          timeout: float = 900) -> list[dict[str, Any]]:
    cmd = [str(require("llama-bench")), "-m", str(model), "-p", str(n_prompt), "-n", str(n_gen),
           "-r", str(repetitions), "-o", "json"]
    if threads:
        cmd += ["-t", str(threads)]
    cmd += extra or []
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise LlamaCppError(f"llama-bench failed ({p.returncode}): {p.stderr.strip()[-600:]}")
    return parse_bench_json(p.stdout)
