"""Filesystem locations and external binaries. Every location can be overridden by an
environment variable so the same code runs in Docker, natively on Windows, and in CI."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("BENCH_REPO_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = REPO_ROOT / "configs"
MODELS_DIR = Path(os.environ.get("BENCH_MODELS_DIR", REPO_ROOT / "models"))
RESULTS_DIR = Path(os.environ.get("BENCH_RESULTS_DIR", REPO_ROOT / "results"))
RAW_DIR = RESULTS_DIR / "raw"
PLOTS_DIR = RESULTS_DIR / "plots"

MODELS_CONFIG = CONFIG_DIR / "models.yaml"
MODELS_LOCK = CONFIG_DIR / "models.lock.json"

_EXE = ".exe" if sys.platform == "win32" else ""


def _which(name: str, env_dir: str | None = None, env_file: str | None = None) -> Path | None:
    if env_file and os.environ.get(env_file):
        p = Path(os.environ[env_file])
        return p if p.exists() else None
    if env_dir and os.environ.get(env_dir):
        p = Path(os.environ[env_dir]) / f"{name}{_EXE}"
        if p.exists():
            return p
    found = shutil.which(name)
    return Path(found) if found else None


def llama_bin(name: str) -> Path | None:
    """Locate a llama.cpp tool (llama-server, llama-bench, ...): $LLAMA_CPP_BIN_DIR, then PATH."""
    return _which(name, env_dir="LLAMA_CPP_BIN_DIR")


def stream_bin() -> Path | None:
    """Locate the STREAM memory-bandwidth binary: $STREAM_BIN, then PATH."""
    return _which("stream", env_file="STREAM_BIN")


def qwen2_vocab() -> Path | None:
    """Tokenizer-only GGUF shipped with llama.cpp, used to build the tiny offline test model."""
    p = os.environ.get("LLAMA_VOCAB_QWEN2")
    return Path(p) if p and Path(p).exists() else None
