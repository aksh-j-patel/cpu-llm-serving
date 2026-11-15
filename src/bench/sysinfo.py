"""Environment capture. Every result file embeds this so numbers can be traced to the exact
hardware, software and code version that produced them. No hostname or user name is
recorded, because results are committed to a public repository."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil

from bench import __version__, paths

KEY_PACKAGES = (
    "torch", "transformers", "huggingface-hub", "tokenizers", "fastapi", "uvicorn",
    "httpx", "numpy", "pandas", "psutil",
)
INTERESTING_FLAGS = ("sse4_2", "avx", "avx2", "fma", "f16c", "avx_vnni", "avx512f", "avx512_vnni",
                     "avx512_bf16", "amx_tile", "neon", "asimd", "sve")


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (p.stdout + "\n" + p.stderr).strip()
    return out or None


def cpu_model() -> str:
    if sys.platform.startswith("linux"):
        info = _read("/proc/cpuinfo") or ""
        m = re.search(r"^model name\s*:\s*(.+)$", info, re.M)
        if m:
            return m.group(1).strip()
    elif sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    return platform.processor() or "unknown"


def cpu_flags() -> list[str]:
    info = _read("/proc/cpuinfo") or ""
    m = re.search(r"^(flags|Features)\s*:\s*(.+)$", info, re.M)
    present = set(m.group(2).split()) if m else set()
    return [f for f in INTERESTING_FLAGS if f in present]


def hybrid_topology() -> dict[str, str] | None:
    """Intel hybrid CPUs on bare-metal Linux expose performance and efficiency cores separately.
    Inside a VM (WSL2, Docker Desktop) this information is usually hidden."""
    core, atom = _read("/sys/devices/cpu_core/cpus"), _read("/sys/devices/cpu_atom/cpus")
    if core or atom:
        return {"performance_cpus": core or "", "efficiency_cpus": atom or ""}
    return None


def in_container() -> bool:
    if os.environ.get("BENCH_IN_CONTAINER") == "1" or Path("/.dockerenv").exists():
        return True
    cg = _read("/proc/1/cgroup") or ""
    return any(s in cg for s in ("docker", "containerd", "kubepods"))


def cgroup_limits() -> dict[str, str] | None:
    cpu, mem = _read("/sys/fs/cgroup/cpu.max"), _read("/sys/fs/cgroup/memory.max")
    if cpu is None and mem is None:
        return None
    return {"cpu.max": cpu or "", "memory.max": mem or ""}


def llama_cpp_version() -> str | None:
    version_file = os.environ.get("LLAMA_CPP_VERSION_FILE", "/opt/llama.cpp/VERSION")
    tag = _read(version_file)
    binary = paths.llama_bin("llama-server") or paths.llama_bin("llama-bench")
    reported = None
    if binary:
        out = _run([str(binary), "--version"])
        if out:
            m = re.search(r"version:\s*(.+)", out)
            reported = m.group(1).strip() if m else out.splitlines()[-1]
    if tag is None and reported is None:
        return None
    return " | ".join(x for x in (tag, reported) if x)


def git_state() -> dict[str, Any]:
    env_commit = os.environ.get("BENCH_GIT_COMMIT")
    if env_commit:
        return {"commit": env_commit, "dirty": None}
    base = ["git", "-c", "safe.directory=*", "-C", str(paths.REPO_ROOT)]
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        rev = subprocess.run(base + ["rev-parse", "--verify", "-q", "HEAD"], capture_output=True,
                             text=True, timeout=10, env=env)
        commit = rev.stdout.strip() if rev.returncode == 0 else None
        status = subprocess.run(base + ["status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=10, env=env).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip()) if commit else None}


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in KEY_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def torch_threads() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    return {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}


def collect() -> dict[str, Any]:
    freq = psutil.cpu_freq()
    vm = psutil.virtual_memory()
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bench_version": __version__,
        "git": git_state(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "wsl_kernel": "microsoft" in platform.release().lower(),
        },
        "container": {"in_container": in_container(), "cgroup_limits": cgroup_limits()},
        "cpu": {
            "model": cpu_model(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
            "usable_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "max_mhz": round(freq.max) if freq and freq.max else None,
            "current_mhz": round(freq.current) if freq and freq.current else None,
            "flags": cpu_flags(),
            "hybrid": hybrid_topology(),
        },
        "memory": {
            "total_gb": round(vm.total / 2**30, 2),
            "available_gb": round(vm.available / 2**30, 2),
        },
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "packages": package_versions(),
        "torch_threads": torch_threads(),
        "llama_cpp": llama_cpp_version(),
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
                                                "MKL_NUM_THREADS") if os.environ.get(k)},
    }


def summary_lines(info: dict[str, Any]) -> list[str]:
    cpu, mem, osd = info["cpu"], info["memory"], info["os"]
    where = "container" if info["container"]["in_container"] else "native"
    if osd["wsl_kernel"]:
        where += " (WSL2 VM)"
    return [
        f"CPU      : {cpu['model']} | {cpu['physical_cores']} cores / {cpu['logical_cpus']} threads"
        f" (usable: {cpu['usable_cpus']})",
        f"Flags    : {', '.join(cpu['flags']) or 'n/a'}",
        f"Hybrid   : {cpu['hybrid'] or 'not visible'}",
        f"Memory   : {mem['total_gb']} GB total, {mem['available_gb']} GB available",
        f"OS       : {osd['system']} {osd['release']} [{where}]",
        f"Python   : {info['python']['version']} | torch {info['packages'].get('torch')}"
        f" | transformers {info['packages'].get('transformers')}",
        f"llama.cpp: {info['llama_cpp'] or 'not found'}",
        f"Git      : {info['git']['commit'] or 'n/a'}{' (dirty)' if info['git']['dirty'] else ''}",
    ]
