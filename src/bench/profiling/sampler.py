"""Background sampler for the server process: memory, CPU, per-core load and clock speed.

Sampling every 0.5 s costs well under 1% of one core. Inside Docker Desktop the per-core
numbers describe the VM's virtual CPUs, not physical performance/efficiency cores.
"""

from __future__ import annotations

import threading
import time
from statistics import fmean
from typing import Any

import psutil


class ResourceSampler:
    def __init__(self, pid: int, interval_s: float = 0.5):
        self.proc = psutil.Process(pid)
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)
        self.t0 = time.perf_counter()
        self.label = ""

    def _procs(self) -> list[psutil.Process]:
        try:
            return [self.proc] + self.proc.children(recursive=True)
        except psutil.Error:
            return []

    def _loop(self) -> None:
        procs = self._procs()
        for p in procs:
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass
        psutil.cpu_percent(percpu=True)
        while not self._stop.wait(self.interval_s):
            rss = cpu = 0.0
            for p in self._procs():
                try:
                    rss += p.memory_info().rss
                    cpu += p.cpu_percent(None)
                except psutil.Error:
                    continue
            cores = psutil.cpu_percent(percpu=True)
            freq = psutil.cpu_freq()
            self.samples.append({
                "t": round(time.perf_counter() - self.t0, 3),
                "label": self.label,
                "rss_mb": round(rss / 2**20, 1),
                "proc_cpu_percent": round(cpu, 1),           # 100 = one full core
                "system_cpu_percent": round(fmean(cores), 1) if cores else None,
                "busy_cores": sum(1 for c in cores if c >= 50),
                "cpu_mhz": round(freq.current) if freq and freq.current else None,
                "per_core": " ".join(f"{c:.0f}" for c in cores),
            })

    def start(self) -> "ResourceSampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def window(self, start: float, end: float) -> dict[str, Any]:
        samples = list(self.samples)
        rows = [s for s in samples if start <= s["t"] <= end]
        if not rows:   # run shorter than one interval: use the latest reading taken before it ended
            rows = [s for s in samples if s["t"] <= end][-1:]
        if not rows:
            return {}
        return {
            "peak_rss_mb": max(r["rss_mb"] for r in rows),
            "mean_proc_cpu_percent": round(fmean(r["proc_cpu_percent"] for r in rows), 1),
            "mean_system_cpu_percent": round(fmean(r["system_cpu_percent"] for r in rows
                                                   if r["system_cpu_percent"] is not None), 1),
            "mean_busy_cores": round(fmean(r["busy_cores"] for r in rows), 2),
            "mean_cpu_mhz": round(fmean(r["cpu_mhz"] for r in rows if r["cpu_mhz"]), 0)
            if any(r["cpu_mhz"] for r in rows) else None,
            "samples": len(rows),
        }

    def peak_rss_mb(self) -> float | None:
        return max((s["rss_mb"] for s in self.samples), default=None)
