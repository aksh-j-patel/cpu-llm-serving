"""Launch a server under test as a subprocess and wait until it is ready.

Both engines expose the same OpenAI-compatible API, so the rest of the harness only needs
the base URL and the process id (for resource sampling).
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import httpx

from bench import paths


class LaunchError(RuntimeError):
    pass


@dataclass
class ServerSpec:
    name: str
    engine: str                          # "baseline" | "llama.cpp"
    model_path: str
    model_key: str = ""
    threads: int | None = None           # None = engine default
    parallel: int = 1                    # llama.cpp slots (concurrent sequences)
    ctx_per_slot: int = 2048
    cont_batching: bool = True
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    ready_timeout_s: float = 900.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def command(spec: ServerSpec, port: int) -> list[str]:
    if spec.engine == "baseline":
        cmd = [sys.executable, "-m", "bench.baseline.server", "--model", spec.model_path,
               "--port", str(port), "--name", spec.name]
        if spec.threads:
            cmd += ["--threads", str(spec.threads)]
    elif spec.engine == "llama.cpp":
        binary = paths.llama_bin("llama-server")
        if binary is None:
            raise LaunchError("llama-server not found (set LLAMA_CPP_BIN_DIR or use the Docker image)")
        cmd = [str(binary), "-m", spec.model_path, "--host", "127.0.0.1", "--port", str(port),
               "-np", str(spec.parallel), "-c", str(spec.ctx_per_slot * spec.parallel),
               "--alias", spec.name]
        if spec.threads:
            cmd += ["-t", str(spec.threads)]
        if not spec.cont_batching:
            cmd += ["--no-cont-batching"]
    else:
        raise LaunchError(f"unknown engine '{spec.engine}'")
    return cmd + list(spec.extra_args)


@dataclass
class RunningServer:
    spec: ServerSpec
    base_url: str
    process: subprocess.Popen
    command: list[str]
    log_path: Path
    ready_s: float

    @property
    def pid(self) -> int:
        return self.process.pid

    def log_tail(self, n: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""


def _wait_ready(url: str, proc: subprocess.Popen, timeout: float, log_path: Path) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc.poll() is not None:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
            raise LaunchError(f"server exited with code {proc.returncode} while starting:\n{tail}")
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return time.perf_counter() - t0
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise LaunchError(f"server not ready after {timeout:.0f} s")


def stop(proc: subprocess.Popen, grace_s: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


@contextmanager
def running(spec: ServerSpec, log_dir: Path | None = None) -> Iterator[RunningServer]:
    port = free_port()
    cmd = command(spec, port)
    log_dir = log_dir or Path(tempfile.gettempdir())
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"server_{spec.name}_{port}.log"
    env = {**os.environ, **spec.env}
    env.setdefault("PYTHONPATH", str(paths.REPO_ROOT / "src"))
    print(f"  starting {spec.name}: {shlex.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    url = f"http://127.0.0.1:{port}"
    try:
        ready_s = _wait_ready(url, proc, spec.ready_timeout_s, log_path)
        print(f"  {spec.name} ready in {ready_s:.1f} s (pid {proc.pid})")
        yield RunningServer(spec, url, proc, cmd, log_path, ready_s)
    finally:
        stop(proc)
