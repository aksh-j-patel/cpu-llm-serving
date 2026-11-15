"""Closed-loop load generation with exact client-side timing.

N simulated users each send a request, wait for the complete streamed answer, then send the
next one immediately (closed loop). Timing is measured on the client from the moment the
request is sent:

  ttft       time to the first streamed chunk that carries text
  e2e        time to the end of the stream
  tpot       time per output token after the first = (e2e - ttft) / (completion_tokens - 1)

Token counts come from the server's usage report (exact), falling back to counting chunks.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from bench import paths

PROMPTS_FILE = paths.CONFIG_DIR / "prompts.json"


@dataclass
class RequestResult:
    user: int
    seq: int
    prompt_id: str
    ok: bool = False
    error: str = ""
    t_start: float = 0.0               # seconds since the run started
    ttft_s: float | None = None
    e2e_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    chunks: int = 0
    finish_reason: str | None = None
    server_timings: dict[str, Any] = field(default_factory=dict)

    @property
    def tpot_s(self) -> float | None:
        if self.ttft_s is None or self.e2e_s is None or not self.completion_tokens or self.completion_tokens < 2:
            return None
        return (self.e2e_s - self.ttft_s) / (self.completion_tokens - 1)

    def row(self) -> dict[str, Any]:
        d = asdict(self)
        timings = d.pop("server_timings") or {}
        d["tpot_s"] = self.tpot_s
        for k in ("prompt_ms", "predicted_ms", "prompt_per_second", "predicted_per_second"):
            d[f"server_{k}"] = timings.get(k)
        return d


def load_prompts(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    data = json.loads((path or PROMPTS_FILE).read_text(encoding="utf-8"))
    return data["classes"]


def build_payload(prompt: str, max_tokens: int, temperature: float, seed: int | None,
                  ignore_eos: bool, cache_prompt: bool) -> dict[str, Any]:
    return {
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        # llama.cpp extensions (ignored by servers that don't know them):
        "ignore_eos": ignore_eos,        # fixed output length, so runs are comparable
        "cache_prompt": cache_prompt,    # prompt (KV) cache reuse; off except in stage S5
    }


def parse_sse_line(line: str) -> dict[str, Any] | None | str:
    """Return the decoded JSON of a `data:` line, the string 'DONE', or None to skip."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return "DONE"
    try:
        return json.loads(data)
    except ValueError:
        return None


async def one_request(client: httpx.AsyncClient, url: str, payload: dict[str, Any],
                      result: RequestResult, run_t0: float) -> RequestResult:
    t0 = time.perf_counter()
    result.t_start = t0 - run_t0
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:300]
                result.error = f"HTTP {resp.status_code}: {body}"
                return result
            async for line in resp.aiter_lines():
                msg = parse_sse_line(line)
                if msg is None:
                    continue
                if msg == "DONE":
                    break
                if msg.get("usage"):
                    result.prompt_tokens = msg["usage"].get("prompt_tokens")
                    result.completion_tokens = msg["usage"].get("completion_tokens")
                if msg.get("timings"):
                    result.server_timings = msg["timings"]
                for choice in msg.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        result.chunks += 1
                        if result.ttft_s is None:
                            result.ttft_s = time.perf_counter() - t0
                    if choice.get("finish_reason"):
                        result.finish_reason = choice["finish_reason"]
        result.e2e_s = time.perf_counter() - t0
        if result.completion_tokens is None:
            result.completion_tokens = result.chunks
        result.ok = result.ttft_s is not None
        if not result.ok:
            result.error = "stream ended without any content"
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


@dataclass
class LoadSpec:
    prompt_class: str
    users: int
    requests_per_user: int
    max_tokens: int = 128
    temperature: float = 0.0
    seed: int | None = 42
    ignore_eos: bool = True
    cache_prompt: bool = False
    timeout_s: float = 1800.0


async def _run(base_url: str, spec: LoadSpec, prompts: list[dict[str, Any]],
               offset: int) -> tuple[list[RequestResult], float]:
    url = f"{base_url}/v1/chat/completions"
    limits = httpx.Limits(max_connections=spec.users + 2, max_keepalive_connections=spec.users + 2)
    timeout = httpx.Timeout(spec.timeout_s, connect=30.0)
    results: list[RequestResult] = []
    run_t0 = time.perf_counter()

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        async def user(u: int) -> None:
            for k in range(spec.requests_per_user):
                idx = offset + u * spec.requests_per_user + k     # different prompt per request
                prompt = prompts[idx % len(prompts)]
                payload = build_payload(prompt["text"], spec.max_tokens, spec.temperature, spec.seed,
                                        spec.ignore_eos, spec.cache_prompt)
                res = await one_request(client, url, payload, RequestResult(u, k, prompt["id"]), run_t0)
                results.append(res)

        await asyncio.gather(*(user(u) for u in range(spec.users)))
    return results, time.perf_counter() - run_t0


def run(base_url: str, spec: LoadSpec, prompts: list[dict[str, Any]],
        offset: int = 0) -> tuple[list[RequestResult], float]:
    """Run one closed-loop load test; returns per-request results and wall-clock seconds."""
    return asyncio.run(_run(base_url, spec, prompts, offset))


def warmup(base_url: str, prompts: list[dict[str, Any]], n: int = 1, max_tokens: int = 16) -> None:
    spec = LoadSpec(prompt_class="warmup", users=1, requests_per_user=n, max_tokens=max_tokens)
    results, _ = run(base_url, spec, prompts)
    bad = [r.error for r in results if not r.ok]
    if bad:
        raise RuntimeError(f"warm-up request failed: {bad[0]}")
