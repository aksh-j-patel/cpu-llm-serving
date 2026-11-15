"""OpenAI-compatible HTTP server for the transformers baseline.

Implements the subset of `/v1/chat/completions` that llama-server also implements, with the
same streaming format (Server-Sent Events, a usage chunk when `stream_options.include_usage`
is set, and llama.cpp-style `timings`), so one load generator drives both engines.

Requests are served strictly one at a time: that sequential behaviour is what the baseline
is meant to show under concurrent load.

    python -m bench.baseline.server --model models/qwen2.5-1.5b-hf --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from bench.baseline.engine import GenParams, GenStats, HFEngine


def create_app(engine: Any, model_name: str = "baseline") -> FastAPI:
    app = FastAPI(title="transformers baseline", docs_url=None, redoc_url=None)
    lock = asyncio.Lock()
    state = {"served": 0, "queued": 0}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "served": state["served"], "queued": state["queued"]}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    @app.get("/info")
    async def info() -> dict[str, Any]:
        return engine.info()

    def parse(body: dict[str, Any]) -> tuple[list[dict[str, str]], GenParams, bool, bool]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "'messages' must be a non-empty list")
        max_tokens = body.get("max_tokens") or body.get("n_predict") or 128
        params = GenParams(max_tokens=int(max_tokens), temperature=float(body.get("temperature") or 0.0),
                           seed=body.get("seed"), ignore_eos=bool(body.get("ignore_eos", False)))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
        return messages, params, bool(body.get("stream", False)), include_usage

    def chunk(cid: str, created: int, delta: dict[str, Any], finish: str | None = None) -> str:
        payload = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(payload)}\n\n"

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        messages, params, stream, include_usage = parse(await request.json())
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        if not stream:
            state["queued"] += 1
            async with lock:
                state["queued"] -= 1
                stats = GenStats()
                text = "".join([piece async for piece in engine.stream(messages, params, stats)])
                state["served"] += 1
            return JSONResponse({
                "id": cid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": stats.finish_reason}],
                "usage": {"prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.completion_tokens,
                          "total_tokens": stats.prompt_tokens + stats.completion_tokens},
                "timings": stats.timings()})

        async def events():
            state["queued"] += 1
            async with lock:
                state["queued"] -= 1
                stats = GenStats()
                yield chunk(cid, created, {"role": "assistant", "content": None})
                async for piece in engine.stream(messages, params, stats):
                    yield chunk(cid, created, {"content": piece})
                final = json.loads(chunk(cid, created, {}, stats.finish_reason)[6:])
                final["timings"] = stats.timings()
                yield f"data: {json.dumps(final)}\n\n"
                if include_usage:
                    usage = {"id": cid, "object": "chat.completion.chunk", "created": created,
                             "model": model_name, "choices": [],
                             "usage": {"prompt_tokens": stats.prompt_tokens,
                                       "completion_tokens": stats.completion_tokens,
                                       "total_tokens": stats.prompt_tokens + stats.completion_tokens}}
                    yield f"data: {json.dumps(usage)}\n\n"
                yield "data: [DONE]\n\n"
                state["served"] += 1

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="transformers baseline server (OpenAI-compatible)")
    ap.add_argument("--model", required=True, help="Hugging Face model directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads (default: PyTorch's)")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--name", default="baseline")
    args = ap.parse_args(argv)

    import uvicorn

    engine = HFEngine(args.model, threads=args.threads, dtype=args.dtype)   # load before listening
    print(f"model loaded: {json.dumps(engine.info())}", flush=True)
    uvicorn.run(create_app(engine, args.name), host=args.host, port=args.port, log_level="warning",
                access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
