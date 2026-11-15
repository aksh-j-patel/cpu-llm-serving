import asyncio

import httpx
import pytest

from bench.analysis import metrics
from bench.baseline.engine import GenStats
from bench.baseline.server import create_app
from bench.loadgen import client


def test_parse_sse_line():
    assert client.parse_sse_line("data: [DONE]") == "DONE"
    assert client.parse_sse_line(": keep-alive") is None
    assert client.parse_sse_line("") is None
    assert client.parse_sse_line('data: {"a": 1}') == {"a": 1}
    assert client.parse_sse_line("data: {broken") is None


def test_payload_has_fixed_length_and_no_cache():
    p = client.build_payload("hi", 32, 0.0, 42, ignore_eos=True, cache_prompt=False)
    assert p["stream"] and p["stream_options"]["include_usage"]
    assert p["ignore_eos"] is True and p["cache_prompt"] is False and p["max_tokens"] == 32


def test_prompt_set_shape():
    prompts = client.load_prompts()
    assert set(prompts) == {"short", "medium", "long"}
    for items in prompts.values():
        assert len(items) == 8 and len({p["id"] for p in items}) == 8
    assert max(p["tokens"] for p in prompts["short"]) < min(p["tokens"] for p in prompts["medium"])
    assert max(p["tokens"] for p in prompts["medium"]) < min(p["tokens"] for p in prompts["long"])


def _row(ttft, e2e, tokens, ok=True):
    r = client.RequestResult(0, 0, "x", ok=ok, ttft_s=ttft, e2e_s=e2e, completion_tokens=tokens,
                             prompt_tokens=10)
    return r.row()


def test_tpot_and_summary():
    rows = [_row(0.5, 2.5, 21), _row(1.0, 3.0, 21), _row(None, None, None, ok=False)]
    assert rows[0]["tpot_s"] == pytest.approx(0.1)
    s = metrics.summarize(rows, wall_s=4.0)
    assert s["requests"] == 3 and s["errors"] == 1
    assert s["output_tokens"] == 42 and s["throughput_tok_s"] == pytest.approx(10.5)
    assert s["ttft_p50_s"] == pytest.approx(0.75) and s["tpot_mean_ms"] == pytest.approx(100.0)
    assert s["decode_tok_s_per_request"] == pytest.approx(10.0)


def test_medians_across_repetitions():
    rows = [{"server": "a", "prompt_class": "short", "users": 1, "x": v} for v in (1.0, 5.0, 3.0)]
    (m,) = metrics.medians(rows, ["x"])
    assert (m["x"], m["x_min"], m["x_max"], m["repetitions"]) == (3.0, 1.0, 5.0, 3)


class FakeEngine:
    """Stands in for HFEngine: emits fixed pieces, fills stats like the real engine."""

    def info(self):
        return {"engine": "fake"}

    async def stream(self, messages, params, stats: GenStats):
        stats.prompt_tokens = 7
        for i in range(params.max_tokens):
            await asyncio.sleep(0)
            yield f"t{i} "
        stats.completion_tokens = params.max_tokens
        stats.prompt_ms, stats.predicted_ms = 5.0, 50.0


def test_baseline_server_speaks_the_loadgen_protocol():
    app = create_app(FakeEngine(), "fake")

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            payload = client.build_payload("hello", 5, 0.0, 1, True, False)
            res = await client.one_request(c, "http://test/v1/chat/completions", payload,
                                           client.RequestResult(0, 0, "p"), 0.0)
            plain = await c.post("/v1/chat/completions", json={**payload, "stream": False})
            health = await c.get("/health")
        return res, plain.json(), health.json()

    res, plain, health = asyncio.run(go())
    assert res.ok and res.error == ""
    assert (res.prompt_tokens, res.completion_tokens, res.chunks) == (7, 5, 5)
    assert res.finish_reason == "length"
    assert res.server_timings["predicted_n"] == 5
    assert plain["choices"][0]["message"]["content"] == "t0 t1 t2 t3 t4 "
    assert plain["usage"]["completion_tokens"] == 5
    assert health["served"] == 2


def test_baseline_server_rejects_bad_request():
    app = create_app(FakeEngine(), "fake")

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            return await c.post("/v1/chat/completions", json={"messages": []})

    assert asyncio.run(go()).status_code == 400
