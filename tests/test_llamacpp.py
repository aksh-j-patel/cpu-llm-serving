import json
from pathlib import Path

import pytest

from bench import llamacpp, paths
from bench.tinymodel import make_tiny_qwen2

# Trimmed from real `llama-bench -o json` output of llama.cpp b12407.
SAMPLE = [
    {"build_commit": "e574a17", "build_number": 12407, "model_type": "qwen2 ?B all F32",
     "model_size": 39192832, "model_n_params": 9798208, "n_threads": 2, "type_k": "f16",
     "n_prompt": 16, "n_gen": 0, "avg_ns": 2971147, "avg_ts": 5385.125677, "stddev_ts": 0.0},
    {"build_commit": "e574a17", "build_number": 12407, "model_type": "qwen2 ?B all F32",
     "n_threads": 2, "n_prompt": 0, "n_gen": 8, "avg_ts": 812.5, "stddev_ts": 3.1},
]


def test_parse_bench_json_labels_tests():
    rows = llamacpp.parse_bench_json("log noise\n" + json.dumps(SAMPLE))
    assert [r["test"] for r in rows] == ["pp16", "tg8"]
    assert rows[0]["avg_ts"] == pytest.approx(5385.125677)
    assert "samples_ns" not in rows[0]


def test_parse_bench_json_requires_array():
    with pytest.raises(llamacpp.LlamaCppError):
        llamacpp.parse_bench_json("no json here")


@pytest.mark.skipif(paths.llama_bin("llama-bench") is None or paths.qwen2_vocab() is None,
                    reason="llama.cpp tools / vocab only available inside the Docker image")
def test_tiny_model_runs_in_llama_bench(tmp_path: Path):
    model = make_tiny_qwen2(tmp_path / "tiny.gguf", paths.qwen2_vocab())
    rows = llamacpp.bench(model, n_prompt=16, n_gen=8, repetitions=1)
    assert {r["test"] for r in rows} == {"pp16", "tg8"}
    assert all(r["avg_ts"] > 0 for r in rows)
