"""End-to-end: run stages S0 and S1 at the 'tiny' tier (random-weight models, real servers,
real load generator). Runs inside the Docker image; skipped where the tools are missing."""

import csv
import importlib.util

import pytest

from bench import paths, runner

needs_tools = pytest.mark.skipif(
    paths.llama_bin("llama-server") is None or paths.qwen2_vocab() is None
    or importlib.util.find_spec("torch") is None,
    reason="needs llama-server, the bundled Qwen2 vocab and PyTorch (available in the Docker image)")


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("bench")
    for name, value in (("RESULTS_DIR", root / "results"), ("RAW_DIR", root / "results" / "raw"),
                        ("PLOTS_DIR", root / "results" / "plots"), ("MODELS_DIR", root / "models")):
        monkeypatch_module.setattr(paths, name, value)
    return root


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_resolve_applies_tier_overrides():
    wl, servers = runner.resolve("s0_baseline", "smoke")
    assert wl["max_tokens"] == 16 and wl["repetitions"] == 1
    assert servers[0]["model"] == "qwen2.5-0.5b-hf"
    wl_full, servers_full = runner.resolve("s0_baseline", "full")
    assert wl_full["repetitions"] == 5 and servers_full[0]["model"] == "qwen2.5-1.5b-hf"
    with pytest.raises(runner.RunError):
        runner.resolve("nope", "full")


@needs_tools
@pytest.mark.parametrize("stage", ["s0_baseline", "s1_engine"])
def test_stage_runs_end_to_end(sandbox, stage):
    assert runner.run_stage(stage, "tiny", plot=True) == 0
    summary = sorted(paths.RAW_DIR.glob(f"{stage}_tiny_summary_*.csv"))[-1]
    rows = list(csv.DictReader(summary.open()))
    wl, _ = runner.resolve(stage, "tiny")
    assert len(rows) == len(wl["prompt_classes"]) * len(wl["users"]) * wl["repetitions"]
    assert all(r["errors"] == "0" and float(r["throughput_tok_s"]) > 0 for r in rows)
    requests = sorted(paths.RAW_DIR.glob(f"{stage}_tiny_requests_*.csv"))[-1]
    assert all(r["completion_tokens"] == str(wl["max_tokens"]) for r in csv.DictReader(requests.open()))
    assert list(paths.PLOTS_DIR.glob(f"{stage}_tiny_*.png"))
