"""End-to-end smoke test: proves the environment, the engines and the model files all work.

Week 1 scope: environment check, offline engine check with a tiny random model, model download
and verification, a short llama-bench run and a short transformers generation on the 0.5B
model. Later weeks add the baseline server and load generator to this pipeline.
"""

from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from bench import llamacpp, models, paths, results, sysinfo
from bench.tinymodel import make_tiny_qwen2

SMOKE_SET = "smoke"
GGUF_MODEL = "qwen2.5-0.5b-q4_k_m"
HF_MODEL = "qwen2.5-0.5b-hf"
PROMPT = "Explain in two sentences why memory bandwidth limits LLM inference on CPUs."


class Skip(Exception):
    pass


def check_environment() -> dict[str, Any]:
    found = {name: str(paths.llama_bin(name) or "") for name in
             ("llama-server", "llama-bench", "llama-perplexity")}
    missing = [k for k, v in found.items() if not v]
    if missing:
        raise RuntimeError(f"missing llama.cpp tools: {', '.join(missing)}")
    import torch  # noqa: F401  (fails the step if PyTorch is absent)
    return {"llama_cpp_tools": found, "stream": str(paths.stream_bin() or "not found")}


def check_tiny_engine() -> dict[str, Any]:
    vocab = paths.qwen2_vocab()
    if vocab is None:
        raise Skip("LLAMA_VOCAB_QWEN2 not set (only available inside the Docker image)")
    with tempfile.TemporaryDirectory() as tmp:
        model = make_tiny_qwen2(Path(tmp) / "tiny-qwen2.gguf", vocab)
        rows = llamacpp.bench(model, n_prompt=16, n_gen=8, repetitions=1)
    return {"llama_bench": rows}


def download_models() -> dict[str, Any]:
    specs = models.download(SMOKE_SET)
    return {"files": len(specs), "bytes": sum(s.size for s in specs)}


def run_llama_bench() -> dict[str, Any]:
    rows = llamacpp.bench(models.model_path(GGUF_MODEL), n_prompt=64, n_gen=32, repetitions=2)
    return {"model": GGUF_MODEL, "llama_bench": rows,
            "tokens_per_s": {r["test"]: round(r.get("avg_ts") or 0, 2) for r in rows}}


def run_transformers(max_new_tokens: int = 32) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = models.model_path(HF_MODEL)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    load_s = time.perf_counter() - t0
    inputs = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True)
    n_in = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        t1 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                             do_sample=False)
        gen_s = time.perf_counter() - t1
    n_new = out.shape[-1] - n_in
    return {"model": HF_MODEL, "dtype": "float32", "torch_threads": torch.get_num_threads(),
            "load_s": round(load_s, 2), "prompt_tokens": n_in, "new_tokens": n_new,
            "generate_s": round(gen_s, 3), "tokens_per_s": round(n_new / gen_s, 2),
            "sample_output": tok.decode(out[0, n_in:], skip_special_tokens=True)[:200]}


def run(skip_download: bool = False) -> int:
    info = sysinfo.collect()
    print("\n".join(sysinfo.summary_lines(info)), "\n")
    # (name, function, skipped by --skip-download, steps it depends on)
    steps: list[tuple[str, Callable[[], dict[str, Any]], bool, tuple[str, ...]]] = [
        ("environment", check_environment, False, ()),
        ("tiny-engine", check_tiny_engine, False, ()),
        ("models", download_models, True, ()),
        ("llama.cpp-0.5b", run_llama_bench, True, ("models",)),
        ("transformers-0.5b", run_transformers, True, ("models",)),
    ]
    report: dict[str, Any] = {"sysinfo": info, "steps": {}}
    for name, fn, offline_skip, needs in steps:
        start = time.perf_counter()
        blocked = [n for n in needs if report["steps"][n]["status"] != "PASS"]
        print(f"--> {name}")
        if offline_skip and skip_download:
            status, detail = "SKIP", {"reason": "--skip-download"}
        elif blocked:
            status, detail = "SKIP", {"reason": f"needs {', '.join(blocked)}"}
        else:
            try:
                detail, status = fn(), "PASS"
            except Skip as e:
                status, detail = "SKIP", {"reason": str(e)}
            except Exception as e:  # report and continue to the summary
                status, detail = "FAIL", {"error": f"{type(e).__name__}: {e}",
                                          "traceback": traceback.format_exc()[-1500:]}
        detail["seconds"] = round(time.perf_counter() - start, 1)
        report["steps"][name] = {"status": status, **detail}
        print(f"    {status} ({detail['seconds']} s)"
              + (f": {detail.get('error') or detail.get('reason')}" if status != "PASS" else ""))
    failed = any(s["status"] == "FAIL" for s in report["steps"].values())

    out = results.write_json("smoke", report)
    print("\nSmoke test summary")
    for name, step in report["steps"].items():
        extra = ""
        if name.startswith("llama.cpp") and step["status"] == "PASS":
            extra = "  " + ", ".join(f"{k}: {v} tok/s" for k, v in step["tokens_per_s"].items())
        if name.startswith("transformers") and step["status"] == "PASS":
            extra = f"  generation: {step['tokens_per_s']} tok/s (FP32, {step['torch_threads']} threads)"
        print(f"  {step['status']:4s}  {name}{extra}")
    print(f"\nWritten: {out.relative_to(paths.REPO_ROOT) if out.is_relative_to(paths.REPO_ROOT) else out}")
    return 1 if failed else 0
