# High-Performance LLM Inference and Serving on a CPU-Only Laptop

Capstone Project for AI Engineering and ML Ops (suggested project #2).

Repository: https://github.com/aksh-j-patel/cpu-llm-serving | Licence: MIT (code only; model weights keep their own Apache-2.0 licence)

Serve a small open-source LLM (Qwen2.5-1.5B-Instruct) on an Intel i5-12450H laptop with no
dedicated GPU, then optimise it with quantisation, thread/core tuning, continuous batching
and prompt caching. Every stage is benchmarked against a naive FastAPI + transformers baseline.

Status: week 2 of 9. Environment, model pinning, STREAM, baseline server, load generator and
the S0 (baseline) and S1 (engine switch) benchmark stages are implemented.
See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md). Submission target: 15 January 2027.

## Quick start

```bash
docker compose build                          # compiles llama.cpp for your CPU (~10-20 min once)
docker compose run --rm bench make smoke      # end-to-end check with the 0.5B model
docker compose run --rm bench make stream     # memory bandwidth
docker compose run --rm bench make bench-quick   # S0 baseline + S1 llama.cpp, reduced grid
```

Full instructions, hardware requirements and troubleshooting: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Reproduction tiers

| Command | What it does | Approx. time |
|---|---|---|
| `make plots` | Rebuild every report figure from the committed raw CSVs | < 1 min, no download |
| `make smoke` | End-to-end check with the 0.5B model | 5–15 min |
| `make bench-tiny` | Every implemented stage with tiny random-weight models (offline check) | ~1 min |
| `make bench-quick` | Reduced benchmark grid on the real models, on your hardware | ~20–30 min |
| `make bench-full` | Full grid as run by the author | ~1.5–2 hours |

## Repository layout

| Path | Purpose |
|---|---|
| `Dockerfile`, `compose.yaml`, `Makefile` | Pinned environment and one-command entry points |
| `requirements.in` / `requirements.txt` | Top-level Python dependencies / hash-locked set |
| `configs/models.yaml` | Which model files make up each set (smoke, quick, full) |
| `configs/models.lock.json` | Exact Hugging Face revisions and file hashes (generated, committed) |
| `configs/experiments.yaml` | Stages (servers to start) and tiers (how much to run) |
| `configs/prompts.json` | Fixed prompt set: 8 short, 8 medium (~250 tokens), 8 long (~1000 tokens) |
| `src/bench/` | Benchmark harness: `python -m bench --help` |
| `src/bench/baseline/` | S0 baseline: transformers FP32 behind an OpenAI-compatible FastAPI server |
| `src/bench/serving/` | Starts, health-checks and stops either server |
| `src/bench/loadgen/` | Closed-loop streaming load generator (TTFT, latency, TPOT, throughput) |
| `src/bench/profiling/` | Resource sampler (memory, CPU, per-core load, clock speed) |
| `src/bench/runner.py` | Runs a stage: models, servers, warm-up, load grid, results, plots |
| `src/bench/analysis/` | Plots generated from `results/raw` |
| `tests/` | Unit tests and an offline engine test with a tiny random-weight model |
| `results/raw/` | One CSV/JSON per run, each with full environment metadata |
| `results/plots/` | Generated figures for the report |
| `docs/` | Project plan, reproduction guide |
| `scripts/make_prompts.py` | How `configs/prompts.json` was generated |
| `report/` | Technical report source and final PDF |

## Hardware

Intel Core i5-12450H (4 performance + 4 efficiency cores, 12 threads), 16 GB LPDDR5,
Intel UHD integrated graphics, 1 TB NVMe SSD, Windows.
