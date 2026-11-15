# High-Performance LLM Inference and Serving on a CPU-Only Laptop

Capstone Project for AI Engineering and ML Ops (suggested project #2).

Repository: https://github.com/aksh-j-patel/cpu-llm-serving | Licence: MIT (code only; model weights keep their own Apache-2.0 licence)

Serve a small open-source LLM (Qwen2.5-1.5B-Instruct) on an Intel i5-12450H laptop with no
dedicated GPU, then optimise it with quantisation, thread/core tuning, continuous batching
and prompt caching. Every stage is benchmarked against a naive FastAPI + transformers baseline.

Status: week 1 of 9 (environment, model download, memory-bandwidth measurement, smoke test).
See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md). Submission target: 15 January 2027.

## Quick start

```bash
docker compose build                          # compiles llama.cpp for your CPU (~10-20 min once)
docker compose run --rm bench make smoke      # end-to-end check with the 0.5B model
docker compose run --rm bench make stream     # memory bandwidth
```

Full instructions, hardware requirements and troubleshooting: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Reproduction tiers

| Command | What it does | Approx. time |
|---|---|---|
| `make plots` | Rebuild every report figure from the committed raw CSVs | < 1 min, no download |
| `make smoke` | End-to-end check with the 0.5B model | 5–15 min |
| `make bench-quick` | Reduced benchmark grid on your hardware (from week 2) | ~1 hour |
| `make bench-full` | Full grid as run by the author (from week 2) | several hours |

## Repository layout

| Path | Purpose |
|---|---|
| `Dockerfile`, `compose.yaml`, `Makefile` | Pinned environment and one-command entry points |
| `requirements.in` / `requirements.txt` | Top-level Python dependencies / hash-locked set |
| `configs/models.yaml` | Which model files make up each set (smoke, quick, full) |
| `configs/models.lock.json` | Exact Hugging Face revisions and file hashes (generated, committed) |
| `configs/experiments.yaml` | Parameter grid for every experiment stage |
| `src/bench/` | Benchmark harness: `python -m bench --help` |
| `src/bench/{baseline,serving,loadgen,profiling}/` | Later weeks: baseline server, llama.cpp launch, load generator, profiling |
| `src/bench/analysis/` | Plots generated from `results/raw` |
| `tests/` | Unit tests and an offline engine test with a tiny random-weight model |
| `results/raw/` | One CSV/JSON per run, each with full environment metadata |
| `results/plots/` | Generated figures for the report |
| `docs/` | Project plan, reproduction guide |
| `report/` | Technical report source and final PDF |

## Hardware

Intel Core i5-12450H (4 performance + 4 efficiency cores, 12 threads), 16 GB LPDDR5,
Intel UHD integrated graphics, 1 TB NVMe SSD, Windows.
