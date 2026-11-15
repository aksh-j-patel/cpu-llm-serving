# High-Performance LLM Inference and Serving on a CPU-Only Laptop

Capstone Project for AI Engineering and ML Ops (suggested project #2).

Repository: https://github.com/aksh-j-patel/cpu-llm-serving | Licence: MIT (code only; model weights keep their own Apache-2.0 licence)

Serve a small open-source LLM (Qwen2.5-1.5B-Instruct) on an Intel i5-12450H laptop with no
dedicated GPU, then optimise it with quantisation, thread/core tuning, continuous batching
and prompt caching. Every stage is benchmarked against a naive FastAPI + transformers baseline.

Status: planning. No code yet. See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md).
Submission target: 15 January 2027.

## Reproducing the results (planned)

Everything runs inside one Docker image. Absolute numbers depend on your hardware; the relative
trends should hold on any machine (see `docs/EXPECTED_RESULTS.md`, to be written).

| Command | What it does | Approx. time |
|---|---|---|
| `make plots` | Rebuild every report figure from the committed raw CSVs | < 1 min, no download |
| `make smoke` | End-to-end check with the 0.5B model | < 15 min |
| `make bench-quick` | Reduced benchmark grid on your hardware | ~1 hour |
| `make bench-full` | Full grid as run by the author | several hours |

Full instructions will be in `docs/REPRODUCE.md`.

## Repository layout

| Path | Purpose |
|---|---|
| `docs/PROJECT_PLAN.md` | Objective, experiment stages, metrics, protocol, timeline |
| `configs/experiments.yaml` | Parameter grid for every experiment stage |
| `src/baseline/` | S0: FastAPI + Hugging Face transformers server (FP32, sequential) |
| `src/serving/` | Launch wrappers for llama.cpp `llama-server` per configuration |
| `src/loadgen/` | Async streaming load generator (TTFT, latency, tokens/s) |
| `src/profiling/` | psutil sampler, PyTorch profiler and py-spy helpers |
| `src/analysis/` | Scripts that turn `results/raw` CSVs into plots and tables |
| `scripts/` | One entry point per stage, model download, system-info capture |
| `models/` | GGUF and Hugging Face weights (not committed) |
| `results/raw/` | One CSV per run, with environment metadata |
| `results/profiles/` | Flame graphs and profiler output |
| `results/plots/` | Generated figures for the report |
| `report/` | Technical report source and final PDF |
| `assets/` | Course brief (capstone list and expectations) |

## Hardware

Intel Core i5-12450H (4 performance + 4 efficiency cores, 12 threads), 16 GB LPDDR5,
Intel UHD integrated graphics, 1 TB NVMe SSD, Windows.
