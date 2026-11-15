# Project Plan: High-Performance LLM Inference and Serving on a CPU-Only Laptop

Capstone Project for AI Engineering and ML Ops (suggested project #2)
Author: Aksh Patel | Drafted: 15 Nov 2026 | Status: draft v2.1 | Submission target: 15 Jan 2027
Repository (public, proposed name): https://github.com/aksh-j-patel/cpu-llm-serving

## 1. Objective

Deploy a small open-source LLM behind an HTTP API on a laptop with no dedicated GPU, then
make it measurably faster and lighter using HPC techniques. The project reports the gain in
throughput, latency, memory use and concurrent-user capacity at every step, and packages
everything so evaluators can re-run it on their own machines.

**Research question:** how much serving performance can be recovered on commodity CPU
hardware through quantisation, thread/core tuning, request batching and caching, and which
hardware limit (compute, memory bandwidth, or core heterogeneity) dominates at each stage?

## 2. Hardware and constraints

| Component | Spec | Consequence for the project |
|---|---|---|
| CPU | Intel Core i5-12450H, 8 cores / 12 threads (4 performance cores with hyper-threading + 4 efficiency cores) | Thread scaling will be non-linear; core pinning is a real optimisation |
| RAM | 16 GB LPDDR5 (Task Manager reports 4800 MT/s) | Caps model size; memory bandwidth limits token generation speed |
| GPU | Intel UHD integrated only | All inference runs on CPU |
| Disk | 1 TB NVMe SSD | Model load time is not a concern |
| OS | Windows; benchmarks run in Docker (WSL2 backend), see section 8 | Docker's VM gets half of host RAM by default; raise it in `.wslconfig` for the FP32 baseline |

Memory-bandwidth ceiling: if the memory runs at 4800 MT/s on a 128-bit bus, theoretical peak is
about 77 GB/s. Week 1 measures the real figure with the STREAM benchmark rather than assuming it.

## 3. Model and software stack

| Item | Choice | Why |
|---|---|---|
| Model | Qwen2.5-1.5B-Instruct (Apache-2.0) | Small enough for 16 GB RAM; official GGUF files exist at every precision |
| Model files | `Qwen/Qwen2.5-1.5B-Instruct-GGUF`: fp16 (about 3.6 GB), q8_0, q4_k_m (about 1.1 GB) | One source for all precisions keeps the comparison fair |
| Smoke-test / draft model | Qwen2.5-0.5B-Instruct (same tokenizer family) | Fast evaluator smoke test; draft model for speculative decoding (S7) |
| Baseline engine | Hugging Face `transformers` + PyTorch (CPU wheel), FP32, behind FastAPI | Represents the "naive" deployment most tutorials produce |
| Optimised engine | llama.cpp `llama-server`, pinned to one release tag and built from source | Quantised GGUF, thread control, parallel slots, continuous batching, prompt caching, speculative decoding |
| Load generator | Custom `asyncio` + `httpx` client with streaming | Exact time-to-first-token and per-request latency |
| Orchestration | One Python CLI (`python -m bench <stage>`) plus a Makefile inside the container | Same code path on Windows, Linux and macOS |
| Profiling | PyTorch profiler, py-spy, psutil, llama.cpp timings, `llama-bench`, STREAM | See section 6 |
| Plots | matplotlib, generated from committed CSVs by script | Reproducible figures |

Fallback: if the FP32 baseline (about 6 GB of weights) does not fit comfortably in free
memory, use Qwen2.5-0.5B-Instruct for every stage instead.

## 4. Experiment stages

Each stage changes one thing, so the report can attribute the gain to a single technique.

| Stage | Change | HPC concept demonstrated | Main metric |
|---|---|---|---|
| S0 Baseline | FastAPI + transformers, FP32, default threads, one request at a time | Reference point | Tokens/s, p95 latency, peak RAM |
| S1 Engine | Same model as F16 GGUF in llama-server, 1 slot, default threads | Vectorised (AVX2) kernels, memory-mapped weights | Tokens/s |
| S2 Quantisation | F16 vs Q8_0 vs Q4_K_M, with perplexity check | Memory bandwidth as the bottleneck (roofline-style analysis) | Tokens/s, RAM, quality |
| S3 Threading | Sweep `-t` over 1, 2, 4, 6, 8, 12; performance cores only vs all cores (native Windows) | Parallel scaling, speedup and efficiency, heterogeneous cores | Speedup curve |
| S4 Concurrency | `-np` 1, 2, 4, 8 with continuous batching on/off; 1 to 16 simulated users | Batching, throughput vs latency trade-off | Total tokens/s, p95 latency |
| S5 Caching | Shared system prompt with prompt cache on/off | Avoiding recomputation (KV cache reuse) | Time to first token |
| S6 Memory | Context per slot; KV-cache precision (f16 vs q8_0); mmap vs mlock | Memory-efficient serving | Peak RSS, users served within a RAM budget |
| S7 Speculative decoding (stretch) | 0.5B draft model, and draft-free n-gram mode | Trading cheap compute for fewer memory-bound passes | Tokens/s, acceptance rate |

S2, S3 and S4 alone satisfy "at least two HPC optimisation techniques". S5 to S7 are stretch goals.
S7 is a genuine open question: with only a 3x size gap between draft and main model on a CPU, it
may not help, and a negative result with an explanation is acceptable.

Optional bonus: repeat S0 and the best configuration on a free Kaggle or Colab GPU.

## 5. Metrics

| Metric | Definition | How measured |
|---|---|---|
| Generation throughput | Output tokens per second, per request and summed across users | Load generator; cross-checked with server timings |
| Time to first token (TTFT) | Request sent to first streamed token | Load generator |
| End-to-end latency | p50, p95, p99 per request | Load generator |
| Prompt processing speed | Prompt tokens per second | llama.cpp timings |
| Peak memory | Resident set size of the server process | psutil sampler, 0.5 s interval |
| CPU utilisation | Per-core utilisation during a run | psutil sampler |
| Speedup and efficiency | S(n) = T(1)/T(n); E(n) = S(n)/n | Derived from S3 |
| Achieved bandwidth | Model bytes read per token x tokens/s, compared with STREAM | Derived from S2 |
| Output quality | Perplexity on a fixed WikiText-2 sample, plus 20 fixed prompts compared by eye | `llama-perplexity` and a saved prompt set |

## 6. Profiling plan (bottleneck identification)

1. **Machine characterisation (week 1):** STREAM memory bandwidth (OpenMP build) and core
   topology, so later results can be compared against hardware limits.
2. **Baseline (S0):** PyTorch profiler for operator-level time; py-spy flame graph of the
   FastAPI process; per-core utilisation trace showing cores idle under sequential serving.
3. **Engine level (S1 onward):** llama.cpp per-request timings (load, prompt eval, eval) and
   `llama-bench` for controlled single-stream numbers.
4. **System level:** psutil sampler for RSS and per-core load; CPU frequency logging to detect
   thermal throttling.
5. **Memory-bandwidth evidence:** single-stream decode reads roughly the whole model per token,
   so tokens/s x model size approximates achieved bandwidth. If that lands near the STREAM figure
   across precisions, decode is bandwidth-bound.

## 7. Benchmark protocol (for fair, repeatable numbers)

- Reboot before a benchmark session; close browsers, IDEs and chat apps.
- Laptop plugged in, fixed "Best performance" power plan, on a hard surface.
- Fixed workload: 3 prompt-length classes (short, medium, long), `max_tokens` = 128, temperature 0, fixed seed.
- One warm-up run discarded, then 5 measured repetitions; report median and interquartile range.
- 60-second cool-down between configurations; log CPU frequency to flag throttled runs.
- Closed-loop load: N users each send the next request as soon as the previous one finishes.
- Every result CSV carries environment metadata: CPU model, core counts, RAM, OS, container or native,
  llama.cpp tag, model file SHA-256, Python package versions, git commit of this repo.

## 8. Reproducibility for evaluators

Evaluators will run the code on hardware that differs from the author's laptop, so absolute numbers
will differ. The package is designed so that (a) the author's exact figures can be regenerated, and
(b) the same trends can be observed on any reasonable x86-64 or ARM machine.

**Three tiers, each a single command inside the container:**

| Tier | Command (planned) | What it does | Target time / download |
|---|---|---|---|
| 1. Regenerate plots | `make plots` | Rebuilds every figure and table in the report from the committed raw CSVs | Under 1 minute, no model download |
| 2. Smoke test | `make smoke` | Downloads the 0.5B model, runs baseline and optimised servers briefly, checks the pipeline end to end | Under 15 minutes on a 4-core machine, roughly 1.5 GB |
| 3. Quick benchmark | `make bench-quick` | Reduced grid of S0 to S4 on the evaluator's hardware, producing the same plots | About 1 hour, roughly 5 GB |
| (4. Full benchmark) | `make bench-full` | Entire grid as run by the author | Several hours, roughly 10 GB |

**What is pinned:**
- Base image by digest; Python version; dependency lock file with hashes (pip-tools).
- PyTorch CPU-only wheel (avoids multi-gigabyte CUDA downloads).
- llama.cpp release tag, compiled from source in the Dockerfile so it targets the evaluator's CPU.
- Model files by Hugging Face revision hash, verified by SHA-256 after download.

**Supporting files:**
- `docs/REPRODUCE.md`: prerequisites (Docker, disk, RAM), Windows `.wslconfig` note, the four tiers, troubleshooting.
- `docs/EXPECTED_RESULTS.md`: which relative trends should hold on any machine (for example, Q4_K_M
  faster than F16; total throughput rising with slots until cores saturate) and which are
  machine-specific (absolute tokens/s, the performance-core vs efficiency-core result).
- GitHub Actions workflow: builds the image and runs the smoke test on every push, which proves the
  package works on a clean machine. The repository is public, so standard runners are 4 vCPU / 16 GB
  and free; the smoke test should still stay under about 15 minutes. Model files are cached between runs.
- Public-repository hygiene: no model weights, secrets or course materials committed; MIT licence.

**Environment decision:** all headline benchmarks run inside the same Docker image evaluators use.
Two experiments also run natively on Windows with the same llama.cpp tag: performance-core pinning
(the container's VM hides which cores are performance cores) and the best configuration, which
measures the virtualisation overhead as a finding in its own right.

## 9. Deliverables mapped to course expectations

| Expectation | Deliverable |
|---|---|
| Working baseline | `src/baseline/` FastAPI + transformers server, with S0 results |
| Bottlenecks identified with profiling tools | STREAM result, flame graph, profiler table, per-core utilisation plots (`results/profiles/`) |
| At least two HPC optimisation techniques | Quantisation, thread/core tuning, continuous batching (S2 to S4), plus stretch stages |
| Quantified improvements | CSV results and an ablation table, baseline to best configuration |
| Reproducible code | Docker image, lock file, pinned models, four-tier reproduction, CI smoke test |
| Performance plots | Speedup curve, throughput vs concurrency, latency percentiles, memory and quality vs precision, achieved vs peak bandwidth |
| Concise technical report | `report/` source exported to PDF (format proposed in section 11) |

## 10. Timeline (16 Nov 2026 to 15 Jan 2027, 9 weeks)

| Week | Dates | Work | Exit criterion |
|---|---|---|---|
| 1 | 16 to 22 Nov | Dockerfile, lock file, model download with hashes, sysinfo capture, STREAM | `make smoke` skeleton builds; bandwidth measured |
| 2 | 23 to 29 Nov | S0 baseline server, load generator, psutil sampler | Baseline benchmark reproducible with one command |
| 3 | 30 Nov to 6 Dec | Profile S0; S1 engine switch; GitHub Actions smoke test | **M1: minimum viable capstone** (baseline + profiling + reproducible package) |
| 4 | 7 to 13 Dec | S2 quantisation, perplexity, bandwidth analysis | Precision comparison and bandwidth plot |
| 5 | 14 to 20 Dec | S3 thread sweep in Docker; native Windows core-pinning run | **M2: two required techniques complete** |
| 6 | 21 to 27 Dec | S4 concurrency and batching (holiday week, lighter load) | Throughput-vs-users and latency plots |
| 7 | 28 Dec to 3 Jan | S5 caching, S6 memory, S7 speculative if time allows; buffer | **M3: all experiments frozen** |
| 8 | 4 to 10 Jan | Full clean re-run; `REPRODUCE.md`, `EXPECTED_RESULTS.md`; report draft; reproduction test on a second machine | **M4: code freeze** |
| 9 | 11 to 15 Jan | Report final edits and figures; final tagged release; submit by 14 Jan | Submitted with a day of buffer |

Everything after M2 adds depth; if time runs short, cut S7, then S6, then S5, in that order.

## 11. Report

The course does not specify a format. Proposal: an 8 to 10 page PDF written in Markdown inside
`report/`, so it is versioned with the code and its figures come straight from `results/plots/`.

Outline: abstract; introduction and research question; hardware and setup; baseline; profiling
and bottlenecks; one subsection per optimisation; results and ablation table; reproducibility;
limitations; conclusion; appendix with exact commands.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Thermal throttling distorts results | Cool-downs, frequency logging, repeat runs, report medians |
| Memory pressure causes paging | Reboot, close apps, raise Docker/WSL2 memory limit; fall back to the 0.5B model |
| Docker/WSL2 overhead or hidden core types skew results | Native Windows runs for pinning and best config; overhead reported |
| Evaluator hardware too weak or very different | Tier 1 needs no compute; tier 2 sized for a 2-core, 8 GB CI runner; trends documented |
| Hugging Face download fails or file changes | Pinned revisions, SHA-256 checks, clear error message with manual download steps |
| Results vary run to run with parallel slots | Temperature 0, fixed seed, 5 repetitions; batched decoding is not bit-identical, so compare quality by perplexity |
| Holiday season and part-time hours | M1 and M2 before Christmas; one buffer week; cut list above |

## 13. Decisions

Settled:
- Submission around 15 Jan 2027; evaluators run the code themselves; report format left to the author.
- Public GitHub repository under `aksh-j-patel`.
- Headline benchmarks in Docker; core pinning and best configuration also run natively on Windows.

Open:
1. Whether to include the optional GPU comparison.
2. Report format confirmation: Markdown to PDF as proposed, or LaTeX if the course prefers it.
