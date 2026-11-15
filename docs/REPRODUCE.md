# Reproducing the results

Everything runs inside one Docker image, so the steps are the same on Windows, macOS and Linux.
Absolute numbers depend on your hardware; the relative trends should hold on any machine
(see `docs/EXPECTED_RESULTS.md`, added once results exist).

Status (week 1): environment, model download, STREAM and the smoke test are in place.
The baseline server and the benchmark tiers arrive from week 2.

## 1. Prerequisites

| Item | Requirement |
|---|---|
| Docker | Docker Desktop (Windows, macOS) or Docker Engine with the Compose plugin (Linux) |
| CPU | Any x86-64 with AVX2, or Apple Silicon / other ARM64 |
| RAM | 8 GB for the smoke test; 16 GB for the full benchmark (the FP32 baseline alone needs about 7 GB) |
| Disk | About 3 GB for the image, plus model files: smoke ≈ 1.5 GB, quick ≈ 8 GB, full ≈ 11 GB (estimates) |
| Network | Needed for the first build and model download only |

### Windows: give Docker enough memory

Docker Desktop runs containers in a WSL2 virtual machine that gets **half of your RAM by default**
(8 GB on a 16 GB laptop). That is too little for the FP32 baseline. Create or edit
`%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=12GB
processors=12
```

Set `processors` to your logical CPU count. Then run `wsl --shutdown` in PowerShell and restart
Docker Desktop. Check from inside the container with `make sysinfo`.

## 2. Build the image

```bash
git clone https://github.com/aksh-j-patel/cpu-llm-serving.git
cd cpu-llm-serving
docker compose build
```

The first build compiles llama.cpp for your CPU and takes roughly 10 to 20 minutes. Build the
image on the machine you will benchmark on, because it is optimised for that processor.

## 3. Run

All commands use `docker compose run --rm bench make <target>`:

| Target | What it does | Time (first run) |
|---|---|---|
| `make help` | List targets | instant |
| `make sysinfo` | Show the detected CPU, memory, OS, versions; save to `results/raw` | seconds |
| `make test` | Unit tests plus an offline llama.cpp check using a tiny random-weight model | < 1 min |
| `make smoke` | Download the 0.5B model (once), verify hashes, run llama.cpp and transformers briefly | 5–15 min |
| `make stream PEAK_GBS=<n>` | Memory-bandwidth sweep over thread counts; writes CSV and a plot | 1–3 min |
| `make plots` | Rebuild figures from the CSVs in `results/raw` (no benchmarking) | seconds |
| `make download SET=quick` | Pre-download the models for the quick tier | depends on network |

`PEAK_GBS` is optional: your memory's theoretical bandwidth, drawn as a reference line. For example,
LPDDR5 at 4800 MT/s on a 128-bit bus gives 4800 × 16 / 1000 ≈ 76.8 GB/s.

Results go to `results/raw/` (CSV and JSON, each with a full description of the machine) and
figures to `results/plots/`.

## 4. What a passing smoke test looks like

```
Smoke test summary
  PASS  environment
  PASS  tiny-engine
  PASS  models
  PASS  llama.cpp-0.5b  pp64: ... tok/s, tg32: ... tok/s
  PASS  transformers-0.5b  generation: ... tok/s (FP32, N threads)
```

Expect llama.cpp generation (tg32) to be several times faster than transformers FP32 even at this
stage; that gap is what the project goes on to measure and explain.

## 5. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Illegal instruction` | The image was built on a different CPU. Rebuild it on this machine: `docker compose build --no-cache` |
| Container killed, exit code 137 | Out of memory. Raise the WSL2 limit (section 1) or close other applications |
| `models.lock.json not found` | Maintainer step not done yet; see section 6 |
| `hash mismatch` after download | Corrupted or changed file. Delete that model's folder and run the command again |
| Hugging Face download errors | Check the network; downloads resume when the command is run again |
| Very slow model loading on Windows | Keep models in the Docker volume (the default), not a folder shared from Windows |

## 6. Maintainer notes

**Model lock (once, then whenever models change).** Pins the exact Hugging Face commit and file hashes:

```bash
docker compose run --rm bench make lock-models
git add configs/models.lock.json && git commit -m "Pin model revisions"
```

**Pin the base image digest.** After the first successful build:

```bash
docker buildx imagetools inspect python:3.12-slim-trixie --format "{{json .Manifest.Digest}}"
```

Then set `ARG PYTHON_IMAGE=python:3.12-slim-trixie@sha256:<digest>` in the Dockerfile.

**Python dependencies.** Edit `requirements.in`, then regenerate the hash-locked file:

```bash
uv pip compile requirements.in --universal --python-version 3.12 --generate-hashes -o requirements.txt
```

**llama.cpp version.** Change `LLAMA_CPP_TAG` in the Dockerfile and rebuild. Record the change in
the report, since engine versions affect performance.

**Pinned versions (week 1):** Python 3.12, PyTorch 2.14.1 (CPU), transformers 5.23.0,
llama.cpp b12407 (commit e574a17), STREAM commit 6703f75.
