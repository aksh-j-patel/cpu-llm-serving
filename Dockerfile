# syntax=docker/dockerfile:1.7
#
# Benchmark environment: Python 3.12 + CPU-only PyTorch + llama.cpp + STREAM, all pinned.
# Build:  docker compose build        Use: docker compose run --rm bench make help
#
# llama.cpp and STREAM are compiled from source for the machine that builds the image
# (GGML_NATIVE=ON, -march=native), so build the image on the machine you benchmark on.
# CI passes GGML_NATIVE=OFF and STREAM_MARCH=x86-64-v3 so cached images run on any runner.

ARG PYTHON_IMAGE=python:3.12-slim-trixie

# ---------------------------------------------------------------- build stage
FROM ${PYTHON_IMAGE} AS builder

ARG LLAMA_CPP_TAG=b12407
ARG GGML_NATIVE=ON
ARG STREAM_COMMIT=6703f7504a38a8da96b353cadafa64d3c2d7a2d3
ARG STREAM_SHA256=c388924eb140fda95f534cdb808ae7f1f8ebb18da41d8aec1b512a3c8d303c9b
ARG STREAM_ARRAY_SIZE=50000000
ARG STREAM_MARCH=native

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMA_CPP_TAG}" https://github.com/ggml-org/llama.cpp.git

# Static build of the four tools used by the project. The build number is passed explicitly
# because a shallow clone cannot count commits.
RUN cmake -S llama.cpp -B llama.cpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE="${GGML_NATIVE}" \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_BUILD_NUMBER="${LLAMA_CPP_TAG#b}" \
      -DLLAMA_BUILD_IS_DEV=OFF \
      -DLLAMA_OPENSSL=OFF \
      -DLLAMA_USE_PREBUILT_UI=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
 && cmake --build llama.cpp/build -j"$(nproc)" \
      --target llama-server llama-bench llama-perplexity llama-cli \
 && mkdir -p /opt/llama.cpp/bin /opt/llama.cpp/share \
 && cp llama.cpp/build/bin/llama-server llama.cpp/build/bin/llama-bench \
       llama.cpp/build/bin/llama-perplexity llama.cpp/build/bin/llama-cli /opt/llama.cpp/bin/ \
 && cp llama.cpp/models/ggml-vocab-qwen2.gguf /opt/llama.cpp/share/ \
 && echo "${LLAMA_CPP_TAG} ($(git -C llama.cpp rev-parse --short HEAD), GGML_NATIVE=${GGML_NATIVE})" \
      > /opt/llama.cpp/VERSION

# STREAM memory-bandwidth benchmark, fetched at a pinned commit and checked against its hash.
RUN curl -fsSL -o stream.c \
      "https://raw.githubusercontent.com/jeffhammond/STREAM/${STREAM_COMMIT}/stream.c" \
 && echo "${STREAM_SHA256}  stream.c" | sha256sum -c - \
 && mkdir -p /opt/stream \
 && gcc -O3 -march="${STREAM_MARCH}" -fopenmp \
      -DSTREAM_ARRAY_SIZE="${STREAM_ARRAY_SIZE}" -DNTIMES=20 stream.c -o /opt/stream/stream

# ---------------------------------------------------------------- runtime stage
FROM ${PYTHON_IMAGE} AS runtime

ARG TORCH_VERSION=2.14.1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 git make \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Hash-checked dependencies first, then the CPU-only torch wheel (its own dependencies are
# already satisfied by the locked set, so only the torch wheel itself is unhashed).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --require-hashes -r /tmp/requirements.txt \
 && pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
 && python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"

COPY --from=builder /opt/llama.cpp /opt/llama.cpp
COPY --from=builder /opt/stream /opt/stream

ENV PATH=/opt/llama.cpp/bin:/opt/stream:${PATH} \
    LLAMA_CPP_BIN_DIR=/opt/llama.cpp/bin \
    LLAMA_CPP_VERSION_FILE=/opt/llama.cpp/VERSION \
    LLAMA_VOCAB_QWEN2=/opt/llama.cpp/share/ggml-vocab-qwen2.gguf \
    STREAM_BIN=/opt/stream/stream \
    BENCH_IN_CONTAINER=1 \
    BENCH_REPO_ROOT=/workspace \
    PYTHONPATH=/workspace/src \
    HF_HOME=/workspace/models/.hf-home \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /workspace
CMD ["make", "help"]
