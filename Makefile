# Run targets inside the container:   docker compose run --rm bench make <target>
# (They also work natively if llama.cpp tools and Python deps are installed; see docs/REPRODUCE.md.)

PY       ?= python
SET      ?= smoke
STAGE    ?= s0_baseline
TIER     ?= quick
STAGES   ?= s0_baseline s1_engine
THREADS  ?=
PEAK_GBS ?=

.DEFAULT_GOAL := help
.PHONY: help sysinfo lock-models download verify-models stream smoke test plots tiny-model \
        bench bench-tiny bench-quick bench-full

help:
	@echo 'Targets:'
	@echo '  sysinfo        show the environment and save it to results/raw'
	@echo '  lock-models    pin model revisions and hashes into configs/models.lock.json (maintainer, once)'
	@echo '  download       download and verify a model set: make download SET=smoke|quick|full'
	@echo '  verify-models  re-check model files against the lock: make verify-models SET=...'
	@echo '  stream         memory bandwidth (STREAM): make stream [THREADS=1,2,4] [PEAK_GBS=76.8]'
	@echo '  smoke          end-to-end check with the 0.5B model (downloads ~1.5 GB once)'
	@echo '  test           unit tests, plus an offline llama.cpp check with a tiny random model'
	@echo '  plots          regenerate figures from results/raw: make plots [PEAK_GBS=76.8]'
	@echo '  bench          one stage: make bench STAGE=s0_baseline TIER=tiny|smoke|quick|full'
	@echo '  bench-tiny     all implemented stages with tiny random models (offline, ~1 min)'
	@echo '  bench-quick    all implemented stages, reduced grid on the real models (~20-30 min + first download)'
	@echo '  bench-full     all implemented stages, full grid as reported (~1.5-2 h)'

sysinfo:
	$(PY) -m bench sysinfo --save

lock-models:
	$(PY) -m bench models lock

download:
	$(PY) -m bench models download --set $(SET)

verify-models:
	$(PY) -m bench models verify --set $(SET)

stream:
	$(PY) -m bench stream $(if $(THREADS),--threads $(THREADS)) $(if $(PEAK_GBS),--peak-gbs $(PEAK_GBS))

smoke:
	$(PY) -m bench smoke

test:
	$(PY) -m pytest -q

plots:
	$(PY) -m bench plots $(if $(PEAK_GBS),--peak-gbs $(PEAK_GBS))

tiny-model:
	$(PY) -m bench tiny-model

bench:
	$(PY) -m bench run $(STAGE) --tier $(TIER)

bench-tiny:
	$(PY) -m bench run $(STAGES) --tier tiny

bench-quick:
	$(PY) -m bench run $(STAGES) --tier quick

bench-full:
	$(PY) -m bench run $(STAGES) --tier full