"""Run one experiment stage end to end: resolve the configuration, make sure the models are
present, start each server, warm it up, run every load configuration with resource sampling,
and write the results.

Outputs in results/raw (one timestamp per run):
  <stage>_<tier>_requests_<ts>.csv    one row per request
  <stage>_<tier>_summary_<ts>.csv     one row per (server, prompt class, users, repetition)
  <stage>_<tier>_resources_<ts>.csv   resource samples every 0.5 s
  <stage>_<tier>_<ts>.json            environment, resolved configuration, server command lines
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import yaml

from bench import models, paths, results, sysinfo
from bench.analysis import metrics
from bench.loadgen import client
from bench.profiling.sampler import ResourceSampler
from bench.serving.launch import ServerSpec, running
from bench.tinymodel import TINY_KEYS, ensure_tiny

EXPERIMENTS = paths.CONFIG_DIR / "experiments.yaml"
SERVER_FIELDS = ("threads", "parallel", "ctx_per_slot", "cont_batching", "extra_args", "env")
TABLE_METRICS = ("throughput_tok_s", "ttft_p50_s", "e2e_p95_s", "tpot_mean_ms", "peak_rss_mb")


class RunError(RuntimeError):
    pass


def load_experiments(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or EXPERIMENTS).read_text(encoding="utf-8"))


def resolve(stage: str, tier: str, cfg: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = cfg or load_experiments()
    if stage not in cfg["stages"]:
        raise RunError(f"unknown stage '{stage}' (runnable: {', '.join(cfg['stages'])})")
    if tier not in cfg["tiers"]:
        raise RunError(f"unknown tier '{tier}' (known: {', '.join(cfg['tiers'])})")
    st, tr = cfg["stages"][stage], cfg["tiers"][tier]
    wl = copy.deepcopy(cfg["workload"])
    wl.update(st.get("workload") or {})
    wl.update(tr.get("overrides") or {})
    wl.update((tr.get("stage_overrides") or {}).get(stage) or {})
    servers = []
    for s in st["servers"]:
        s = copy.deepcopy(s)
        s["model"] = (tr.get("model_map") or {}).get(s["model"], s["model"])
        servers.append(s)
    return wl, servers


def model_location(key: str) -> Path:
    if key in TINY_KEYS:
        vocab = paths.qwen2_vocab()
        if vocab is None:
            raise RunError("tiny models need LLAMA_VOCAB_QWEN2 (set inside the Docker image)")
        return ensure_tiny(key, paths.MODELS_DIR, vocab)
    return models.model_path(key)


def _table(rows: list[dict[str, Any]]) -> str:
    head = f"  {'server':18s} {'class':7s} {'users':>5s} {'tok/s':>8s} {'TTFT p50':>9s} {'e2e p95':>8s} {'TPOT ms':>8s} {'RSS MB':>8s}"
    lines = [head, "  " + "-" * (len(head) - 2)]

    def f(x: Any, fmt: str) -> str:
        return format(x, fmt) if isinstance(x, (int, float)) else "-"

    for r in rows:
        lines.append(f"  {r['server']:18s} {r['prompt_class']:7s} {r['users']:>5d} "
                     f"{f(r['throughput_tok_s'], '8.2f')} {f(r['ttft_p50_s'], '9.3f')} "
                     f"{f(r['e2e_p95_s'], '8.2f')} {f(r['tpot_mean_ms'], '8.1f')} {f(r['peak_rss_mb'], '8.0f')}")
    return "\n".join(lines)


def run_stage(stage: str, tier: str, plot: bool = True) -> int:
    wl, server_cfgs = resolve(stage, tier)
    tag = f"{stage}_{tier}"
    info = sysinfo.collect()
    print("\n".join(sysinfo.summary_lines(info)), "\n")
    print(f"Stage {stage} / tier {tier}: {len(server_cfgs)} server(s) x {len(wl['prompt_classes'])} prompt "
          f"class(es) x users {wl['users']} x {wl['repetitions']} repetition(s), {wl['max_tokens']} output tokens\n")

    downloadable = sorted({s["model"] for s in server_cfgs if s["model"] not in TINY_KEYS})
    if downloadable:
        models.download_keys(downloadable, f"{tag} models")

    prompts = client.load_prompts()
    ts = results.timestamp()
    log_dir = paths.RESULTS_DIR / "logs"
    request_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    server_meta: list[dict[str, Any]] = []
    total = len(server_cfgs) * len(wl["prompt_classes"]) * len(wl["users"]) * wl["repetitions"]
    done, t_stage = 0, time.perf_counter()

    for si, sc in enumerate(server_cfgs):
        spec = ServerSpec(name=sc["name"], engine=sc["engine"], model_key=sc["model"],
                          model_path=str(model_location(sc["model"])),
                          **{k: sc[k] for k in SERVER_FIELDS if k in sc})
        with running(spec, log_dir) as srv:
            sampler = ResourceSampler(srv.pid).start()
            try:
                sampler.label = "warmup"
                client.warmup(srv.base_url, prompts["short"], n=wl["warmup_requests"])
                offset = 0
                blocks = [(c, u) for c in wl["prompt_classes"] for u in wl["users"]]
                for bi, (cls, users) in enumerate(blocks):
                    for rep in range(1, wl["repetitions"] + 1):
                        done += 1
                        load = client.LoadSpec(cls, users, wl["requests_per_user"], wl["max_tokens"],
                                               float(wl["temperature"]), wl["seed"], wl["ignore_eos"],
                                               wl["cache_prompt"])
                        sampler.label = f"{cls}/u{users}/r{rep}"
                        t0 = sampler.now()
                        res, wall = client.run(srv.base_url, load, prompts[cls], offset)
                        window = sampler.window(t0, sampler.now())
                        offset += users * wl["requests_per_user"]
                        base = {"server": spec.name, "engine": spec.engine, "model": spec.model_key,
                                "prompt_class": cls, "users": users, "repetition": rep}
                        rows = [{**base, **r.row()} for r in res]
                        request_rows += rows
                        summ = {**base, **metrics.summarize(rows, wall), **window}
                        summary_rows.append(summ)
                        errs = f"  ERRORS: {summ['errors']} ({next(r['error'] for r in rows if r['error'])[:120]})" \
                            if summ["errors"] else ""
                        print(f"  [{done}/{total}] {spec.name} | {cls} | users={users} | rep {rep}: "
                              f"{summ['throughput_tok_s']} tok/s, TTFT p50 {summ['ttft_p50_s']} s, "
                              f"e2e p95 {summ['e2e_p95_s']} s{errs}")
                    last = si == len(server_cfgs) - 1 and bi == len(blocks) - 1
                    if wl["cooldown_seconds"] and not last:
                        sampler.label = "cooldown"
                        time.sleep(wl["cooldown_seconds"])
            finally:
                sampler.stop()
            resource_rows += [{"server": spec.name, **s} for s in sampler.samples]
            server_meta.append({"spec": spec.as_dict(), "command": srv.command, "ready_s": round(srv.ready_s, 2),
                                "peak_rss_mb": sampler.peak_rss_mb(), "model_lock": models.lock_entry(spec.model_key),
                                "log_tail": srv.log_tail(15)})

    paths_out = {
        "requests": results.write_csv(f"{tag}_requests", request_rows, ts),
        "summary": results.write_csv(f"{tag}_summary", summary_rows, ts),
        "resources": results.write_csv(f"{tag}_resources", resource_rows, ts),
    }
    results.write_json(tag, {"stage": stage, "tier": tier, "workload": wl, "servers": server_meta,
                             "duration_s": round(time.perf_counter() - t_stage, 1), "sysinfo": info,
                             "files": {k: v.name for k, v in paths_out.items()}}, ts)

    agg = metrics.medians(summary_rows, TABLE_METRICS)
    print(f"\nMedians over {wl['repetitions']} repetition(s):\n{_table(agg)}")
    for k, v in paths_out.items():
        print(f"Written: {v.relative_to(paths.REPO_ROOT) if v.is_relative_to(paths.REPO_ROOT) else v}")
    if plot:
        from bench.analysis import plots

        for p in plots.concurrency_plots(paths_out["summary"]):
            print(f"Plot:    {p.relative_to(paths.REPO_ROOT) if p.is_relative_to(paths.REPO_ROOT) else p}")

    errors = sum(r["errors"] for r in summary_rows)
    if errors:
        print(f"\n{errors} request(s) failed; see the 'error' column in the requests CSV.")
    print(f"Stage finished in {(time.perf_counter() - t_stage) / 60:.1f} min")
    return 1 if errors else 0
