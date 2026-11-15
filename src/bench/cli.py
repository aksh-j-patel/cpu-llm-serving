"""Command-line entry point: python -m bench <command>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench import paths


def _threads(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def cmd_sysinfo(args: argparse.Namespace) -> int:
    from bench import results, sysinfo

    info = sysinfo.collect()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print("\n".join(sysinfo.summary_lines(info)))
    if args.save:
        print(f"Written: {results.write_json('sysinfo', info)}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from bench import models

    try:
        if args.action == "list":
            print("\n".join(models.describe()))
        elif args.action == "lock":
            models.lock(args.model or None)
            print(f"Written: {paths.MODELS_LOCK} (commit this file)")
        elif args.action == "download":
            models.download(args.set)
        elif args.action == "verify":
            return 0 if models.verify(args.set) else 1
    except models.ModelError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    from bench import results, stream, sysinfo
    from bench.analysis import plots

    info = sysinfo.collect()
    print("\n".join(sysinfo.summary_lines(info)), "\n")
    try:
        rows = stream.run(args.threads, args.repeats)
    except stream.StreamError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    ts = results.timestamp()
    csv_path = results.write_csv("stream", rows, ts)
    results.write_json("stream", {"sysinfo": info, "repeats": args.repeats,
                                  "best_triad_gbs": stream.best_by_threads(rows)}, ts)
    best = stream.best_by_threads(rows)
    t_best = max(best, key=best.get)
    print(f"\nBest Triad bandwidth: {best[t_best]:.2f} GB/s at {t_best} threads")
    if args.peak_gbs:
        print(f"That is {100 * best[t_best] / args.peak_gbs:.0f}% of the {args.peak_gbs:.1f} GB/s theoretical peak")
    print(f"Written: {csv_path}")
    png = plots.stream_bandwidth(csv_path, paths.PLOTS_DIR / "stream_bandwidth.png", args.peak_gbs)
    print(f"Plot:    {png}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from bench import smoke

    return smoke.run(skip_download=args.skip_download)


def cmd_run(args: argparse.Namespace) -> int:
    from bench import models, runner

    rc = 0
    for stage in args.stage:
        try:
            rc |= runner.run_stage(stage, args.tier, plot=not args.no_plot)
        except (runner.RunError, models.ModelError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    return rc


def cmd_serve(args: argparse.Namespace) -> int:
    from bench.baseline import server

    argv = ["--model", args.model, "--host", args.host, "--port", str(args.port)]
    if args.threads:
        argv += ["--threads", str(args.threads)]
    return server.main(argv)


def cmd_tiny_model(args: argparse.Namespace) -> int:
    from bench.tinymodel import make_tiny_qwen2

    vocab = Path(args.vocab) if args.vocab else paths.qwen2_vocab()
    if vocab is None:
        print("ERROR: pass --vocab or set LLAMA_VOCAB_QWEN2", file=sys.stderr)
        return 1
    print(f"Written: {make_tiny_qwen2(Path(args.out), vocab)}")
    return 0


def cmd_plots(args: argparse.Namespace) -> int:
    from bench.analysis import plots

    made = plots.make_all(args.peak_gbs)
    if not made:
        print("No result files found in results/raw yet; nothing to plot.")
    for p in made:
        print(f"Plot: {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bench",
                                description="CPU-only LLM inference benchmark harness")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sysinfo", help="show (and optionally save) the environment description")
    s.add_argument("--json", action="store_true", help="print the full JSON")
    s.add_argument("--save", action="store_true", help="also write results/raw/sysinfo_<ts>.json")
    s.set_defaults(func=cmd_sysinfo)

    s = sub.add_parser("models", help="lock, download and verify model files")
    s.add_argument("action", choices=["list", "lock", "download", "verify"])
    s.add_argument("--set", default="smoke", help="model set from configs/models.yaml (default: smoke)")
    s.add_argument("--model", action="append", help="lock only this model key (repeatable)")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("stream", help="measure memory bandwidth with STREAM")
    s.add_argument("--threads", type=_threads, default=None, help="comma list, default: auto")
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--peak-gbs", type=float, default=None, help="theoretical peak, for the plot")
    s.set_defaults(func=cmd_stream)

    s = sub.add_parser("smoke", help="end-to-end environment and engine check")
    s.add_argument("--skip-download", action="store_true", help="offline: skip model download and model runs")
    s.set_defaults(func=cmd_smoke)

    s = sub.add_parser("run", help="run experiment stage(s) from configs/experiments.yaml")
    s.add_argument("stage", nargs="+", help="e.g. s0_baseline s1_engine")
    s.add_argument("--tier", default="quick", help="tiny | smoke | quick | full (default: quick)")
    s.add_argument("--no-plot", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("serve-baseline", help="run the transformers baseline server in the foreground")
    s.add_argument("--model", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--threads", type=int, default=None)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("tiny-model", help="write a tiny random Qwen2 GGUF for offline testing")
    s.add_argument("--out", default=str(paths.MODELS_DIR / "tiny" / "tiny-qwen2.gguf"))
    s.add_argument("--vocab", default=None)
    s.set_defaults(func=cmd_tiny_model)

    s = sub.add_parser("plots", help="regenerate report figures from results/raw")
    s.add_argument("--peak-gbs", type=float, default=None)
    s.set_defaults(func=cmd_plots)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
