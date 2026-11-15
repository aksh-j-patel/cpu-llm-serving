"""Generate configs/prompts.json: the fixed benchmark prompt set.

Three length classes (short ~30, medium ~250, long ~1000 prompt tokens with the Qwen2.5
tokenizer), eight prompts each. The text is synthetic and deterministic, so no dataset
download or licence is involved. The JSON file is committed; this script documents how it was
made. Re-running it reproduces the same file.

    python scripts/make_prompts.py [--vocab path/to/ggml-vocab-qwen2.gguf]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "prompts.json"
PER_CLASS = 8
TARGET_TOKENS = {"medium": 250, "long": 1000}

SHORT = [
    "Explain in two sentences why memory bandwidth limits LLM inference on CPUs.",
    "List three practical uses of a small language model running on a laptop.",
    "What is the difference between latency and throughput? Answer briefly.",
    "Give a one-paragraph summary of how a hash table works.",
    "Suggest a short checklist for reviewing a pull request.",
    "Describe what a cache miss is to a new programmer.",
    "Why do databases use indexes? Keep it short.",
    "Write a haiku-free, plain explanation of what an API is.",
]

TASKS = [
    "Summarise the following project notes in five bullet points.",
    "Read the notes below and list the three biggest risks they mention.",
    "Using the notes below, draft a short status update for a manager.",
    "Extract every decision recorded in the notes below.",
    "From the notes below, write a list of open questions for the next meeting.",
    "Identify which teams are mentioned in the notes below and what each owns.",
    "Turn the notes below into a numbered action list with owners.",
    "Explain the main trade-off discussed in the notes below.",
]

TEAMS = ["platform", "data", "search", "payments", "mobile", "infrastructure", "security", "analytics"]
SYSTEMS = ["the ingestion pipeline", "the billing service", "the recommendation model", "the cache layer",
           "the reporting dashboard", "the authentication flow", "the batch scheduler", "the API gateway"]
VERBS = ["reviewed", "proposed changes to", "raised concerns about", "measured the latency of",
         "agreed to refactor", "postponed work on", "documented", "load-tested"]
DETAILS = ["p95 latency rose after the last release", "memory use doubled under peak traffic",
           "the migration needs one more week", "two customers reported timeouts",
           "costs fell by about ten percent", "test coverage is still below target",
           "the on-call rotation is understaffed", "a dependency upgrade is overdue",
           "throughput is limited by a single worker", "logs are missing request identifiers"]
ACTIONS = ["will prepare a design note", "will run a benchmark", "will contact the vendor",
           "will add monitoring", "will write a rollback plan", "will pair with the data team"]


def sentence(rng: random.Random, i: int) -> str:
    return (f"Item {i}: the {rng.choice(TEAMS)} team {rng.choice(VERBS)} {rng.choice(SYSTEMS)}; "
            f"they noted that {rng.choice(DETAILS)}, and {rng.choice(['Asha', 'Ravi', 'Meera', 'Karan', 'Nisha', 'Dev'])} "
            f"{rng.choice(ACTIONS)}.")


def build(task: str, n_tokens: int, count, seed: int) -> str:
    rng = random.Random(seed)
    lines = [task, "", "Notes:"]
    i = 1
    while count("\n".join(lines)) < n_tokens:
        lines.append(sentence(rng, i))
        i += 1
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="/opt/llama.cpp/share/ggml-vocab-qwen2.gguf")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    vocab = Path(args.vocab)
    tok = AutoTokenizer.from_pretrained(vocab.parent, gguf_file=vocab.name)

    def count(text: str) -> int:
        return len(tok(text)["input_ids"])

    data = {"description": "Fixed benchmark prompts; token counts are for the Qwen2.5 tokenizer "
                           "(message text only, excluding the chat template).",
            "classes": {}}
    data["classes"]["short"] = [{"id": f"short-{i}", "text": t, "tokens": count(t)} for i, t in enumerate(SHORT)]
    for cls, n in TARGET_TOKENS.items():
        items = []
        for i, task in enumerate(TASKS[:PER_CLASS]):
            text = build(task, n, count, seed=1000 * (n + i))
            items.append({"id": f"{cls}-{i}", "text": text, "tokens": count(text)})
        data["classes"][cls] = items
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for cls, items in data["classes"].items():
        toks = [p["tokens"] for p in items]
        print(f"{cls:6s}: {len(items)} prompts, {min(toks)}-{max(toks)} tokens")
    print(f"Written: {OUT}")


if __name__ == "__main__":
    main()
