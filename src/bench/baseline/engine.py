"""Token-streaming generation with Hugging Face transformers.

This is the deliberately naive reference: FP32 weights, PyTorch's default threading, greedy
decoding through `model.generate`, and one request at a time (the server holds a lock).
Sampling settings are pinned explicitly (no repetition penalty, no top-k/top-p) so the work
per token matches llama.cpp's defaults rather than the model card's generation_config.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator


@dataclass
class GenParams:
    max_tokens: int = 128
    temperature: float = 0.0
    seed: int | None = None
    ignore_eos: bool = False


@dataclass
class Event:
    kind: str                      # "token" | "end" | "error"
    text: str = ""
    t: float = 0.0
    error: BaseException | None = None


@dataclass
class GenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_ms: float = 0.0         # generate() start -> first new token (prefill + first decode)
    predicted_ms: float = 0.0      # first new token -> last new token
    finish_reason: str = "length"
    extra: dict[str, Any] = field(default_factory=dict)

    def timings(self) -> dict[str, float]:
        decode_tokens = max(self.completion_tokens - 1, 0)
        return {
            "prompt_n": self.prompt_tokens,
            "prompt_ms": round(self.prompt_ms, 3),
            "prompt_per_second": round(self.prompt_tokens / (self.prompt_ms / 1000), 3) if self.prompt_ms else 0.0,
            "predicted_n": self.completion_tokens,
            "predicted_ms": round(self.predicted_ms, 3),
            "predicted_per_second": round(decode_tokens / (self.predicted_ms / 1000), 3)
            if self.predicted_ms and decode_tokens else 0.0,
        }


class HFEngine:
    def __init__(self, model_dir: str | Path, threads: int | None = None, dtype: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.model_dir = str(model_dir)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, dtype=getattr(torch, dtype))
        self.model.eval()
        self.load_s = time.perf_counter() - t0
        self.dtype = dtype
        self.threads = torch.get_num_threads()
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        self.eos_token_id = eos
        self.pad_token_id = self.tokenizer.pad_token_id or (eos[0] if isinstance(eos, list) else eos)

    def info(self) -> dict[str, Any]:
        return {"engine": "transformers", "model_dir": self.model_dir, "dtype": self.dtype,
                "torch_threads": self.threads, "load_s": round(self.load_s, 2),
                "parameters": sum(p.numel() for p in self.model.parameters())}

    def encode_chat(self, messages: list[dict[str, str]]):
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                  return_tensors="pt", return_dict=True)

    def _generation_config(self, p: GenParams):
        from transformers import GenerationConfig

        sample = p.temperature > 0
        return GenerationConfig(
            max_new_tokens=p.max_tokens,
            min_new_tokens=p.max_tokens if p.ignore_eos else 0,
            do_sample=sample,
            temperature=p.temperature if sample else 1.0,
            top_k=0 if sample else None,
            top_p=1.0,
            repetition_penalty=1.0,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    async def stream(self, messages: list[dict[str, str]], params: GenParams,
                     stats: GenStats) -> AsyncIterator[str]:
        """Yield text deltas as tokens are generated. Fills `stats` as it goes."""
        from transformers.generation.streamers import BaseStreamer

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Event] = asyncio.Queue()
        inputs = self.encode_chat(messages)
        stats.prompt_tokens = int(inputs["input_ids"].shape[-1])
        tokenizer = self.tokenizer

        class QueueStreamer(BaseStreamer):
            def __init__(self) -> None:
                self.prompt_seen = False
                self.ids: list[int] = []
                self.emitted = ""

            def put(self, value) -> None:
                if not self.prompt_seen:          # first call carries the prompt
                    self.prompt_seen = True
                    return
                now = time.perf_counter()
                self.ids.extend(int(x) for x in value.reshape(-1).tolist())
                text = tokenizer.decode(self.ids, skip_special_tokens=True)
                delta = "" if text.endswith("�") else text[len(self.emitted):]
                if delta:
                    self.emitted = text
                loop.call_soon_threadsafe(queue.put_nowait, Event("token", delta, now))

            def end(self) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, Event("end", t=time.perf_counter()))

        streamer = QueueStreamer()
        gen_config = self._generation_config(params)

        def work() -> None:
            try:
                if params.seed is not None:
                    self.torch.manual_seed(params.seed)
                with self.torch.inference_mode():
                    self.model.generate(**inputs, generation_config=gen_config, streamer=streamer)
            except BaseException as e:  # surface errors to the event loop
                loop.call_soon_threadsafe(queue.put_nowait, Event("error", error=e))

        t_start = time.perf_counter()
        thread = threading.Thread(target=work, name="hf-generate", daemon=True)
        thread.start()
        t_first = t_last = None
        n = 0
        while True:
            ev = await queue.get()
            if ev.kind == "error":
                raise RuntimeError(f"generation failed: {ev.error}") from ev.error
            if ev.kind == "end":
                break
            n += 1
            t_first = t_first or ev.t
            t_last = ev.t
            if ev.text:
                yield ev.text
        await asyncio.to_thread(thread.join)
        stats.completion_tokens = n
        stats.prompt_ms = ((t_first or t_start) - t_start) * 1000
        stats.predicted_ms = ((t_last or t_start) - (t_first or t_start)) * 1000
        stats.finish_reason = "length" if n >= params.max_tokens else "stop"
