"""Build a tiny random-weight Qwen2 model in GGUF format, with the real Qwen2 tokenizer.

Its output is gibberish, but it exercises exactly the same llama.cpp code paths as the real
model (loading, tokenisation, prompt processing, generation, the HTTP server). That lets the
tests and CI check the engine in seconds, with no download. The tokenizer comes from the
tokenizer-only file that ships with llama.cpp (models/ggml-vocab-qwen2.gguf).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TOKENIZER_PREFIX = "tokenizer."


def make_tiny_qwen2(out_path: Path, vocab_gguf: Path, n_embd: int = 64, n_layer: int = 2,
                    n_head: int = 4, n_head_kv: int = 2, n_ff: int = 128, n_ctx: int = 1024,
                    seed: int = 0) -> Path:
    import gguf

    reader = gguf.GGUFReader(str(vocab_gguf))
    tokens = reader.fields["tokenizer.ggml.tokens"].contents()
    n_vocab = len(tokens)
    head_dim = n_embd // n_head
    n_embd_kv = head_dim * n_head_kv

    w = gguf.GGUFWriter(str(out_path), "qwen2")
    w.add_name("tiny-random-qwen2")
    w.add_context_length(n_ctx)
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head_kv)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1_000_000.0)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # Copy tokenizer metadata unchanged.
    for name, field in reader.fields.items():
        if not name.startswith(TOKENIZER_PREFIX):
            continue
        value = field.contents()
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            w.add_array(name, value)
        else:
            w.add_key_value(name, value, vtype)

    rng = np.random.default_rng(seed)

    def t(name: str, *shape: int, scale: float = 0.02) -> None:
        w.add_tensor(name, (rng.standard_normal(shape) * scale).astype(np.float32))

    def ones(name: str, n: int) -> None:
        w.add_tensor(name, np.ones(n, dtype=np.float32))

    # numpy shapes are (rows, cols) = (out_features, in_features); gguf reverses them.
    t("token_embd.weight", n_vocab, n_embd)
    ones("output_norm.weight", n_embd)
    for i in range(n_layer):
        b = f"blk.{i}"
        ones(f"{b}.attn_norm.weight", n_embd)
        t(f"{b}.attn_q.weight", n_embd, n_embd)
        t(f"{b}.attn_q.bias", n_embd)
        t(f"{b}.attn_k.weight", n_embd_kv, n_embd)
        t(f"{b}.attn_k.bias", n_embd_kv)
        t(f"{b}.attn_v.weight", n_embd_kv, n_embd)
        t(f"{b}.attn_v.bias", n_embd_kv)
        t(f"{b}.attn_output.weight", n_embd, n_embd)
        ones(f"{b}.ffn_norm.weight", n_embd)
        t(f"{b}.ffn_gate.weight", n_ff, n_embd)
        t(f"{b}.ffn_up.weight", n_ff, n_embd)
        t(f"{b}.ffn_down.weight", n_embd, n_ff)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out_path
