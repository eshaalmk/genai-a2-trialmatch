"""Model / tokenizer loading shared by evaluation, benchmarks and the app."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch


def pick_device_dtype(dtype: Optional[str] = None):
    if torch.cuda.is_available():
        dev = "cuda"
        if dtype is None:
            # Native bf16 needs Ampere (compute capability >= 8). Older GPUs such as the T4
            # (7.5) report bf16 as "supported" but emulate it slowly, so use fp16 there.
            # NOTE: the submitted Colab results were produced before this fix, in bf16.
            dtype = "bf16" if torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"
    else:
        dev = "cpu"
        dtype = dtype or "fp32"
    return dev, {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]


def resolve_adapter(adapter: Optional[str]) -> Optional[str]:
    """Accept a local folder or a Hugging Face Hub repo id (downloaded on demand)."""
    if not adapter:
        return None
    if Path(adapter).exists():
        return str(adapter)
    from huggingface_hub import snapshot_download
    return snapshot_download(adapter)


def load_model(base_id: str, adapter: Optional[str] = None, dtype: Optional[str] = None,
               merge: bool = True, attn_impl: str = "sdpa") -> Tuple[torch.nn.Module, object]:
    """Load base model (+ optional LoRA adapter, merged into the weights for inference)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev, dt = pick_device_dtype(dtype)
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dt, attn_implementation=attn_impl)
    adapter = resolve_adapter(adapter)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    model.to(dev).eval()
    model.config.use_cache = True
    # Qwen2.5-Instruct ships a generation_config with sampling settings AND a
    # repetition_penalty > 1. do_sample=False disables sampling but NOT the repetition
    # penalty, so HF generate() would not be pure greedy decoding. Neutralise everything
    # so generate() == argmax decoding, identical to the hand-written decoders.
    gc = model.generation_config
    model._tm_original_generation_config = gc.to_dict()
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    gc.repetition_penalty = 1.0
    model._tm_meta = {"base_model": base_id, "adapter": adapter, "dtype": str(dt), "device": dev}
    return model, tok


def stop_token_ids(tok):
    ids = set()
    for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id and i is not None and i >= 0:
            ids.add(i)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def describe_model(model) -> dict:
    cfg = model.config
    n = sum(p.numel() for p in model.parameters())
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return {
        "architecture": type(model).__name__, "params": n, "layers": cfg.num_hidden_layers,
        "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads,
        "kv_heads": cfg.num_key_value_heads, "head_dim": hd, "vocab": cfg.vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rope_theta": getattr(cfg, "rope_theta", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "kv_cache_bytes_per_token": 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * hd
                                    * next(model.parameters()).element_size(),
        **getattr(model, "_tm_meta", {}),
    }
