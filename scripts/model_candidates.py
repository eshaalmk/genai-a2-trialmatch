"""Task 1 / A2: architecture facts for candidate open-weight decoders, read from their
configs (no weight download) plus an fp16 memory estimate for inference and LoRA training.
Gated models (e.g. Llama) need HF_TOKEN and accepted licence; failures are recorded."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trialmatch import config as C

CANDIDATES = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Apache-2.0",
    "Qwen/Qwen2.5-0.5B-Instruct": "Apache-2.0",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen Research License (non-commercial)",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "Apache-2.0",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 Community License (gated)",
}


def count_params(model_id):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)

    # Ensure tied embeddings are counted only once.
    m.tie_weights()

    n = sum(p.numel() for p in m.parameters())
    return cfg, n


def main():
    rows = []
    for mid, lic in CANDIDATES.items():
        try:
            cfg, n = count_params(mid)
            hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            kv_per_tok = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * hd * 2
            rows.append({
                "model": mid, "license": lic, "architecture": cfg.architectures[0] if cfg.architectures else cfg.model_type,
                "params_B": round(n / 1e9, 3), "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
                "heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads, "head_dim": hd,
                "vocab": cfg.vocab_size, "context": cfg.max_position_embeddings,
                "rope_theta": getattr(cfg, "rope_theta", None), "rope_scaling": getattr(cfg, "rope_scaling", None),
                "weights_fp16_GB": round(n * 2 / 1e9, 2),
                "kv_cache_fp16_MB_per_1k_tokens": round(kv_per_tok * 1024 / 1e6, 1),
                # rough LoRA training footprint: fp16 frozen weights + ~1.5x for activations w/ checkpointing at 1.5k tokens
                "lora_train_estimate_GB": round(n * 2 / 1e9 * 1.6 + 1.0, 1),
            })
        except Exception as e:
            rows.append({"model": mid, "license": lic, "error": str(e)[:300]})
        print(rows[-1])
    out = C.RESULTS_DIR / "research"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_candidates.json").write_text(json.dumps(rows, indent=2))
    cols = ["model", "license", "params_B", "layers", "hidden", "heads", "kv_heads", "head_dim", "vocab",
            "context", "weights_fp16_GB", "kv_cache_fp16_MB_per_1k_tokens", "lora_train_estimate_GB"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, r.get("error", "") if c == "params_B" else "")) for c in cols) + " |")
    (out / "model_candidates.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
