"""Supervised LoRA fine-tuning with completion-only loss and crash-safe resume from Drive."""
from __future__ import annotations

import inspect
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import torch

from . import config as C
from .prompts import build_prompt, target_text


@dataclass
class TrainConfig:
    base_model: str = C.TARGET_MODEL
    output_dir: str = "checkpoints/qwen15_lora"
    train_file: str = str(C.DATA_DIR / "train.jsonl")
    val_file: str = str(C.DATA_DIR / "val.jsonl")
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    epochs: float = 3.0
    lr: float = 2e-4
    per_device_batch: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_prompt_tokens: int = C.MAX_PROMPT_TOKENS
    eval_steps: int = 20
    save_steps: int = 20
    logging_steps: int = 5
    seed: int = 42
    gradient_checkpointing: bool = True


def tokenize_records(recs: List[Dict], tok, max_prompt_tokens: int):
    """Prompt tokens get label -100 (no loss); loss only on the JSON answer + <|im_end|>."""
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if end_id is None or end_id == tok.unk_token_id:
        end_id = tok.eos_token_id
    feats, trunc = [], []
    for r in recs:
        p_ids, info = build_prompt(tok, r["sentences"], r["criterion_type"], r["criterion_text"],
                                   max_prompt_tokens, gold_evidence=r["gold_evidence"])
        # evidence lost to truncation cannot be cited; keep the label, drop lost IDs from target
        ev = [i for i in r["gold_evidence"] if i not in set(info["dropped_sentence_ids"])]
        a_ids = tok(target_text(r["gold_label"], ev), add_special_tokens=False)["input_ids"] + [end_id]
        feats.append({"input_ids": p_ids + a_ids,
                      "labels": [-100] * len(p_ids) + a_ids})
        trunc.append({"annotation_id": r["annotation_id"], **{k: info[k] for k in
                      ("prompt_tokens_full", "prompt_tokens", "truncated",
                       "dropped_sentence_ids", "gold_evidence_lost")}})
    return feats, trunc


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, feats):
        self.f = feats

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        return self.f[i]


def make_collator(pad_id: int):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            lab[i, :n] = torch.tensor(b["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return collate


def _training_args(cfg: TrainConfig, use_bf16: bool, use_fp16: bool):
    from transformers import TrainingArguments
    kw = dict(
        output_dir=cfg.output_dir, num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.per_device_batch, per_device_eval_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum, warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay, lr_scheduler_type="cosine",
        logging_steps=cfg.logging_steps, save_steps=cfg.save_steps, eval_steps=cfg.eval_steps,
        save_strategy="steps", save_total_limit=3, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        bf16=use_bf16, fp16=use_fp16, report_to="none", seed=cfg.seed,
        gradient_checkpointing=cfg.gradient_checkpointing, remove_unused_columns=False,
        dataloader_num_workers=0, optim="adamw_torch",
    )
    if cfg.gradient_checkpointing:
        kw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    kw = {k: v for k, v in kw.items() if k in params or k in ("eval_strategy", "evaluation_strategy")}
    return TrainingArguments(**kw)


def train(cfg: TrainConfig) -> Dict:
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
                              set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    from .data import read_jsonl

    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_recs, val_recs = read_jsonl(cfg.train_file), read_jsonl(cfg.val_file)
    tr_feats, tr_trunc = tokenize_records(train_recs, tok, cfg.max_prompt_tokens)
    va_feats, va_trunc = tokenize_records(val_recs, tok, cfg.max_prompt_tokens)
    lens = [len(f["input_ids"]) for f in tr_feats]
    data_stats = {
        "n_train": len(tr_feats), "n_val": len(va_feats),
        "train_seq_len": {"min": min(lens), "max": max(lens), "mean": sum(lens) / len(lens)},
        "train_truncated": sum(t["truncated"] for t in tr_trunc),
        "val_truncated": sum(t["truncated"] for t in va_trunc),
        "train_gold_evidence_lost": sum(bool(t["gold_evidence_lost"]) for t in tr_trunc),
        "val_gold_evidence_lost": sum(bool(t["gold_evidence_lost"]) for t in va_trunc),
    }
    (out / "train_data_stats.json").write_text(json.dumps(data_stats, indent=2))
    print("data:", data_stats)

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=cfg.target_modules, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    for n, p in model.named_parameters():        # trainable LoRA params in fp32 (AMP-safe)
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    log_path = out / "log_history.json"

    class PersistLogs(TrainerCallback):
        """Write log history to Drive on every log event so curves survive a disconnect."""
        def on_log(self, args, state, control, logs=None, **kw):
            log_path.write_text(json.dumps(state.log_history, indent=1))

    args = _training_args(cfg, use_bf16=bf16, use_fp16=(cuda and not bf16))
    trainer = Trainer(model=model, args=args, train_dataset=ListDataset(tr_feats),
                      eval_dataset=ListDataset(va_feats), data_collator=make_collator(tok.pad_token_id),
                      callbacks=[PersistLogs()])
    last = get_last_checkpoint(str(out)) if any(out.glob("checkpoint-*")) else None
    print("Resuming from", last) if last else print("Starting fresh")
    t0 = time.time()
    res = trainer.train(resume_from_checkpoint=last)
    elapsed = time.time() - t0

    final = out / "final_adapter"
    trainer.model.save_pretrained(str(final))
    tok.save_pretrained(str(final))
    log_path.write_text(json.dumps(trainer.state.log_history, indent=1))
    summary = {
        "base_model": cfg.base_model, "dtype": str(dtype), "trainable_params": trainable,
        "total_params": total, "train_runtime_s_this_session": elapsed,
        "train_loss": res.training_loss, "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric, "global_steps": trainer.state.global_step,
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
        **data_stats,
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot_curves(log_history_path, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hist = json.loads(Path(log_history_path).read_text())
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    lr = [(h["step"], h["learning_rate"]) for h in hist if "learning_rate" in h]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    if tr:
        ax[0].plot(*zip(*tr), label="train loss", color="#3b6ea5")
    if ev:
        ax[0].plot(*zip(*ev), "o-", label="validation loss", color="#d1495b")
    ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("cross-entropy (answer tokens)")
    ax[0].legend(frameon=False); ax[0].spines[["top", "right"]].set_visible(False)
    if lr:
        ax[1].plot(*zip(*lr), color="#555")
    ax[1].set_xlabel("optimizer step"); ax[1].set_ylabel("learning rate")
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
