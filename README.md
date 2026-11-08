# TrialMatch — Transformer-based clinical-trial criterion screening (GenAI Assignment 2)

Fine-tuned open-weight decoder (Qwen2.5-1.5B-Instruct + LoRA) that takes **(patient note, criterion type,
criterion text)** and returns a **structured eligibility label + supporting patient-sentence IDs**, with
hand-implemented **KV-cached decoding**, **tiled online-softmax attention** and **greedy speculative
decoding**, wrapped in a **browser application**.

> ⚠ Educational screening prototype. Not a clinical enrollment decision tool.

## Repository layout

| Path | What it contains | Assignment task |
|---|---|---|
| `trialmatch/data.py` | loading, GPT-4 column removal, sentence-ID parsing, fixed 37/8/8 split (seed 42), manifest, leakage checks | B1, B2 |
| `trialmatch/prompts.py` | prompt, JSON output schema, truncation policy, robust output parser, grounded explanations | B2, C4 |
| `trialmatch/train.py` | LoRA SFT, completion-only loss, Drive-safe checkpoint resume, curves | B3 |
| `trialmatch/evaluate.py` | batched greedy prediction, accuracy / macro-F1 / per-class / confusion matrices (overall, inclusion, exclusion), evidence P/R/F1 (micro + macro), error categories | B4, C1–C4 |
| `trialmatch/manual_decoder.py` | from-scratch Qwen2 forward (RoPE, GQA, causal masking with offset) with **own pre-allocated KV cache** (growth, rollback) | D1 |
| `trialmatch/decoding.py` | uncached reference vs explicit-cache greedy loops (HF `DynamicCache` and own cache), timing, mismatch investigation | D1–D3 |
| `trialmatch/attention.py` | reference SDPA and **tiled online-softmax attention** (running max / normaliser) | E1–E2 |
| `trialmatch/speculative.py` | **greedy speculative decoding**: drafting, one-pass target verification, accepted prefix, correction/bonus token, cache rollback | F1–F2 |
| `trialmatch/screen.py`, `aggregation.py` | end-to-end screening, explicit patient–trial aggregation rule, HTML/JSON report with provenance | G2, G4 |
| `app/app.py` | Gradio browser app: screening, evidence view, report export, **live inference benchmark** | G1–G4 |
| `scripts/` | CLI entry points for every experiment (see below) | — |
| `tests/` | unit tests (split procedure, parsing, metrics, attention, speculative logic on a mock model) | — |
| `notebooks/GenAI_A2_TrialMatch_Colab.ipynb` | one-click Colab runner (Drive checkpoints + GitHub push) | H2 |
| `results/` | manifest, logs, metrics, benchmark JSON/MD/PNG (committed) | — |

## Design decisions (summary — details go in the report)

- **Target model:** `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, 1.54 B params, 28 layers, GQA 12 q-heads / 2 kv-heads,
  head_dim 128, 32k context, ~3.1 GB in fp16) — fits LoRA training and the benchmarks on a free T4.
  **Draft model:** `Qwen/Qwen2.5-0.5B-Instruct` — same tokenizer/vocabulary (checked in code). Alternatives
  compared in `results/research/model_candidates.md` (SmolLM2-1.7B, Llama-3.2-1B [gated licence], Qwen2.5-3B
  [non-commercial licence, tight on T4]).
- **Output schema:** `{"evidence": [sentence ids], "label": "<label>"}`. Evidence comes first so the label is
  conditioned on the cited sentences. All 6 expert categories are kept (inclusion: included / not included /
  not enough information / not applicable; exclusion: excluded / not excluded / not enough information / not applicable).
- **Explanations** are built deterministically from the cited sentences only (the dataset has no expert
  explanations and GPT-4 explanations are forbidden as inputs). Decisive labels without cited evidence are
  flagged **UNSUPPORTED**; "not enough information" is never treated as proof either way.
- **Truncation:** whole trailing sentences are dropped beyond a 1,792-token prompt budget; original IDs are
  preserved, and every drop / lost gold-evidence ID is counted (`results/data/prompt_length_report.json`).
- **Evidence metrics:** micro P/R/F1 pooled over (record, sentence) pairs; macro P/R/F1 over records with
  non-empty gold evidence (empty prediction → 0); empty-gold records reported separately as "correct
  empty prediction" rate.
- **Test protection:** `evaluate_model.py --split test` refuses to run without `--final`.

## Running on Google Colab (recommended)

1. Push this folder to a new GitHub repo (or put `trialmatch-a2.zip` in `MyDrive/genai_a2/` and let the
   notebook bootstrap the empty repo).
2. Open `notebooks/GenAI_A2_TrialMatch_Colab.ipynb` in Colab, set a T4 GPU, add Colab secrets
   `GITHUB_TOKEN`, `GITHUB_REPO` (and optionally `HF_TOKEN`).
3. Run all. Each step skips itself if its outputs exist; training resumes from Drive checkpoints after a
   disconnect; results are mirrored to Drive and pushed to GitHub after every step.

## Running locally (GPU with ≥ 8 GB recommended; CPU works but is slow)

```bash
pip install -r requirements.txt
python -m pytest -q tests
python scripts/prepare_data.py                                  # split + manifest
python scripts/evaluate_model.py --tag base --split val         # before fine-tuning
python scripts/train_lora.py --output_dir checkpoints/qwen15_lora --results_dir results/training_qwen15
python scripts/evaluate_model.py --tag finetuned --split val --adapter checkpoints/qwen15_lora/final_adapter
python scripts/bench_kvcache.py   --adapter checkpoints/qwen15_lora/final_adapter
python scripts/bench_attention.py --with_model --adapter checkpoints/qwen15_lora/final_adapter
python scripts/bench_speculative.py --adapter checkpoints/qwen15_lora/final_adapter
python scripts/evaluate_model.py --tag base --split test --final
python scripts/evaluate_model.py --tag finetuned --split test --final --adapter checkpoints/qwen15_lora/final_adapter
python scripts/compare_before_after.py
```

## Starting the application (no coding needed)

Requirements: Python 3.10+ ([python.org](https://www.python.org/downloads/), tick *Add python.exe to PATH*),
internet on first start, ~8 GB free RAM and ~10 GB disk. An NVIDIA GPU is optional (CPU works: a few
seconds per criterion).

1. Download this repository (green **Code** button → *Download ZIP*) and unzip it.
2. **Windows:** double-click `start_app.bat`. **macOS/Linux:** run `bash start_app.sh`.
3. The first start creates a local environment, installs packages and downloads the weights
   (about 15–20 minutes, once). The browser then opens at **http://127.0.0.1:7860**.
   Later starts take about a minute. Keep the launcher window open while using the app; close it to stop.

The app validates inputs (empty note, too-long note or criterion, no criteria, bad NCT ID,
ClinicalTrials.gov unreachable) and reports errors in the UI. Reports are saved to `results/app_reports/`
and offered as downloads.

Optional NVIDIA GPU acceleration (Windows), run once after the first start:
`.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall`

Manual start: `python app/app.py` (uses `weights/qwen15_lora` and `weights/qwen05_lora` automatically;
`--share` gives a public link, e.g. in Colab).

## Weights

LoRA adapters (attached to the GitHub Release; downloaded automatically by the launchers):
- Target, Qwen2.5-1.5B-Instruct + LoRA: `<release URL of qwen15_lora_adapter.zip>`
- Draft, Qwen2.5-0.5B-Instruct + LoRA: `<release URL of qwen05_lora_adapter.zip>`

Manual download: `python scripts/download_weights.py --url <zip URL> --dest weights/qwen15_lora`.
The base models download automatically from Hugging Face on first start (public, no token needed).

Note on precision: the reported Colab results ran in bfloat16, which the T4 emulates. Inference code now
chooses fp16 on pre-Ampere GPUs, so app outputs may differ from the reported runs at near-tie tokens.

## Reproducibility notes
Split: `results/data/split_manifest.json` (patient IDs and counts per split). Training config and log
history: `results/training_*`. Every metrics file records the model, adapter, dtype, decoding settings.
Seeds: split 42 (fixed by the assignment), training seed 42.
