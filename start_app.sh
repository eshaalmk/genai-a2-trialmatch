#!/usr/bin/env bash
# TrialMatch launcher for macOS / Linux:  bash start_app.sh   (or double-click if executable)
set -e
cd "$(dirname "$0")"
ADAPTER_URL="PASTE_QWEN15_ADAPTER_ZIP_URL_HERE"
DRAFT_ADAPTER_URL="PASTE_QWEN05_DRAFT_ADAPTER_ZIP_URL_HERE"
PY=.venv/bin/python

if [ ! -x "$PY" ]; then
  echo "[setup] Creating Python environment (first run only)..."
  python3 -m venv .venv
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt || { echo "Package install failed"; rm -rf .venv; exit 1; }
fi
if [ ! -f weights/qwen15_lora/adapter_config.json ]; then
  echo "[setup] Downloading fine-tuned adapter..."
  "$PY" scripts/download_weights.py --url "$ADAPTER_URL" --dest weights/qwen15_lora
fi
if [ ! -f weights/qwen05_lora/adapter_config.json ]; then
  "$PY" scripts/download_weights.py --url "$DRAFT_ADAPTER_URL" --dest weights/qwen05_lora \
    || echo "NOTE: draft adapter not available; speculative mode uses the base 0.5B draft."
fi
if [ ! -f data/processed/val.jsonl ]; then
  "$PY" scripts/prepare_data.py --no_tokenizer --out_results data/processed/manifest_check >/dev/null 2>&1 || true
fi
echo "Starting TrialMatch at http://127.0.0.1:7860"
"$PY" app/app.py
