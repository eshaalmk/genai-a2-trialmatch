@echo off
REM TrialMatch launcher for Windows: double-click this file.
REM First run: creates a Python environment, installs packages, downloads weights (~15-20 min).
REM Later runs: starts the app and opens the browser (~1 min to load models).
setlocal
cd /d "%~dp0"

REM ---- weight download links (direct links to the GitHub Release zip files) ----
set ADAPTER_URL=
set DRAFT_ADAPTER_URL=

if exist ".venv\Scripts\python.exe" goto have_env
echo [setup] Creating Python environment - first run only...
py -3 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Python 3.10 or newer was not found. Install it from python.org - tick "Add python.exe to PATH" - then run this file again.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo ERROR: package installation failed - check your internet connection and run this file again.
  rmdir /s /q .venv
  pause
  exit /b 1
)
:have_env

if exist "weights\qwen15_lora\adapter_config.json" goto have_weights
echo [setup] Downloading fine-tuned adapter...
".venv\Scripts\python.exe" scripts\download_weights.py --url "%ADAPTER_URL%" --dest weights\qwen15_lora
if errorlevel 1 (
  echo ERROR: could not download the adapter. Put the unzipped adapter folder at weights\qwen15_lora - see README.
  pause
  exit /b 1
)
:have_weights

if exist "weights\qwen05_lora\adapter_config.json" goto have_draft
echo [setup] Downloading fine-tuned draft adapter - optional...
".venv\Scripts\python.exe" scripts\download_weights.py --url "%DRAFT_ADAPTER_URL%" --dest weights\qwen05_lora
if errorlevel 1 echo NOTE: draft adapter not available; speculative mode will use the base 0.5B draft.
:have_draft

if not exist "data\processed\val.jsonl" (
  echo [setup] Preparing validation examples for the demo dropdown...
  ".venv\Scripts\python.exe" scripts\prepare_data.py --no_tokenizer --out_results data\processed\manifest_check >nul 2>&1
)

echo Starting TrialMatch with fine-tuned adapters...
".venv\Scripts\python.exe" app\app.py --adapter "weights\qwen15_lora" --draft_adapter "weights\qwen05_lora"
pause
