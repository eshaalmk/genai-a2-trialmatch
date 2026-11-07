"""Central constants. Everything that the assignment fixes lives here."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("TM_RESULTS_DIR", REPO_ROOT / "results"))
DATA_DIR = Path(os.environ.get("TM_DATA_DIR", REPO_ROOT / "data" / "processed"))

# ---- dataset / split (fixed by the assignment) ------------------------------------
DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
SPLIT_SEED = 42
N_TRAIN, N_VAL, N_TEST = 37, 8, 8

# Columns produced by GPT-4 in the original TrialGPT study. They are dropped right
# after loading and must never reach a prompt, a label, or a metric.
FORBIDDEN_COLUMNS = ["gpt4_explanation", "explanation_correctness",
                     "gpt4_sentences", "gpt4_eligibility"]
# The dataset's own "training" flag is ignored: we use the prescribed patient split.
IGNORED_COLUMNS = ["training"]

# ---- label space (expert categories preserved, never collapsed) -------------------
INCLUSION_LABELS = ["included", "not included", "not enough information", "not applicable"]
EXCLUSION_LABELS = ["excluded", "not excluded", "not enough information", "not applicable"]
ALL_LABELS = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]
UNDECIDED_LABELS = {"not enough information", "not applicable"}


def labels_for(criterion_type: str):
    return INCLUSION_LABELS if criterion_type == "inclusion" else EXCLUSION_LABELS


# ---- models ------------------------------------------------------------------------
TARGET_MODEL = os.environ.get("TM_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DRAFT_MODEL = os.environ.get("TM_DRAFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

MAX_PROMPT_TOKENS = 1792     # prompt budget; trailing note sentences are dropped beyond it
MAX_NEW_TOKENS = 96          # JSON answers are ~15-40 tokens
