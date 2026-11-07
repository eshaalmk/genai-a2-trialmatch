"""LoRA fine-tuning. Checkpoints go to --output_dir (put it on Google Drive in Colab);
re-running the same command after a disconnect resumes from the latest checkpoint."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trialmatch.train import TrainConfig, plot_curves, train


def main():
    ap = argparse.ArgumentParser()
    d = TrainConfig()
    for k, v in vars(d).items():
        if isinstance(v, list):
            ap.add_argument(f"--{k}", nargs="+", default=v)
        elif isinstance(v, bool):
            ap.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--results_dir", default=None, help="copy logs/curves here (e.g. results/training_qwen15)")
    a = vars(ap.parse_args())
    results_dir = a.pop("results_dir")
    cfg = TrainConfig(**a)
    summary = train(cfg)
    out = Path(cfg.output_dir)
    plot_curves(out / "log_history.json", out / "training_curves.png")
    if results_dir:
        import shutil
        r = Path(results_dir)
        r.mkdir(parents=True, exist_ok=True)
        for f in ("train_config.json", "train_summary.json", "train_data_stats.json",
                  "log_history.json", "training_curves.png"):
            if (out / f).exists():
                shutil.copy(out / f, r / f)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
