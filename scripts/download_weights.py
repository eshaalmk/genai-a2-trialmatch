"""Download the trained LoRA adapter(s).

Usage:
  python scripts/download_weights.py --hub_repo <hf-user>/trialmatch-qwen15-lora --dest weights/qwen15_lora
  python scripts/download_weights.py --url <direct .zip link> --dest weights/qwen15_lora

The adapter is ~70 MB; the base model (Qwen/Qwen2.5-1.5B-Instruct) is downloaded
automatically by transformers on first use."""
import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub_repo", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    dest = Path(a.dest)
    if a.hub_repo:
        from huggingface_hub import snapshot_download
        p = snapshot_download(a.hub_repo, local_dir=str(dest))
        print("downloaded to", p)
    elif a.url:
        data = urllib.request.urlopen(a.url).read()
        dest.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
        # flatten a single top-level folder
        inner = [p for p in dest.iterdir()]
        if len(inner) == 1 and inner[0].is_dir() and not (dest / "adapter_config.json").exists():
            for f in inner[0].iterdir():
                shutil.move(str(f), dest / f.name)
            inner[0].rmdir()
        print("extracted to", dest)
    else:
        ap.error("give --hub_repo or --url")
    assert (dest / "adapter_config.json").exists(), "adapter_config.json missing in download"


if __name__ == "__main__":
    main()
