"""Upload a trained adapter folder to the Hugging Face Hub (free account; needs HF_TOKEN
with write access)."""
import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo", required=True, help="<hf-username>/<repo-name>")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(a.repo, exist_ok=True, private=a.private, repo_type="model")
    api.upload_folder(folder_path=a.folder, repo_id=a.repo, repo_type="model",
                      commit_message="Upload trialmatch LoRA adapter")
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
