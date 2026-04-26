"""
upload_hf_model_cards.py — One-click script to:
  1. Upload polished model card (README) to jdsb06/lifestack-grpo-v4  (with training plots)
  2. Upload polished model card (README) to jdsb06/lifestack-grpo     (v1)
  3. Upload train_run_v1.log to jdsb06/lifestack-grpo                  (v1 training log)
  4. Upload docs/blog.md to both model repos as blog.md

Run from the repo root:
    python scripts/upload_hf_model_cards.py

Requires: pip install huggingface_hub
          HF_TOKEN env var set, OR will prompt for login.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("[upload] huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


def get_api() -> HfApi:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[upload] HF_TOKEN not set — opening browser login...")
        login()
    return HfApi(token=token)


def upload_v4_model_card(api: HfApi) -> None:
    repo_id = "jdsb06/lifestack-grpo-v4"
    card_path = REPO_ROOT / "docs" / "HF_MODEL_CARD_V4.md"
    plots_dir = REPO_ROOT / "plots"

    print(f"\n[upload] Uploading model card to {repo_id}...")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card: add blog narrative, training plots, full v4 training story",
    )
    print(f"  ✅ README.md uploaded to {repo_id}")

    # Upload any plots not already there
    for png in ["reward_curve.png", "loss_curve.png", "reward_components.png", "training_summary.png"]:
        src = plots_dir / png
        if src.exists():
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=f"plots/{png}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add training evidence plot: {png}",
            )
            print(f"  ✅ plots/{png} uploaded")
        else:
            print(f"  ⚠️  {src} not found — skipping")


def upload_v1_model_card(api: HfApi) -> None:
    repo_id = "jdsb06/lifestack-grpo"
    card_path = REPO_ROOT / "docs" / "HF_MODEL_CARD_V1.md"
    log_path  = REPO_ROOT / "train_run_v1.log"

    print(f"\n[upload] Uploading model card to {repo_id}...")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card: v1 training details, log reference, link to v4",
    )
    print(f"  ✅ README.md uploaded to {repo_id}")

    if log_path.exists():
        print(f"\n[upload] Uploading train_run_v1.log to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(log_path),
            path_in_repo="train_run_v1.log",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add v1 training log (stages 4-5, Unsloth TRL on T4)",
        )
        print(f"  ✅ train_run_v1.log uploaded to {repo_id}")
    else:
        print(f"  ⚠️  {log_path} not found — skipping log upload")


def upload_blog_md_to_both_repos(api: HfApi) -> None:
    blog_path = REPO_ROOT / "docs" / "blog.md"
    if not blog_path.exists():
        print(f"\n[upload] {blog_path} not found — skipping blog upload")
        return

    for repo_id in ("jdsb06/lifestack-grpo-v4", "jdsb06/lifestack-grpo"):
        print(f"\n[upload] Uploading blog.md to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(blog_path),
            path_in_repo="blog.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update blog.md from docs/blog.md",
        )
        print(f"  ✅ blog.md uploaded to {repo_id}")


def main() -> None:
    print("=" * 60)
    print("  LifeStack HF Model Card Uploader")
    print("=" * 60)

    api = get_api()

    upload_v4_model_card(api)
    upload_v1_model_card(api)
    upload_blog_md_to_both_repos(api)

    print("\n" + "=" * 60)
    print("  Done! Check your model cards:")
    print("  https://huggingface.co/jdsb06/lifestack-grpo-v4")
    print("  https://huggingface.co/jdsb06/lifestack-grpo")
    print("=" * 60)


if __name__ == "__main__":
    main()
