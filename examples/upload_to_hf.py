"""
Upload the RISED synthetic cohort dataset to HuggingFace Hub.

Prereqs:
1. pip install huggingface_hub  (already installed)
2. Get a write token at https://huggingface.co/settings/tokens
3. Set env var: HF_TOKEN=<your_token>   OR   run `hf auth login` first.

Usage:
    python upload_to_hf.py [--repo-id rohithreddybc/rised-synthetic-cohort-10k]
                           [--private]
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file


DEFAULT_REPO_ID = "rohithreddybc/rised-synthetic-cohort-10k"
HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                        help="HF repo id (username/dataset-name)")
    parser.add_argument("--private", action="store_true",
                        help="Create as a private dataset")
    parser.add_argument("--token", default=None,
                        help="HF token (overrides HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: No HF token found. Either:")
        print("  1. Set HF_TOKEN environment variable, or")
        print("  2. Run `hf auth login` first, or")
        print("  3. Pass --token <token>")
        sys.exit(1)

    csv_path = HERE / "synthetic_cohort_10k.csv"
    card_path = HERE / "HF_DATASET_CARD.md"
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)
    if not card_path.exists():
        print(f"ERROR: Dataset card not found at {card_path}")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"Creating repo: {args.repo_id} ({'private' if args.private else 'public'})")
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    print(f"Uploading dataset card -> {args.repo_id}/README.md")
    upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add dataset card",
    )

    print(f"Uploading CSV -> {args.repo_id}/synthetic_cohort_10k.csv ({csv_path.stat().st_size/1024:.1f} KB)")
    upload_file(
        path_or_fileobj=str(csv_path),
        path_in_repo="synthetic_cohort_10k.csv",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add 10K synthetic patient cohort CSV",
    )

    url = f"https://huggingface.co/datasets/{args.repo_id}"
    print()
    print(f"Done! View dataset at: {url}")
    print(f"Load with: datasets.load_dataset('{args.repo_id}')")


if __name__ == "__main__":
    main()
