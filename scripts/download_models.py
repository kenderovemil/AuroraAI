#!/usr/bin/env python3
"""
Download all files from a Hugging Face model repository into a local directory using
huggingface_hub.snapshot_download().

Usage:
    export HF_TOKEN="$(cat secrets/hf_aurora.txt)"
    python scripts/download_models.py --repo-id kenderov-emil4108/aurora-models --out models/

This script does not store tokens in the repo. It will use the HF_TOKEN environment variable
or a token passed via --token for private repositories.
"""
import argparse
import os
import sys
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="Hugging Face repo id (e.g. user/repo)")
    p.add_argument("--out", required=True, help="Output directory to store downloaded files")
    p.add_argument("--revision", default=None, help="Revision/branch to download (optional)")
    p.add_argument("--token-env", default="HF_TOKEN", help="Environment variable that contains the HF token")
    p.add_argument("--token", default=None, help="Hugging Face token (use only if you can't set env var)")
    args = p.parse_args()

    token = args.token or os.environ.get(args.token_env)
    if not token:
        print(f"Warning: No token provided via --token or ${args.token_env}. Public repos may still download.")

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading snapshot of {args.repo_id} into {out_dir}...")
    try:
        local_path = snapshot_download(repo_id=args.repo_id, revision=args.revision, cache_dir=out_dir, token=token)
        print(f"Downloaded files are available under: {local_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()

