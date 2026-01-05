#!/usr/bin/env python3
"""
scripts/monitor_hf_upload.py

Monitor a Hugging Face model repo upload progress by comparing local model files to remote files.
It computes total local bytes (for tracked model extensions) and queries the HF repo to sum
remote file sizes where available, then prints percent complete and remaining bytes.

Usage:
  export HF_TOKEN="$(cat secrets/hf_aurora.txt)"
  python scripts/monitor_hf_upload.py --repo kenderov-emil4108/aurora-models --local models/

Notes:
- The script uses the Hugging Face REST API via huggingface_hub.HfApi.list_repo_files and
  huggingface_hub.hf_api.HfApi().info_file to gather remote sizes where available.
- If remote size is not available via list_repo_files, the script will treat that file as 0 until it's uploaded.
"""
import argparse
import os
import time
from huggingface_hub import HfApi

MODEL_EXTS = {'.gguf','.bin','.safetensors','.pt','.ckpt'}


def iter_model_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in MODEL_EXTS:
                yield os.path.join(dirpath, fn)


def human_bytes(n):
    # simple human readable conversion
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True, help='HF repo id (user/repo)')
    p.add_argument('--local', default='models/', help='Local models root')
    p.add_argument('--interval', type=int, default=30, help='Polling interval in seconds')
    args = p.parse_args()

    api = HfApi()

    while True:
        local_files = list(iter_model_files(args.local))
        total_local = sum(os.path.getsize(p) for p in local_files)

        try:
            remote_files = api.list_repo_files(args.repo)
        except Exception as e:
            print(f"Could not list remote files: {e}")
            remote_files = []

        # map basename -> remote path
        remote_map = {os.path.basename(p): p for p in remote_files}

        total_remote = 0
        found_remote = 0
        for p in local_files:
            bn = os.path.basename(p)
            if bn in remote_map:
                # try to get info for this file (size not always returned in list)
                try:
                    info = api.info_repo_file(args.repo, remote_map[bn])
                    size = int(info['size']) if info and 'size' in info else 0
                except Exception:
                    size = 0
                total_remote += size
                found_remote += 1

        percent = (total_remote / total_local * 100) if total_local > 0 else 0
        remaining = total_local - total_remote

        print(f"Local total: {human_bytes(total_local)} | Remote total: {human_bytes(total_remote)} | {percent:.1f}% complete | Remaining: {human_bytes(remaining)} | Files found remote: {found_remote}/{len(local_files)}")

        time.sleep(args.interval)


if __name__ == '__main__':
    main()

