#!/usr/bin/env python3
"""
Upload model files to a Hugging Face model repo using the huggingface_hub HfApi.upload_file.
This version adds a per-file progress bar using tqdm by wrapping the file object that is streamed
to the HF REST API, and also an aggregate progress bar that tracks total bytes uploaded across
all files. It keeps the retry / exponential backoff behavior and reports an overall
summary and percent completion.

Usage:
    export HF_TOKEN="$(cat secrets/hf_aurora.txt)"
    python scripts/upload_models_to_hf.py --repo kenderov-emil4108/aurora-models --path models/

Flags added:
    --dry-run   : lists files that would be uploaded and total size, then exits
    --resume    : queries the HF repo for existing file names and skips any files with the same basename

Note: For very large files you may prefer to push with git-lfs or the hf_hub CLI. This script is
intended for reliable REST uploads and will retry transient failures.
"""
import argparse
import io
import os
import time
from huggingface_hub import HfApi
from tqdm import tqdm

MODEL_EXTS = {".gguf", ".bin", ".safetensors", ".pt", ".ckpt"}


class ProgressBufferedReader(io.BufferedReader):
    """A BufferedReader that updates tqdm progress bars on read().

    This class is a subclass of io.BufferedReader (thus an instance of io.BufferedIOBase),
    which satisfies HfApi.upload_file's type checks when passing a file-like object.
    """

    def __init__(self, raw, pbar: tqdm, aggregate_pbar: tqdm = None):
        # raw should be a raw file-like (e.g. FileIO)
        super().__init__(raw)
        self._pbar = pbar
        self._aggregate_pbar = aggregate_pbar

    def read(self, size=-1):
        chunk = super().read(size)
        if chunk:
            n = len(chunk)
            try:
                self._pbar.update(n)
            except Exception:
                pass
            if self._aggregate_pbar is not None:
                try:
                    self._aggregate_pbar.update(n)
                except Exception:
                    pass
        return chunk


def iter_model_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in MODEL_EXTS:
                yield os.path.join(dirpath, fn)


def upload_file_with_retries(api, repo_id, token, local_path, path_in_repo, max_retries=5, aggregate_pbar=None):
    backoff = 1.0
    file_size = os.path.getsize(local_path)
    for attempt in range(1, max_retries + 1):
        try:
            # open raw file and wrap its raw FileIO in ProgressBufferedReader
            f = open(local_path, "rb")
            try:
                raw = getattr(f, "raw", f)
                with tqdm(total=file_size, unit="B", unit_scale=True, desc=os.path.basename(local_path)) as pbar:
                    wrapped = ProgressBufferedReader(raw, pbar, aggregate_pbar)
                    # HfApi.upload_file will read from the file-like object
                    api.upload_file(
                        path_or_fileobj=wrapped,
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        token=token,
                    )
                    # ensure wrapped is closed
                    try:
                        wrapped.close()
                    except Exception:
                        pass
            finally:
                try:
                    f.close()
                except Exception:
                    pass
            return True
        except Exception as e:
            print(f"Upload failed (attempt {attempt}) for {local_path}: {e}")
            if attempt == max_retries:
                return False
            time.sleep(backoff)
            backoff *= 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Hugging Face repo id (e.g. user/repo)")
    p.add_argument("--path", required=True, help="Root path to search for model files")
    p.add_argument("--token-env", default="HF_TOKEN", help="Environment variable that contains the HF token")
    p.add_argument("--max-retries", type=int, default=5, help="Max upload retries per file")
    p.add_argument("--dry-run", action="store_true", help="Print files and total size and exit")
    p.add_argument("--resume", action="store_true", help="Skip files whose basename already exists in the HF repo")
    args = p.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"Environment variable {args.token_env} not set. Set it to your Hugging Face token.")

    api = HfApi()
    # create repo if not exists (private by default)
    try:
        api.create_repo(repo_id=args.repo, token=token, private=True, repo_type="model")
        print(f"Created repo {args.repo}")
    except Exception as e:
        # 409 or similar if already exists
        print(f"Could not create repo (it may already exist): {e}")

    found = list(iter_model_files(args.path))
    if not found:
        print("No model files found to upload.")
        return

    total_files = len(found)
    total_bytes = sum(os.path.getsize(p) for p in found)

    # dry-run: just print and exit
    if args.dry_run:
        print(f"Dry-run: found {total_files} model file(s); total size: {total_bytes / (1024**2):.2f} MB")
        for p in found:
            print(p, os.path.getsize(p))
        return

    # if resume is requested, list remote files and build a skip set
    skip_basenames = set()
    if args.resume:
        try:
            remote_files = api.list_repo_files(args.repo, token=token)
            # remote_files contains paths; we compare basenames
            skip_basenames = {os.path.basename(r) for r in remote_files}
            print(f"Resume: found {len(skip_basenames)} remote file(s); will skip matching basenames")
        except Exception as e:
            print(f"Could not list remote files for resume: {e}")

    print(f"Found {total_files} model file(s) to upload; total size: {total_bytes / (1024**2):.2f} MB")

    success = []
    failed = []

    # Aggregate progress bar across all files (but subtract sizes of skipped files)
    to_upload = []
    for pth in found:
        name = os.path.basename(pth)
        if name in skip_basenames:
            print(f"Skipping {name} (already present in remote)")
            continue
        to_upload.append(pth)

    total_bytes_upload = sum(os.path.getsize(p) for p in to_upload)

    with tqdm(total=total_bytes_upload, unit="B", unit_scale=True, desc="Total") as aggregate_pbar:
        for idx, pth in enumerate(to_upload, start=1):
            name = os.path.basename(pth)
            print(f"[{idx}/{len(to_upload)}] Uploading {name}...")
            ok = upload_file_with_retries(api, args.repo, token, pth, name, max_retries=args.max_retries, aggregate_pbar=aggregate_pbar)
            if ok:
                success.append(name)
            else:
                failed.append(name)

    print("Upload summary:")
    print(f"  Uploaded: {len(success)} files")
    for s in success:
        print(f"    - {s}")
    if failed:
        print(f"  Failed: {len(failed)} files")
        for f in failed:
            print(f"    - {f}")
        raise SystemExit("Some uploads failed; see above")


if __name__ == "__main__":
    main()

