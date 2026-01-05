#!/usr/bin/env python3
"""
Safe repair and re-download for MarianMT models.

Usage:
  # dry-run (shows what would be done):
  python scripts/repair_redownload_marianmt.py

  # perform destructive repair (backup then redownload):
  python scripts/repair_redownload_marianmt.py --yes

Options:
  --yes            Actually delete/backup and re-download broken models (default is dry-run)
  --no-backup      Do not keep backups of the removed folders (default: keep backups)
  --hf-home PATH   Override HF cache directory (default: $HF_HOME or ~/.cache/huggingface)
  --pairs a b c    Optional list of specific language pairs to repair (e.g., bg-da en-bg)

The script detects "broken" models as folders with missing weights or tokenizer files, or tokenizer failing to load locally.

It downloads snapshots into an internal cache tmp dir, then moves them atomically to the external models folder.
"""

import argparse
import os
import sys
import time
import random
import shutil
import errno
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARIAN_DIR = ROOT / "models" / "MarianMT"
TOKEN_PATH = ROOT / "secrets" / "hf_aurora.txt"

# Read token if available
if not TOKEN_PATH.exists():
    print(f"⚠️ Token file not found: {TOKEN_PATH}. Downloads will fail unless you set token manually.")
    token = None
else:
    token = TOKEN_PATH.read_text().strip()

# Export token to common HF env vars so huggingface_hub functions authenticate whether passed a token or rely on env
if token:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    os.environ.setdefault("HF_HUB_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_TOKEN", token)
    os.environ.setdefault("HF_TOKEN", token)

# lock file used to avoid concurrent HF downloads
LOCK_PATH = ROOT / ".hf_download_lock"
LOCK_STALE_SECONDS = 60 * 60  # 1 hour


def acquire_lock(lock_path, stale_seconds=LOCK_STALE_SECONDS, wait=True):
    pid = os.getpid()
    content = f"{pid}\n{int(time.time())}\n"
    lock_path = Path(lock_path)
    while True:
        try:
            # exclusive create
            fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(fd, "w") as f:
                f.write(content)
            print(f"🔒 Acquired download lock: {lock_path}")
            return True
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            # read existing lock
            try:
                txt = lock_path.read_text()
                lines = txt.splitlines()
                lock_pid = lines[0] if lines else "?"
                lock_ts = int(lines[1]) if len(lines) > 1 else 0
            except Exception:
                lock_pid = "?"
                lock_ts = 0
            age = time.time() - lock_ts
            if age > stale_seconds:
                print(f"⚠️ Stale lock found (age {int(age)}s). Removing and retrying.")
                try:
                    lock_path.unlink()
                except Exception:
                    time.sleep(1)
                    continue
            if not wait:
                print("⏭ Lock present and wait==False; not acquiring.")
                return False
            print(f"🔁 Waiting for existing lock (pid={lock_pid}, age={int(age)}s)...")
            time.sleep(3)


def release_lock(lock_path):
    try:
        if Path(lock_path).exists():
            Path(lock_path).unlink()
            print(f"🔓 Released download lock: {lock_path}")
    except Exception as e:
        print(f"⚠️ Failed to remove lock file: {e}")


# detection logic for a "good" model folder
ESSENTIAL_WEIGHT_NAMES = ("pytorch_model.bin", "model.safetensors", "tf_model.h5")
TOKENIZER_PATTERNS = (".spm", "vocab.json", "tokenizer_config.json")


def model_folder_health(path: Path):
    """Return (is_ok:bool, details:dict)"""
    if not path.exists() or not path.is_dir():
        return False, {"reason": "missing_dir"}
    files = [p for p in path.rglob("*") if p.is_file()]
    if not files:
        return False, {"reason": "empty_dir"}
    names = [p.name for p in files]
    has_weight = any(n in ESSENTIAL_WEIGHT_NAMES for n in names)
    has_tokenizer = any(any(n.endswith(suf) for suf in TOKENIZER_PATTERNS) or 'tokenizer' in n for n in names)
    return (has_weight and has_tokenizer), {"has_weight": has_weight, "has_tokenizer": has_tokenizer, "files_count": len(files)}


def find_cache_snapshot_for_pair(internal_cache: Path, pair: str):
    """Look for cache snapshot directories matching the Helsinki naming for pair (try both orderings)."""
    candidates = []
    patterns = [f"models--Helsinki-NLP--opus-mt-{pair}", f"models--Helsinki-NLP--opus-mt-{pair.replace('-','-')}"]
    # also consider swapped pair (en-bg vs bg-en)
    alt = "-".join(pair.split("-")[::-1])
    patterns.append(f"models--Helsinki-NLP--opus-mt-{alt}")
    for p in internal_cache.iterdir() if internal_cache.exists() else []:
        if p.is_dir() and any(p.name.startswith(pat) for pat in patterns):
            candidates.append(p)
    return candidates


def safe_move(src: Path, dst: Path):
    """Move across devices: try shutil.move, else copytree/copy2 then remove src."""
    src = Path(src)
    dst = Path(dst)
    try:
        shutil.move(str(src), str(dst))
        return True
    except Exception:
        # fallback: copy
        try:
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(str(dst))
                shutil.copytree(str(src), str(dst))
                shutil.rmtree(str(src))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                src.unlink()
            return True
        except Exception as e:
            print(f"❌ Failed to move {src} -> {dst}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Perform changes instead of dry-run")
    parser.add_argument("--no-backup", action="store_true", help="Do not keep backups of removed folders")
    parser.add_argument("--hf-home", type=str, default=os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface"))
    parser.add_argument("--pairs", nargs="*", help="Specific pairs to repair (e.g., bg-da en-bg)")
    args = parser.parse_args()

    internal_cache = Path(args.hf_home)
    print(f"HF cache: {internal_cache}")
    print("Scanning MarianMT model folders...\n")

    existing_pairs = sorted([p.name for p in MARIAN_DIR.iterdir() if p.is_dir()]) if MARIAN_DIR.exists() else []
    target_pairs = args.pairs if args.pairs else existing_pairs

    broken = []
    ok = []

    # try importing tokenizer to validate further
    try:
        from transformers import MarianTokenizer
        HAVE_TF = True
    except Exception:
        HAVE_TF = False

    for pair in target_pairs:
        path = MARIAN_DIR / pair
        is_ok, details = model_folder_health(path)
        # if tokenizer check is still plausible, attempt load
        token_ok = None
        if is_ok and HAVE_TF:
            try:
                MarianTokenizer.from_pretrained(str(path), local_files_only=True)
                token_ok = True
            except Exception:
                token_ok = False
        if is_ok and token_ok is not False:
            ok.append((pair, details))
        else:
            broken.append((pair, details))

    print("Found OK models:")
    for p,_ in ok:
        print("  ", p)
    print("\nFound broken/missing models:")
    for p,d in broken:
        print(f"  {p}: {d}")

    if not args.yes:
        print("\nDry-run mode (no changes). To repair these models run with --yes")
        return

    # perform repair
    acquired = acquire_lock(LOCK_PATH)
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        MAX_RETRIES = 5
        INITIAL_BACKOFF = 5

        for pair, details in broken:
            model_id = f"Helsinki-NLP/opus-mt-{pair}"
            target_dir = MARIAN_DIR / pair
            # backup or remove existing target
            if target_dir.exists() and any(target_dir.iterdir()):
                if not args.no_backup:
                    bak = Path(str(target_dir) + ".bak." + time.strftime('%Y%m%d-%H%M%S'))
                    print(f"Backing up {target_dir} -> {bak}")
                    shutil.move(str(target_dir), str(bak))
                else:
                    print(f"Removing existing folder {target_dir}")
                    shutil.rmtree(str(target_dir))
            # Ensure clean tmp dir
            tmp = internal_cache / "marian_tmp" / pair
            if tmp.exists():
                try:
                    shutil.rmtree(str(tmp))
                except Exception:
                    pass
            tmp.mkdir(parents=True, exist_ok=True)

            # Prefetch large known files with generous timeout
            COMMON_LARGE_FILES = ["pytorch_model.bin", "model.safetensors", "tf_model.h5", "vocab.json", "source.spm", "target.spm"]
            for fname in COMMON_LARGE_FILES:
                try:
                    print(f"  prefetch {fname}...")
                    hf_hub_download(repo_id=model_id, filename=fname, cache_dir=str(internal_cache), token=token, timeout=1800)
                except Exception:
                    pass

            attempt = 0
            success = False
            last_err = None
            while attempt < MAX_RETRIES and not success:
                try:
                    attempt += 1
                    print(f"  snapshot_download attempt {attempt} for {model_id} -> {tmp} ...")
                    snapshot_download(repo_id=model_id, local_dir=str(tmp), cache_dir=str(internal_cache), token=token)
                    success = True
                except Exception as e:
                    last_err = e
                    backoff = INITIAL_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 2)
                    print(f"   ⚠️ failed: {e}; retrying in {int(backoff)}s")
                    time.sleep(backoff)

            if not success:
                print(f"❌ Could not download {model_id}: {last_err}")
                continue

            # move tmp -> target_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            for name in os.listdir(str(tmp)):
                src = tmp / name
                dst = target_dir / name
                print(f"  moving {src} -> {dst}")
                safe_move(src, dst)
            try:
                shutil.rmtree(str(tmp))
            except Exception:
                pass

            # verify
            try:
                from transformers import MarianTokenizer
                MarianTokenizer.from_pretrained(str(target_dir), local_files_only=True)
                print(f"✅ Repaired and verified {pair}")
            except Exception as e:
                print(f"⚠️ Repaired {pair} but verifier failed: {e}")

    finally:
        if acquired:
            release_lock(LOCK_PATH)


if __name__ == '__main__':
    main()
