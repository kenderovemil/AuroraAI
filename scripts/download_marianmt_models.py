import os
import time
import random
import shutil
import errno

# Read token first
TOKEN_PATH = "/run/media/msi/C086D41086D408B41/AuroraAI/secrets/hf_aurora.txt"
with open(TOKEN_PATH, "r") as f:
    token = f.read().strip()

# Set an internal HF cache (prefer an internal SSD) to avoid writing partial files to the slow external drive
# You can override by setting HF_HOME in your environment before running the script
INTERNAL_CACHE = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
os.makedirs(INTERNAL_CACHE, exist_ok=True)
os.environ["HF_HOME"] = INTERNAL_CACHE

# Add a simple lockfile to avoid concurrent huggingface downloads (Stable Diffusion etc.)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_PATH = os.path.join(PROJECT_ROOT, ".hf_download_lock")
LOCK_STALE_SECONDS = 60 * 60  # 1 hour


def acquire_lock(lock_path, stale_seconds=LOCK_STALE_SECONDS, wait=True):
    pid = os.getpid()
    content = f"{pid}\n{int(time.time())}\n"
    while True:
        try:
            # O_EXCL ensures we fail if file exists
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(fd, "w") as f:
                f.write(content)
            print(f"🔒 Acquired download lock: {lock_path}")
            return True
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            # lock exists: check age
            try:
                with open(lock_path, "r") as f:
                    lines = f.read().splitlines()
                    lock_pid = lines[0] if lines else "?"
                    lock_ts = int(lines[1]) if len(lines) > 1 else 0
            except Exception:
                lock_pid = "?"
                lock_ts = 0
            age = time.time() - lock_ts
            if age > stale_seconds:
                print(f"⚠️ Stale lock found (age {int(age)}s). Removing and retrying.")
                try:
                    os.remove(lock_path)
                except Exception:
                    time.sleep(1)
                    continue
            if not wait:
                print(f"⏭ Lock present and wait==False; not acquiring.")
                return False
            print(f"🔁 Waiting for existing lock (pid={lock_pid}, age={int(age)}s)...")
            time.sleep(5)


def release_lock(lock_path):
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
            print(f"🔓 Released download lock: {lock_path}")
    except Exception as e:
        print(f"⚠️ Failed to remove lock file: {e}")

# Import HF download utilities after HF_HOME is set
from huggingface_hub import snapshot_download, hf_hub_download

# Language pairs to download
language_pairs = [
    "bg-en", "en-bg",
    "bg-de", "de-bg",
    "bg-fr", "fr-bg",
    "bg-ru", "ru-bg",
    "bg-uk", "uk-bg",
    "bg-hi", "hi-bg",
    "bg-ja", "ja-bg",
    "bg-zh", "zh-bg",
    "bg-sv", "sv-bg",
    "bg-no", "no-bg",
    "bg-da", "da-bg"
]

# External (final) directory on your drive
BASE_DIR = "/run/media/msi/C086D41086D408B41/AuroraAI/models/MarianMT"
os.makedirs(BASE_DIR, exist_ok=True)

# Temp area inside the internal cache to download each repo before moving it
TMP_BASE = os.path.join(INTERNAL_CACHE, "marian_tmp")
os.makedirs(TMP_BASE, exist_ok=True)

MAX_RETRIES = 5
INITIAL_BACKOFF = 5  # seconds

# Common large filenames to prefetch with extended timeout
COMMON_LARGE_FILES = [
    "pytorch_model.bin",
    "model.safetensors",
    "pytorch_model.bin.index.json",
    "tf_model.h5",
    "source.spm",
    "target.spm",
    "vocab.json",
]

# Acquire global lock so we don't run concurrently with other download scripts (e.g., SD)
acquired = acquire_lock(LOCK_PATH)
try:
    for pair in language_pairs:
        model_id = f"Helsinki-NLP/opus-mt-{pair}"
        save_path = os.path.join(BASE_DIR, pair)
        print(f"⬇️ Downloading {model_id} to {save_path}...")

        # If the target already has files, skip
        if os.path.isdir(save_path) and os.listdir(save_path):
            print(f"ℹ️ Skipping {pair}: target directory already contains files.")
            continue

        tmp_dir = os.path.join(TMP_BASE, pair)
        # Clean any previous incomplete tmp dir
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

        # Prefetch common large files individually with a long timeout; ignore failures (snapshot_download is fallback)
        for filename in COMMON_LARGE_FILES:
            try:
                print(f"  ⤷ prefetching {filename} (long timeout)...")
                hf_hub_download(repo_id=model_id, filename=filename, cache_dir=INTERNAL_CACHE, token=token, timeout=1800)
            except Exception:
                # Not all files exist for every model, ignore
                pass

        attempt = 0
        success = False
        last_error = None
        while attempt < MAX_RETRIES and not success:
            try:
                attempt += 1
                print(f"  attempt {attempt}/{MAX_RETRIES}...")
                # Download the full repo snapshot into a temp folder on the internal cache
                snapshot_download(repo_id=model_id, local_dir=tmp_dir, cache_dir=INTERNAL_CACHE, token=token)
                success = True
            except Exception as e:
                last_error = e
                backoff = INITIAL_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"  ⚠️ Download failed (attempt {attempt}): {e}")
                if attempt < MAX_RETRIES:
                    print(f"  Retrying in {int(backoff)}s...")
                    time.sleep(backoff)

        if not success:
            print(f"❌ Failed to download {model_id} after {MAX_RETRIES} attempts: {last_error}")
            # cleanup partial tmp
            if os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
            continue

        # Move the completed tmp dir contents to the external target atomically
        try:
            os.makedirs(save_path, exist_ok=True)
            for name in os.listdir(tmp_dir):
                src = os.path.join(tmp_dir, name)
                dst = os.path.join(save_path, name)
                # Use shutil.move which handles cross-device moves by copying then removing
                shutil.move(src, dst)
            # remove the empty tmp_dir
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
        except Exception as e:
            print(f"❌ Error moving files to {save_path}: {e}")
            # leave tmp_dir for inspection
            continue

        # Verify by loading tokenizer + model locally-only (no network)
        try:
            from transformers import MarianTokenizer, MarianMTModel
            tokenizer = MarianTokenizer.from_pretrained(save_path, local_files_only=True)
            model = MarianMTModel.from_pretrained(save_path, local_files_only=True)
            # free memory (models can be large)
            del tokenizer
            del model
            print(f"✅ Done: {pair}")
        except Exception as e:
            print(f"⚠️ Download succeeded but loading locally failed for {pair}: {e}")
            print(f"  You can still try to use the files in {save_path} or inspect {tmp_dir}.")

    print("All done.")
finally:
    if acquired:
        release_lock(LOCK_PATH)
