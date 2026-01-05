#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARIAN_DIR = ROOT / "models" / "MarianMT"
SD_DIR = ROOT / "models" / "stable_diffusion"

MARIAN_PAIRS = [p.name for p in MARIAN_DIR.iterdir() if p.is_dir()] if MARIAN_DIR.exists() else []

# Helper: human-readable size
def hr(n):
    for u in ["B","KB","MB","GB"]:
        if n < 1024.0:
            return f"{n:3.1f}{u}"
        n /= 1024.0
    return f"{n:.1f}TB"

print("Checking MarianMT model folders:\n")

from importlib import import_module

# Try to import transformers tokenizer class lazily
try:
    from transformers import MarianTokenizer
    HAVE_TRANSFORMERS = True
except Exception as e:
    HAVE_TRANSFORMERS = False
    print("⚠️ transformers not available (tokenizer load tests will be skipped):", e)

summary = {"marian":{}, "stable_diffusion":{}}

if not MARIAN_DIR.exists():
    print(f"MarianMT dir not found: {MARIAN_DIR}")
else:
    for pair in sorted(MARIAN_PAIRS):
        d = MARIAN_DIR / pair
        files = []
        total = 0
        for p in d.rglob('*'):
            if p.is_file():
                sz = p.stat().st_size
                files.append((p.relative_to(d).as_posix(), sz))
                total += sz
        files.sort()
        print(f"- {pair}: {len(files)} files, total {hr(total)}")
        essential = {
            'weights': any(n in ('pytorch_model.bin','model.safetensors','tf_model.h5') for n,_ in files),
            'tokenizer': any(n.endswith('.spm') or n.endswith('vocab.json') or 'tokenizer' in n for n,_ in files)
        }
        token_ok = None
        if HAVE_TRANSFORMERS:
            try:
                # attempt tokenizer load (local only)
                MarianTokenizer.from_pretrained(str(d), local_files_only=True)
                token_ok = True
            except Exception as e:
                token_ok = False
                token_err = str(e)
        summary['marian'][pair] = {
            'files_count': len(files),
            'total_size_bytes': total,
            'essential_files_present': essential,
            'tokenizer_load_ok': token_ok,
        }
        if token_ok is True:
            print(f"  tokenizer: OK")
        elif token_ok is False:
            print(f"  tokenizer: FAILED to load locally")
        else:
            print(f"  tokenizer: SKIPPED (transformers missing)")

print("\nChecking Stable Diffusion model folder:\n")
if not SD_DIR.exists():
    print(f"Stable Diffusion dir not found: {SD_DIR}")
else:
    files = []
    total = 0
    for p in SD_DIR.rglob('*'):
        if p.is_file():
            sz = p.stat().st_size
            files.append((p.relative_to(SD_DIR).as_posix(), sz))
            total += sz
    files.sort()
    print(f"- stable_diffusion: {len(files)} files, total {hr(total)}")
    # look for known large artifacts
    found_weights = [n for n,s in files if n.endswith('.safetensors') or n.endswith('.ckpt') or n.endswith('.pt') or n.endswith('.bin')]
    print(f"  large weight files found: {found_weights[:10]}")
    summary['stable_diffusion'] = {
        'files_count': len(files),
        'total_size_bytes': total,
        'weight_files': found_weights,
    }

# Print a short JSON-like summary
print("\nSummary:\n")
import json
print(json.dumps(summary, indent=2))

# Exit non-zero if any tokenizer load failed (to surface problems in CI)
fail = False
for p,v in summary['marian'].items():
    if v.get('tokenizer_load_ok') is False:
        fail = True
if fail:
    sys.exit(2)

sys.exit(0)

