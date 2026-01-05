#!/usr/bin/env python3
"""
Run specialized smoke tests for local MarianMT models.
- Discovers model folders in models/MarianMT
- For each model <src>-<tgt>, picks a short specialized sentence in the source language and translates it
- Prints results and writes a log file at scripts/test_marian_specialized.log
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
MARIAN_DIR = ROOT / "models" / "MarianMT"
LOG_PATH = ROOT / "scripts" / "test_marian_specialized.log"

# Specialized example sentences per source language (short, domain-specific)
SENTENCES = {
    "bg": [
        "Пациентът показва признаци на пневмония с висока температура и задух.",
        "Синтезирахме естер чрез реакция на карбоксилова киселина с алкохол в присъствие на концентрирана сярна киселина."
    ],
    "en": [
        "The patient presented with acute myocardial ischemia requiring immediate catheterization.",
        "We measured catalytic rate constants for the enzymatic reaction under varying pH conditions."
    ],
    "de": [
        "Der Patient zeigt Anzeichen einer Lungenembolie mit akuter Dyspnoe und Brustschmerzen."
    ],
    "fr": [
        "Le patient présente des symptômes de pneumonie nosocomiale nécessitant une antibiothérapie ciblée."
    ],
    "ru": [
        "Пациент жалуется на интенсивную головную боль и нарушение координации, подозрение на инсульт."
    ],
    "sv": [
        "Patienten uppvisar tecken på sepsis efter operation med feber och förhöjda inflammationsparametrar."
    ],
    "uk": [
        "Пацієнт має лихоманку та респіраторні симптоми; підозра на бактеріальну пневмонію."
    ],
    "ja": [
        "患者は高熱と呼吸困難を訴え、細菌性肺炎が疑われる。",
        "有機合成で芳香族求核置換反応を用いて新規化合物を合成した。"
    ],
    "zh": [
        "患者出现发热和呼吸急促，考虑细菌性肺炎。",
        "在合成化学中我们采用催化氢化反应制备了目标分子。"
    ]
}

GENERIC = "This is a short specialized sentence for testing translation quality in a technical domain."

results = []

if not MARIAN_DIR.exists():
    print(f"No MarianMT dir found at {MARIAN_DIR}")
    sys.exit(2)

model_dirs = sorted([p for p in MARIAN_DIR.iterdir() if p.is_dir()])

from transformers import MarianTokenizer, MarianMTModel
import torch

for mdir in model_dirs:
    name = mdir.name
    # expect name like 'bg-en'
    if '-' not in name:
        print(f"Skipping unexpected folder name: {name}")
        continue
    src, tgt = name.split('-', 1)
    # pick a sentence for src language
    sent_list = SENTENCES.get(src, [GENERIC])
    text = sent_list[0]

    entry = {"model": name, "src": src, "tgt": tgt, "input": text, "output": None, "error": None}

    try:
        print(f"\n=== Testing {name}: {src} -> {tgt} ===")
        print("Loading tokenizer and model (local only)...")
        tokenizer = MarianTokenizer.from_pretrained(str(mdir), local_files_only=True)
        model = MarianMTModel.from_pretrained(str(mdir), local_files_only=True)
        # put on CPU (or GPU if available) - keep it simple
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        inputs = tokenizer([text], return_tensors='pt', padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print("Generating...")
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_length=200)
        out_text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        out_text = out_text[0] if out_text else ""
        print("Input:", text)
        print("Output:", out_text)
        entry['output'] = out_text
        # free
        del model
        del tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error testing {name}: {e}")
        entry['error'] = str(e)
    results.append(entry)

# write results to log
LOG_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print("\nAll tests complete. Results written to", LOG_PATH)
print(json.dumps(results, indent=2, ensure_ascii=False))

