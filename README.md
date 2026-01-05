# AuroraAI
# AuroraAI 🌌

AuroraAI is an intelligent assistant inspired by the northern lights and the legends of Sleeping Beauty.  
The project aims to create a modular, expandable, and aesthetically elegant AI system with a focus on:

- 💬 Natural language and dialogue
- 🧠 Memory and context
- 📊 Data processing
- 🖼️ Visual capabilities (optional)



Project scaffold for AuroraAI.

Tree:

AuroraAI/
├── README.md
├── requirements.txt
├── .gitignore
├── aurora/                  # Основна логика на AuroraAI
│   ├── __init__.py
│   ├── core.py              # Ядро на логиката
│   ├── memory.py            # Дългосрочна памет / контекст
│   ├── planner.py           # Модул за планиране и задачи
│   ├── nlp_tools.py         # Езикови инструменти
│   └── vision.py            # Ако ще има визуални възможности
├── models/                  # Модели и конфигурации
│   ├── config/
│   └── checkpoints/
├── data/                    # Данни, които Aurora обработва
│   ├── raw/
│   └── processed/
├── scripts/                 # Помощни скриптове
│   └── setup_env.sh
├── tests/                   # Тестове
│   └── test_core.py
└── notebooks/               # Jupyter експерименти
    └── exploration.ipynb

Description
-----------
This repository is an minimal scaffold to start building AuroraAI: core logic, memory, planning, simple NLP/vision placeholders, and a basic test harness.

How to use
----------
1. Create a virtual environment and install dependencies:

```bash
bash scripts/setup_env.sh
```

2. Run the unit tests:

```bash
python -m unittest discover -v
```

3. Start implementing features under `aurora/`.


## Models
This repository separates code and model binaries. Model files (GGUF, PyTorch, Safetensors, checkpoints, etc.) are hosted on the Hugging Face Hub at:

https://huggingface.co/kenderov-emil4108/aurora-models

Do not commit model files, tokens, logs, or outputs to this Git repository. Model files are explicitly excluded via `.gitignore`.

To download the models into your local `models/` directory, run (from the project root):

```bash
export HF_TOKEN="$(cat secrets/hf_aurora.txt)"
python scripts/download_models.py --repo-id kenderov-emil4108/aurora-models --out models/
```

This script uses `huggingface_hub` to download all files stored in the `kenderov-emil4108/aurora-models` repo into the `models/` directory. Make sure your `secrets/hf_aurora.txt` contains a valid Hugging Face token with `read` access.

If you prefer to manually download models, visit the Hugging Face repo page linked above and use the web UI.

### Uploading models to Hugging Face (maintainers only)

Use the included upload script which streams files and shows progress per file:

```bash
export HF_TOKEN="$(cat secrets/hf_aurora.txt)"
python scripts/upload_models_to_hf.py --repo kenderov-emil4108/aurora-models --path models/
```

Notes:
- The script will attempt to create the HF repo if it does not exist. It will upload only model file extensions (`.gguf`, `.bin`, `.safetensors`, `.pt`, `.ckpt`).
- For very large (>2GB) files you may prefer to use the `huggingface-cli` or `git-lfs`.

### Security & best practices

- Never commit secrets or tokens. The `secrets/` directory is ignored by `.gitignore` and should remain local-only.
- When pushing code to GitHub, verify that `git status --porcelain` shows no secrets or models staged.
- If large files or tokens were accidentally committed in the past, follow the `git filter-repo` / BFG steps to purge them from history before pushing.
