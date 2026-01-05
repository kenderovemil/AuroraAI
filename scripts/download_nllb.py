from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download
import os
import sys

# Прочитане на токена
with open("/run/media/msi/C086D41086D408B41/AuroraAI/secrets/hf_aurora.txt", "r") as f:
    token = f.read().strip()

# Модел и директория
model_id = "facebook/nllb-200-distilled-600M"
save_path = "/run/media/msi/C086D41086D408B41/AuroraAI/models/NLLB"

# Създаваме директория, ако не съществува
os.makedirs(save_path, exist_ok=True)

try:
    # Използваме snapshot_download за да изтеглим цялото snapshot на репото (гарантира, че всички файлове са изтеглени)
    print(f"⬇️ Starting snapshot download of {model_id} to {save_path}...")
    repo_local_path = snapshot_download(repo_id=model_id, cache_dir=save_path, token=token, resume_download=True)
    print(f"⬇️ Snapshot download finished. snapshot_download returned: {repo_local_path}")

    # Опитваме се да намерим директорията, която съдържа моделните файлове (config.json, tokenizer.json, pytorch_model.bin / model.safetensors)
    def find_model_dir(base_path):
        # If the returned path already contains config.json, use it
        if base_path and os.path.isfile(os.path.join(base_path, "config.json")):
            return base_path
        # Walk the base path to find a folder with config.json
        for root, dirs, files in os.walk(base_path):
            if "config.json" in files:
                return root
        return None

    model_dir = find_model_dir(repo_local_path) or find_model_dir(save_path)

    if not model_dir:
        raise RuntimeError("Could not locate the downloaded model files (config.json not found).")

    print(f"⬇️ Loading tokenizer and model from: {model_dir}")

    # Зареждаме от локалния кеш (за да не предизвикаме допълнителни изтегляния от hub)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    print("✅ NLLB-200 downloaded and loaded successfully.")
except Exception as e:
    print("⚠️ An error occurred while downloading or loading the model:", str(e))
    sys.exit(1)
