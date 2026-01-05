from huggingface_hub import snapshot_download

# Прочитане на токена от файл
with open("/run/media/msi/C086D41086D408B41/AuroraAI/secrets/hf_aurora.txt", "r") as f:
    token = f.read().strip()

# Изтегляне на модела в зададената директория
snapshot_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    local_dir="/run/media/msi/C086D41086D408B41/AuroraAI/models/stable_diffusion",
    local_dir_use_symlinks=False,
    token=token
)

