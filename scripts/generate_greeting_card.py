#!/usr/bin/env python3
"""
Generate a rich New Year 2026 greeting card image using the local Stable Diffusion model
located at models/stable_diffusion.

This script attempts to load the model locally (no network). If dependencies are missing,
it prints concrete pip commands to install them.

Output: outputs/greeting_2026.png
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "stable_diffusion"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "greeting_2026.png"

prompt = (
    "A luxurious, richly decorated New Year 2026 greeting card: elegant gold foil typography 'Happy New Year 2026', "
    "festive fireworks in the night sky, warm bokeh lights, ornate frame, winter holiday motifs (snowflakes, pine branches), "
    "subtle cinematic lighting, high detail, photorealistic mixed with painterly textures, 4k, vibrant colors, editorial composition"
)
negative_prompt = "low quality, watermark, text artifacts, deformed, ugly, cropped"

print(f"Model dir: {MODEL_DIR}")
if not MODEL_DIR.exists():
    print("Error: stable_diffusion model directory not found at:", MODEL_DIR)
    print("Please download the Stable Diffusion model into that path first.")
    sys.exit(2)

# Try to import the diffusers pipeline
try:
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image
except Exception as e:
    print("Missing dependencies or import error:", e)
    print("To install required packages, run (recommended):")
    print("  pip install --upgrade pip")
    print("  pip install diffusers['torch'] transformers accelerate safetensors pillow")
    print("Or add versions compatible with your CUDA / CPU setup. See https://github.com/huggingface/diffusers for details.")
    sys.exit(1)

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load pipeline from local dir
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
except Exception as e:
    print('Failed to load pipeline from local files:', e)
    print('If the model is stored in a Hugging Face snapshot layout (folders like unet/, text_encoder/, vae/), this should work.')
    sys.exit(1)

# Move to device
pipe = pipe.to(device)

# Use a reasonable number of inference steps and guidance
generator = None
num_inference_steps = 30
guidance_scale = 7.5
height = 1024
width = 1536

print('Generating image... this may take a while')
with torch.autocast(device.type) if device.type == 'cuda' else nullcontext():
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

# Save
image.save(OUT_PATH)
print('Saved greeting card to', OUT_PATH)

# Try to open the image (best-effort)
try:
    from PIL import Image
    im = Image.open(OUT_PATH)
    im.show()
except Exception:
    pass

# small nullcontext fallback
class _Null:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

# ensure nullcontext symbol exists used above
nullcontext = getattr(__import__('contextlib'), 'nullcontext', lambda: _Null())

