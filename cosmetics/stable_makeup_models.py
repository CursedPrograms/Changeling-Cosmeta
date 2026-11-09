"""stable_makeup_models.py - the files Stable-Makeup needs (about 8 GB), into F:\\AIModels\\StableMakeup.

Stable-Makeup (Zhang et al. 2024, Apache 2.0) is a diffusion model that copies even complex, drawn makeup (face
paint, patterns) from a reference photo. It runs on Stable Diffusion 1.5 (the half-precision files only) with
CLIP ViT-L/14 as its makeup encoder, its own three weight files and SPIGA face landmarks.

    python -m cosmetics.stable_makeup_models     fetch whatever is missing (resumes, retries)
"""
import http.client
import os
import time
import urllib.error
import urllib.request

ROOT = os.environ.get("STABLE_MAKEUP_DIR", r"F:\AIModels\StableMakeup")
HF = "https://huggingface.co"
SPACE = f"{HF}/spaces/sky24h/Stable-Makeup-unofficial/resolve/main/checkpoints"
SD15 = f"{HF}/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main"
CLIP = f"{HF}/openai/clip-vit-large-patch14/resolve/main"
FILES = [   # (url, where under ROOT)
    (f"{SPACE}/stablemakeup/pytorch_model.bin", "stablemakeup/pytorch_model.bin"),
    (f"{SPACE}/stablemakeup/pytorch_model_1.bin", "stablemakeup/pytorch_model_1.bin"),
    (f"{SPACE}/stablemakeup/pytorch_model_2.bin", "stablemakeup/pytorch_model_2.bin"),
    (f"{SPACE}/spiga_300wpublic.pt", "spiga_300wpublic.pt"),
    (f"{SD15}/model_index.json", "sd15/model_index.json"),
    (f"{SD15}/scheduler/scheduler_config.json", "sd15/scheduler/scheduler_config.json"),
    (f"{SD15}/unet/config.json", "sd15/unet/config.json"),
    (f"{SD15}/unet/diffusion_pytorch_model.fp16.safetensors", "sd15/unet/diffusion_pytorch_model.fp16.safetensors"),
    (f"{SD15}/vae/config.json", "sd15/vae/config.json"),
    (f"{SD15}/vae/diffusion_pytorch_model.fp16.safetensors", "sd15/vae/diffusion_pytorch_model.fp16.safetensors"),
    (f"{SD15}/text_encoder/config.json", "sd15/text_encoder/config.json"),
    (f"{SD15}/text_encoder/model.fp16.safetensors", "sd15/text_encoder/model.fp16.safetensors"),
    (f"{SD15}/tokenizer/merges.txt", "sd15/tokenizer/merges.txt"),
    (f"{SD15}/tokenizer/vocab.json", "sd15/tokenizer/vocab.json"),
    (f"{SD15}/tokenizer/special_tokens_map.json", "sd15/tokenizer/special_tokens_map.json"),
    (f"{SD15}/tokenizer/tokenizer_config.json", "sd15/tokenizer/tokenizer_config.json"),
    (f"{SD15}/feature_extractor/preprocessor_config.json", "sd15/feature_extractor/preprocessor_config.json"),
    (f"{CLIP}/config.json", "clip-vit-large-patch14/config.json"),
    (f"{CLIP}/model.safetensors", "clip-vit-large-patch14/model.safetensors"),
    (f"{CLIP}/preprocessor_config.json", "clip-vit-large-patch14/preprocessor_config.json"),
    (f"{CLIP}/tokenizer.json", "clip-vit-large-patch14/tokenizer.json"),
    (f"{CLIP}/tokenizer_config.json", "clip-vit-large-patch14/tokenizer_config.json"),
    (f"{CLIP}/vocab.json", "clip-vit-large-patch14/vocab.json"),
    (f"{CLIP}/merges.txt", "clip-vit-large-patch14/merges.txt"),
    (f"{CLIP}/special_tokens_map.json", "clip-vit-large-patch14/special_tokens_map.json"),
]


def missing():
    return [(u, p) for u, p in FILES if not os.path.exists(os.path.join(ROOT, p))]


def _download(url, path, tries=30):
    part = path + ".part"
    for attempt in range(tries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as out:
                if have and r.status != 206:
                    out.truncate(0)
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                last = 0.0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    have += len(chunk)
                    if time.monotonic() - last > 20:
                        last = time.monotonic()
                        print(f"  {os.path.basename(path)}: {have / 2**20:.0f} / {total / 2**20:.0f} MB", flush=True)
            if total and have < total:
                raise OSError(f"stopped at {have} of {total} bytes")
            os.replace(part, path)
            return
        except (OSError, urllib.error.URLError, http.client.HTTPException) as e:
            if attempt == tries - 1:
                raise
            print(f"  {os.path.basename(path)}: {e}; trying again ({attempt + 2}/{tries})", flush=True)
            time.sleep(min(60, 3 * (attempt + 1)))


def fetch():
    for url, rel in missing():
        path = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"{rel}: downloading", flush=True)
        _download(url, path)
    print("all there:", ROOT, flush=True)


if __name__ == "__main__":
    fetch()
