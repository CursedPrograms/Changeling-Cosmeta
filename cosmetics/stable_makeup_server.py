"""stable_makeup_server.py - Stable-Makeup, loaded once, answering requests (runs in StableMakeup/smenv).

    StableMakeup/smenv/Scripts/python.exe cosmetics/stable_makeup_server.py

Cosmeta's window (cosmetics/stable_makeup.py) starts it and talks to it a line at a time:
    the window writes  {"id": "you.png", "makeup": "ref.png", "out": "result.png", "guidance": 1.6, "steps": 30}
    this writes        {"ok": true, "box": [x0, y0, x1, y1]}   (where in "id" the 512 x 512 result belongs)
                  or   {"ok": false, "error": "..."}
It prints {"ready": true} once the models are loaded (about a minute the first time).

Stable-Makeup (Zhang et al. 2024, github.com/Xiaojiu-z/Stable-Makeup, Apache 2.0) is Stable Diffusion 1.5 with two
ControlNets (your identity, your face's shape from SPIGA's landmarks) and a makeup encoder (CLIP ViT-L/14 on the
reference). Its code is the Space sky24h/Stable-Makeup-unofficial (cloned into StableMakeup/, with its own copy of
diffusers); the models are read from F:\\AIModels\\StableMakeup (STABLE_MAKEUP_DIR), nothing is downloaded.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(HERE, "StableMakeup")
ROOT = os.environ.get("STABLE_MAKEUP_DIR", r"F:\AIModels\StableMakeup")


def say(obj, out):
    out.write(json.dumps(obj) + "\n")
    out.flush()


def main():
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")   # answers only; everything else printed goes to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    os.chdir(CODE)                                      # its spiga_draw reads checkpoints/ from here (a link to ROOT)
    sys.path.insert(0, CODE)

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from spiga_draw import spiga_process, spiga_segmentation
    from pipeline_sd15 import StableDiffusionControlNetPipeline
    from diffusers import DDIMScheduler, ControlNetModel
    from diffusers import UNet2DConditionModel
    from detail_encoder.encoder_plus import detail_encoder
    from batch_face import RetinaFace

    device = "cuda"
    sd15 = os.path.join(ROOT, "sd15")
    clip = os.path.join(ROOT, "clip-vit-large-patch14")
    weights = os.path.join(ROOT, "stablemakeup")
    unet = UNet2DConditionModel.from_pretrained(sd15, subfolder="unet", variant="fp16", torch_dtype=torch.float16)
    id_encoder = ControlNetModel.from_unet(unet)
    pose_encoder = ControlNetModel.from_unet(unet)
    makeup_encoder = detail_encoder(unet, clip, device=device, dtype=torch.float16)
    for model, name in ((makeup_encoder, "pytorch_model.bin"), (id_encoder, "pytorch_model_1.bin"),
                        (pose_encoder, "pytorch_model_2.bin")):
        model.load_state_dict(torch.load(os.path.join(weights, name), map_location="cpu"), strict=False)
        model.half().to(device=device)               # halved on the CPU first: 6 GB cards never hold the fp32 copy
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd15, safety_checker=None, feature_extractor=None, requires_safety_checker=False, unet=unet,
        controlnet=[id_encoder, pose_encoder], variant="fp16", torch_dtype=torch.float16).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    detector = RetinaFace(gpu_id=0)
    print(f"vram: {torch.cuda.memory_allocated() / 2**20:.0f} MB of models", flush=True)

    def square_face(bgr, padding=1.5):
        faces = detector(bgr, cv=True)
        if len(faces) == 0:
            raise ValueError("no face found")
        x1, y1, x2, y2 = faces[0][0]
        length = ((x2 - x1) + (y2 - y1)) / 2
        x1, x2 = x1 - length * (padding - 1), x2 + length * (padding - 1)
        y1, y2 = y1 - length * (padding - 1), y2 + length * (padding - 1)
        y1 -= length * (padding - 1) * 0.2
        y2 -= length * (padding - 1) * 0.2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) / 2
        h, w = bgr.shape[:2]
        box = [max(int(round(cx - half)), 0), max(int(round(cy - half)), 0),
               min(int(round(cx + half)), w), min(int(round(cy + half)), h)]
        crop = bgr[box[1]:box[3], box[0]:box[2]]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), box

    def pose(pil, size):
        spigas = spiga_process(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        if spigas is False:
            return Image.new("RGB", pil.size, (0, 0, 0))
        return spiga_segmentation(spigas, size=size)

    say({"ready": True}, out)
    for line in sys.stdin:
        try:
            job = json.loads(line)
            you, box = square_face(cv2.imdecode(np.fromfile(job["id"], np.uint8), cv2.IMREAD_COLOR))
            ref, _ = square_face(cv2.imdecode(np.fromfile(job["makeup"], np.uint8), cv2.IMREAD_COLOR))
            size = 512
            you, ref = you.resize((size, size)), ref.resize((size, size))
            torch.manual_seed(int(job.get("seed", 1024)))
            result = makeup_encoder.generate(id_image=[you, pose(you, size)], makeup_image=ref, pipe=pipe,
                                             guidance_scale=float(job.get("guidance", 1.6)))
            result.save(job["out"])
            print(f"vram: peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB allocated, "
                  f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MB reserved", flush=True)
            say({"ok": True, "box": box}, out)
        except Exception as e:  # noqa: BLE001 - told to the window
            say({"ok": False, "error": f"{type(e).__name__}: {e}"}, out)
        finally:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
