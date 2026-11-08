"""aethercanvas.py - the edits Cosmetics can't paint: hair length and style, a full beard, a clean shave.

Those need a generative model, so Cosmetics sends the picture to AetherCanvas AI (Qwen-Image Edit, running on
this PC: https://github.com/CursedPrograms/AetherCanvas-AI) with a ready instruction and gets the edited picture
back. When AetherCanvas isn't running, these buttons say so and stay off.

    AETHERCANVAS_URL   where it runs (default http://127.0.0.1:7870)
"""
import base64
import json
import os
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np

URL = os.environ.get("AETHERCANVAS_URL", "http://127.0.0.1:7870")

EDITS = {
    "Hair length": [
        ("Long", "Make the hair long, flowing past the shoulders, same colour and texture"),
        ("Shoulder", "Make the hair shoulder-length, same colour and texture"),
        ("Bob", "Give a chin-length bob haircut, same hair colour"),
        ("Short", "Make the hair short, a short cropped haircut, same hair colour"),
        ("Pixie", "Give a pixie cut, same hair colour"),
        ("Shaved", "Make the head shaved bald"),
    ],
    "Hair style": [
        ("Bangs", "Add straight bangs across the forehead, same hair colour"),
        ("Curly", "Make the hair curly, same length and colour"),
        ("Straight", "Make the hair sleek and straight, same length and colour"),
        ("Ponytail", "Tie the hair back into a high ponytail"),
        ("Bun", "Put the hair up in a neat bun"),
        ("Braids", "Style the hair in long braids"),
    ],
    "Beard": [
        ("Full beard", "Give the man a full, well-groomed beard, same hair colour"),
        ("Goatee", "Give the man a neat goatee"),
        ("Moustache", "Give the man a moustache"),
        ("Clean shaven", "Make the face clean shaven, no beard or stubble"),
    ],
}


def available(timeout=1.5):
    """Is AetherCanvas answering?"""
    try:
        with urllib.request.urlopen(URL + "/", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def edit(image_rgb, instruction, timeout=1800):
    """The picture (RGB uint8) edited by `instruction`; raises RuntimeError with AetherCanvas's reason."""
    ok, png = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("couldn't encode the picture")
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("prompt", instruction), ("resolution", "1024")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="photo.png"\r\n'
                 f"Content-Type: image/png\r\n\r\n".encode() + png.tobytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(URL + "/api/edit", data=b"".join(parts), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            reason = json.loads(e.read()).get("error", str(e))
        except Exception:
            reason = str(e)
        raise RuntimeError(f"AetherCanvas: {reason}")
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"AetherCanvas isn't answering at {URL} ({e})")
    images = body.get("images") or []
    if not images or not images[0].get("data"):
        raise RuntimeError("AetherCanvas sent no picture back")
    data = base64.b64decode(images[0]["data"].split(",")[-1])
    out = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if out is None:
        raise RuntimeError("couldn't read AetherCanvas's picture")
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
