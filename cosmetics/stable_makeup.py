"""stable_makeup.py - Stable-Makeup for Cosmeta's Model switch: the diffusion model redraws your face with the
reference's makeup.

It runs in its own environment (StableMakeup/smenv: an older transformers, SPIGA, RetinaFace) as a child
process (cosmetics/stable_makeup_server.py), loaded once and then answering a photo at a time: about a minute to
load, then a few seconds a photo on the graphics card. It draws a 512 x 512 square around the face; Cosmeta puts that
back into your full-size photo on the face only (skin, brows, eyes, nose, lips: the parser's labels), so hair,
background and the rest stay exactly as they were.
"""
import json
import os
import subprocess
import tempfile
import threading

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(HERE, "StableMakeup")


def python():
    win = os.path.join(CODE, "smenv", "Scripts", "python.exe")
    return win if os.name == "nt" else os.path.join(CODE, "smenv", "bin", "python")


def installed():
    return os.path.exists(os.path.join(CODE, "pipeline_sd15.py")) and os.path.exists(python())


def _lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


class StableMakeup:
    def __init__(self):
        if not installed():
            raise RuntimeError("Stable-Makeup isn't set up: run setup_stable_makeup.bat once.")
        self.proc = subprocess.Popen([python(), "-u", os.path.join(HERE, "cosmetics", "stable_makeup_server.py")],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.err = []
        threading.Thread(target=self._drain, daemon=True).start()
        first = self.proc.stdout.readline()
        if not first or not json.loads(first).get("ready"):
            raise RuntimeError("Stable-Makeup didn't start: " + (self.err[-1] if self.err else "no output"))
        self.lock = threading.Lock()

    def _drain(self):
        for line in self.proc.stderr:
            self.err = (self.err + [line.strip()])[-20:]

    def delta(self, face, reference, paint=False, guidance=None):
        """(LAB change the size of the photo, the result as RGB): the reference's makeup on `face` (a read Face)."""
        ref = reference.image if hasattr(reference, "image") else reference
        tmp = tempfile.mkdtemp(prefix="cosmeta_sm_")
        you_p, ref_p, out_p = (os.path.join(tmp, n) for n in ("you.png", "ref.png", "out.png"))
        cv2.imencode(".png", cv2.cvtColor(face.image, cv2.COLOR_RGB2BGR))[1].tofile(you_p)
        cv2.imencode(".png", cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))[1].tofile(ref_p)
        job = {"id": you_p, "makeup": ref_p, "out": out_p, "guidance": guidance or (2.0 if paint else 1.6)}
        with self.lock:
            self.proc.stdin.write(json.dumps(job) + "\n")
            self.proc.stdin.flush()
            answer = self.proc.stdout.readline()
        if not answer:
            raise RuntimeError("Stable-Makeup stopped: " + (self.err[-1] if self.err else "no output"))
        answer = json.loads(answer)
        if not answer.get("ok"):
            raise RuntimeError("Stable-Makeup: " + answer.get("error", "failed"))
        x0, y0, x1, y1 = answer["box"]
        result = cv2.cvtColor(cv2.imdecode(np.fromfile(out_p, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        result = cv2.resize(result, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4)
        for p in (you_p, ref_p, out_p):
            os.remove(p)
        os.rmdir(tmp)

        T = _lab(face.image)
        R = T.copy()
        R[y0:y1, x0:x1] = _lab(result)
        if face.labels is not None and face.labels.any():
            where = face.mask("skin", "nose", "l_brow", "r_brow", "l_eye", "r_eye", "u_lip", "l_lip", "mouth")
        else:
            where = np.zeros(face.image.shape[:2], np.float32)
            where[y0:y1, x0:x1] = 1
        where = cv2.GaussianBlur(where, (0, 0), max(1.0, face.size * 0.01))
        d = (R - T) * where[..., None]
        out = cv2.cvtColor(T + d, cv2.COLOR_LAB2RGB)
        return d, np.clip(out * 255, 0, 255).astype(np.uint8)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()
