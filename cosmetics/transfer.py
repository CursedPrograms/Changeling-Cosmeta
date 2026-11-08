"""transfer.py - copy the makeup from any photo: a reference face's makeup, onto this face.

The model is DMT (Disentangled Makeup Transfer, a GAN trained on before/after makeup photos), converted to ONNX
(models/dmt.onnx, 48 MB; run.bat fetches it). It works on 256 x 256 faces, too small for a photo, so it isn't
pasted in: Cosmetics takes only the colour change it made (in LAB, smoothed) and adds that to the full-size photo
on the lips, the eyes' surroundings, the cheeks and the skin. The photo's own detail, hair and background stay.
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(HERE, "models", "dmt.onnx")
SIZE = 256


def _crop_box(points, shape, grow=1.7):
    """A square around the face (Face Mesh points), grown to take in the whole face: (x0, y0, side)."""
    x0, y0 = points.min(0)
    x1, y1 = points.max(0)
    side = int(max(x1 - x0, y1 - y0) * grow)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(cx - side / 2), int(cy - side / 2), side


def _crop(image, box):
    x0, y0, side = box
    h, w = image.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return image[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]


def _lab(img):
    return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


class MakeupTransfer:
    def __init__(self):
        if not os.path.exists(MODEL):
            raise FileNotFoundError(f"no makeup transfer model at {MODEL} (run.bat fetches it)")
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(MODEL, providers=providers)
        self.inputs = [i.name for i in self.session.get_inputs()]

    def delta(self, face, reference_face):
        """The makeup's colour change for `face` (a read Face), taken from `reference_face` (another read Face):
        a full-size LAB difference, already limited to where makeup goes."""
        if face.points is None or reference_face.points is None:
            raise ValueError("no face found")
        box = _crop_box(face.points, face.image.shape)
        src = cv2.resize(_crop(face.image, box), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        ref = cv2.resize(_crop(reference_face.image, _crop_box(reference_face.points, reference_face.image.shape)),
                         (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = (src.astype(np.float32) / 255 - 0.5) * 2
        y = (ref.astype(np.float32) / 255 - 0.5) * 2
        out = self.session.run(None, {self.inputs[0]: x[None], self.inputs[1]: y[None]})[0][0]
        out = np.clip((out + 1) / 2 * 255, 0, 255).astype(np.uint8)
        d = _lab(out) - _lab(src)                                   # what the makeup changed
        d = cv2.GaussianBlur(d, (0, 0), 1.6)                        # its colour, not the GAN's grain
        # back to the photo's size and place
        x0, y0, side = box
        d = cv2.resize(d, (side, side), interpolation=cv2.INTER_CUBIC)
        h, w = face.image.shape[:2]
        full = np.zeros((h, w, 3), np.float32)
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(w, x0 + side), min(h, y0 + side)
        full[fy0:fy1, fx0:fx1] = d[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        keep = self.where(face)
        # Lids and brows: makeup only darkens them (the model's paler is its training bias, not makeup);
        # lips may go either way (a nude lipstick is lighter).
        lips = face.mask("u_lip", "l_lip", feather=max(1.0, face.size * 0.006))
        darken = np.minimum(full[..., 0], 0)
        full[..., 0] = lips * full[..., 0] + (1 - lips) * darken
        return full * keep, out

    @staticmethod
    def where(face):
        """How much of the model's change to keep, per place and per LAB channel (H x W x 3): all of it on the
        lips, eyes and brows; on the cheeks and skin only some of the colour and none of the lightness. (The
        model learned from pale "after" photos, so its foundation also lightens the skin: a skin-tone change,
        not makeup.)"""
        from .looks import blush_mask, eyeshadow_mask
        f = face.size
        full = face.mask("u_lip", "l_lip", feather=max(1.0, f * 0.006))
        eyes = eyeshadow_mask(face)
        if eyes is not None:
            full = np.maximum(full, eyes)
        full = np.maximum(full, face.mask("l_brow", "r_brow", feather=2) * 0.7)
        colour = face.mask("skin", "nose", feather=max(1.0, f * 0.01)) * 0.3
        cheeks = blush_mask(face)
        if cheeks is not None:
            colour = np.maximum(colour, cheeks * 0.8)
        light = full                                        # lightness: lips, eyes, brows only
        tint = np.maximum(full, colour)                     # colour: those, and some on the skin and cheeks
        return np.stack([light, tint, tint], axis=-1)
