"""parse.py - which pixel is what: hair, skin, lips, brows, eyes (a SegFormer face parser) and where the
face's 468 points are (MediaPipe Face Mesh).

The parser runs once per photo on a square crop around the face, grown to take in the hair; its 19
labels come back as full-size masks. The looks (looks.py) then only blend colours, so every slider
answers at once.
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(HERE, "models", "face_parsing_int8.onnx")
MODEL_URL = "https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/model_quantized.onnx"
SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
LABELS = ["background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear", "r_ear",
          "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth"]
ID = {name: i for i, name in enumerate(LABELS)}


class Face:
    """One photo, read: .labels (H x W label map), .points (468 x 2 or None), and masks by name."""

    def __init__(self, image, labels, points):
        self.image, self.labels, self.points = image, labels, points

    def mask(self, *names, feather=0):
        m = np.isin(self.labels, [ID[n] for n in names]).astype(np.float32)
        if feather:
            m = cv2.GaussianBlur(m, (0, 0), feather)
        return m

    def resized(self, max_side):
        """The same face at most max_side pixels across (for quick previews), or itself if already smaller."""
        h, w = self.image.shape[:2]
        k = max_side / max(h, w)
        if k >= 1:
            return self
        size = (max(1, int(w * k)), max(1, int(h * k)))
        image = cv2.resize(self.image, size, interpolation=cv2.INTER_AREA)
        labels = cv2.resize(self.labels, size, interpolation=cv2.INTER_NEAREST)
        points = None if self.points is None else self.points * k
        return Face(image, labels, points)

    @property
    def size(self):
        """The face's width in pixels (for sizing strokes and feathers)."""
        if self.points is None:
            return self.image.shape[1] / 3
        return float(np.ptp(self.points[:, 0]))


class Parser:
    def __init__(self):
        if not os.path.exists(MODEL):
            raise FileNotFoundError(f"No face parser at {MODEL}: run the launcher (it downloads it), or get {MODEL_URL}")
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(MODEL, providers=providers)
        self.input = self.session.get_inputs()[0].name
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    def read(self, image):
        """image: RGB uint8. Returns a Face (raises ValueError when there's no face)."""
        h, w = image.shape[:2]
        found = self.mesh.process(image)
        if not found.multi_face_landmarks:
            raise ValueError("No face found in the photo.")
        pts = np.array([(p.x * w, p.y * h) for p in found.multi_face_landmarks[0].landmark], np.float32)
        # a square around the face, grown to take in the hair (above and to the sides) and the neck
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        # (long hair falls well past the chin, so the square reaches about two faces below it)
        fh = max(x1 - x0, y1 - y0)
        cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2 + fh * 0.45, fh * 3.6
        side = int(round(side))
        left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
        box = np.array([left, top, left + side, top + side])          # exactly square
        pad = max(0, -box[0], -box[1], box[2] - w, box[3] - h)
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else image
        bx0, by0, bx1, by1 = box + pad
        crop = padded[by0:by1, bx0:bx1]
        x = (cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - MEAN) / STD
        logits = self.session.run(None, {self.input: x.transpose(2, 0, 1)[None]})[0][0]   # 19 x 128 x 128
        side_px = by1 - by0
        up = cv2.resize(logits.transpose(1, 2, 0), (side_px, side_px), interpolation=cv2.INTER_LINEAR)
        crop_labels = up.argmax(-1).astype(np.uint8)
        labels = np.zeros(padded.shape[:2], np.uint8)
        labels[by0:by1, bx0:bx1] = crop_labels
        if pad:
            labels = labels[pad:pad + h, pad:pad + w]
        return Face(image, labels, pts)
