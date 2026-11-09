"""parse.py - which pixel is what: hair, skin, lips, brows, eyes (a SegFormer face parser) and where the
face's 468 points are (MediaPipe Face Mesh).

The parser runs once per photo on a square crop around the face, grown to take in the hair; its 19
labels come back as full-size masks. The looks (looks.py) then only blend colours, so every slider
answers at once.

Two parsers, the same labels: Parser (SegFormer, int8 ONNX, about 1.2 s on the CPU: photos) and FastParser
(BiSeNet in PyTorch on the graphics card, a few milliseconds: live video, Changeling Cosmetics Realtime). BiSeNet's
weights are face_parsing_79999_iter.pth (zllrunning/face-parsing.PyTorch, MIT), the copy HairFastGAN ships: it's
taken from F:\\AIModels\\face or HairFastGAN's models if it isn't in models/ already.
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
FAST_MODEL = os.path.join(HERE, "models", "face_parsing_bisenet.pth")
FAST_URL = ("https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models/BiSeNet/"
            "face_parsing_79999_iter.pth")
FAST_COPIES = [r"F:\AIModels\face\face_parsing_bisenet.pth",
               r"F:\AIModels\HairFastGAN\pretrained_models\BiSeNet\face_parsing_79999_iter.pth",
               os.path.join(os.path.dirname(HERE), "Changeling-Tonsor", "HairFastGAN", "pretrained_models", "BiSeNet",
                            "face_parsing_79999_iter.pth")]
# BiSeNet's own label order, as names (they're the same 19 CelebAMask-HQ labels in another order)
BISENET_LABELS = ["background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "l_ear", "r_ear", "ear_r",
                  "nose", "mouth", "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat"]


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
    def __init__(self, video=False):
        self._open()
        import mediapipe as mp
        # video: Face Mesh follows the face from frame to frame (quicker and steadier) instead of finding it afresh
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=not video, max_num_faces=1, refine_landmarks=True)

    def _open(self):
        if not os.path.exists(MODEL):
            raise FileNotFoundError(f"No face parser at {MODEL}: run the launcher (it downloads it), or get {MODEL_URL}")
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(MODEL, providers=providers)
        self.input = self.session.get_inputs()[0].name

    def _labels(self, crop):
        """The label map (LABELS ids) of a square RGB crop, at the crop's size."""
        x = (cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - MEAN) / STD
        logits = self.session.run(None, {self.input: x.transpose(2, 0, 1)[None]})[0][0]   # 19 x 128 x 128
        side = crop.shape[0]
        up = cv2.resize(logits.transpose(1, 2, 0), (side, side), interpolation=cv2.INTER_LINEAR)
        return up.argmax(-1).astype(np.uint8)

    def read(self, image):
        """image: RGB uint8. Returns a Face (raises ValueError when there's no face)."""
        h, w = image.shape[:2]
        found = self.mesh.process(image)
        if not found.multi_face_landmarks:
            raise ValueError("No face found in the photo.")
        pts = np.array([(p.x * w, p.y * h) for p in found.multi_face_landmarks[0].landmark], np.float32)
        # a square around the face, grown to take in the hair (above and to the sides) and the neck
        # (long hair falls well past the chin, so the square reaches about two faces below it)
        labels = self._parse_square(image, pts, 3.6, 0.45)
        return Face(image, self._refine(image, pts, labels), pts)

    def _refine(self, image, pts, labels):
        return labels

    def _parse_square(self, image, pts, scale, down):
        """Labels for the whole image (background outside) from a square crop scale faces wide, its centre `down`
        faces below the face's."""
        h, w = image.shape[:2]
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        fh = max(x1 - x0, y1 - y0)
        cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2 + fh * down, fh * scale
        side = int(round(side))
        left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
        box = np.array([left, top, left + side, top + side])          # exactly square
        pad = max(0, -box[0], -box[1], box[2] - w, box[3] - h)
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else image
        bx0, by0, bx1, by1 = box + pad
        crop = padded[by0:by1, bx0:bx1]
        labels = np.zeros(padded.shape[:2], np.uint8)
        labels[by0:by1, bx0:bx1] = self._labels(crop)
        if pad:
            labels = labels[pad:pad + h, pad:pad + w]
        return labels


class FastParser(Parser):
    """The same, with BiSeNet on the graphics card: fast enough for every frame of a video.

    BiSeNet learned on CelebA-HQ's close crops, where the face fills the picture: in the wide square that takes in
    long hair, the face is small and its parts land in the wrong places. So it reads twice: the wide square for the
    hair, neck and clothes, and a close one (as CelebA-HQ's) for the face itself, which wins inside it."""

    FACE_PARTS = ["skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "mouth", "u_lip", "l_lip", "l_ear", "r_ear"]

    def _refine(self, image, pts, labels):
        close = self._parse_square(image, pts, 1.7, 0.05)
        parts = np.isin(close, [ID[n] for n in self.FACE_PARTS])
        out = labels.copy()
        inner = np.isin(labels, [ID[n] for n in self.FACE_PARTS])
        out[inner & ~parts & (close != 0)] = close[inner & ~parts & (close != 0)]   # e.g. hair over the forehead
        out[parts] = close[parts]
        return out

    def _open(self):
        import shutil
        import torch
        if not os.path.exists(FAST_MODEL):
            found = next((p for p in FAST_COPIES if os.path.exists(p)), None)
            os.makedirs(os.path.dirname(FAST_MODEL), exist_ok=True)
            if found is not None:
                shutil.copy(found, FAST_MODEL)
            else:                                       # nowhere on this PC: once, about 50 MB
                import urllib.request
                print("Downloading the BiSeNet face parser (about 50 MB) ...", flush=True)
                urllib.request.urlretrieve(FAST_URL, FAST_MODEL + ".part")
                os.replace(FAST_MODEL + ".part", FAST_MODEL)
        from .bisenet import BiSeNet
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        net = BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(FAST_MODEL, map_location="cpu", weights_only=True))
        self.net = net.to(self.device).eval()
        if self.device == "cuda":
            self.net = self.net.half()
        self.remap = torch.tensor([ID[n] for n in BISENET_LABELS], dtype=torch.uint8, device=self.device)
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    def _labels(self, crop):
        import torch
        import torch.nn.functional as F
        side = crop.shape[0]
        small = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            x = torch.from_numpy(small).to(self.device).permute(2, 0, 1)[None].float() / 255
            x = (x - self.mean) / self.std
            logits = self.net(x.half() if self.device == "cuda" else x)[0]          # 1 x 19 x 512 x 512
            logits = F.interpolate(logits.float(), size=(side, side), mode="bilinear", align_corners=False)
            labels = self.remap[logits[0].argmax(0)]
        return labels.cpu().numpy()
