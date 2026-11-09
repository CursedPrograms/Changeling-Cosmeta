"""live.py - Changeling Cosmeta on video, frame by frame (Changeling Cosmeta Realtime uses it).

    live = Live()
    out = live.apply(frame_rgb, look, adjust)      # the frame with the look painted on (and the editor's changes)

The same parser labels and the same painter as for photos, made quick enough for a webcam: BiSeNet on the
graphics card (FastParser), Face Mesh following the face from frame to frame, and the painting done only in the
box around the face and the hair (most of a webcam picture is the room behind you), at most WORK pixels
across (the change scaled back up), then put back.
"""
import cv2
import numpy as np

from .looks import Painter
from .parse import ID, Face, FastParser

# what the looks paint on: the box around these is the only part of the frame worked on
PAINTED = [ID[n] for n in ("skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "mouth", "u_lip", "l_lip", "hair")]


class Live:
    def __init__(self):
        self.parser = FastParser(video=True)
        self.found = False

    def apply(self, frame, look, adjust=None):
        """frame: RGB uint8. Returns the painted frame (RGB); with no face, the frame as it was (edited)."""
        return self.paint(frame, self.read(frame), look, adjust)

    def read(self, frame):
        """The face in the frame (a parse.Face), or None. (read and paint can run on two threads, one frame apart.)"""
        try:
            face = self.parser.read(frame)
        except ValueError:
            face = None
        self.found = face is not None
        return face

    def paint(self, frame, face, look, adjust=None):
        out = frame
        if face is not None and _anything_on(look):
            ys, xs = np.nonzero(np.isin(face.labels, PAINTED))
            if len(ys):
                h, w = frame.shape[:2]
                pad = int(face.size * 0.08)
                y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
                x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
                crop = Face(frame[y0:y1, x0:x1], face.labels[y0:y1, x0:x1], face.points - (x0, y0))
                out = frame.copy()
                out[y0:y1, x0:x1] = self._paint(crop, look)
        if adjust is not None:
            out = adjust.apply(out)
        return out


    def _paint(self, crop, look):
        """The look on the crop. A big crop is painted at most WORK pixels across and only the change is scaled
        back up (colour changes are smooth, so it loses nothing you'd see): about three times quicker."""
        h, w = crop.image.shape[:2]
        if max(h, w) <= self.WORK:
            return Painter(crop).apply(look)
        small = crop.resized(self.WORK)
        change = Painter(small).apply(look).astype(np.int16) - small.image.astype(np.int16)
        change = cv2.resize(change.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(crop.image.astype(np.float32) + change, 0, 255).astype(np.uint8)

    WORK = 420


def _anything_on(look):
    return any(getattr(look, name).on for name in ("hair", "lips", "eyeshadow", "eyeliner", "blush", "brows", "skin",
                                                    "beard", "stubble", "shave"))
