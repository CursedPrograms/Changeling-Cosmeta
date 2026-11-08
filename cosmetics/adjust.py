"""adjust.py - the Changeling basic editor: brightness, contrast, saturation, warmth, vignette, sharpen.

The same file goes into every Changeling (the realtime ones too), so it's quick: brightness and contrast
are one lookup table, the vignette's mask is made once per frame size, and nothing runs for a slider at 0.
Works on RGB or BGR uint8 frames (say which with `bgr`).

    Adjust(brightness=10, contrast=15, vignette=40).apply(frame)

Every value is -100 .. 100 (vignette and sharpen 0 .. 100); 0 leaves the picture alone.
"""
from dataclasses import dataclass, fields

import cv2
import numpy as np


@dataclass
class Adjust:
    brightness: int = 0
    contrast: int = 0
    saturation: int = 0
    warmth: int = 0
    vignette: int = 0
    sharpen: int = 0

    RANGES = {"brightness": (-100, 100), "contrast": (-100, 100), "saturation": (-100, 100),
              "warmth": (-100, 100), "vignette": (0, 100), "sharpen": (0, 100)}

    def is_neutral(self):
        return all(getattr(self, f.name) == 0 for f in fields(self))

    def apply(self, img, bgr=False):
        if self.is_neutral():
            return img
        out = img
        if self.brightness or self.contrast:
            out = cv2.LUT(out, _levels(self.brightness, self.contrast))
        if self.saturation:
            grey = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY if bgr else cv2.COLOR_RGB2GRAY)
            k = 1 + self.saturation / 100
            out = cv2.addWeighted(out, k, cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), 1 - k, 0)
        if self.warmth:
            shift = self.warmth * 0.35
            r, b = (2, 0) if bgr else (0, 2)
            out = out.astype(np.int16)
            out[..., r] += int(shift)
            out[..., b] -= int(shift)
            out = np.clip(out, 0, 255).astype(np.uint8)
        if self.sharpen:
            soft = cv2.GaussianBlur(out, (0, 0), 1.6)
            amount = self.sharpen / 100 * 1.2
            out = cv2.addWeighted(out, 1 + amount, soft, -amount, 0)
        if self.vignette:
            mask = _vignette(out.shape[0], out.shape[1], self.vignette / 100)
            out = (out.astype(np.float32) * mask[..., None]).astype(np.uint8)
        return out


_level_cache = {}
_vignette_cache = {}


def _levels(brightness, contrast):
    key = (brightness, contrast)
    if key not in _level_cache:
        x = np.arange(256, dtype=np.float32)
        c = (100 + contrast) / 100 if contrast > 0 else 1 + contrast / 150
        x = (x - 128) * c ** 1.5 + 128 + brightness * 1.1
        if len(_level_cache) > 64:
            _level_cache.clear()
        _level_cache[key] = np.clip(x, 0, 255).astype(np.uint8)
    return _level_cache[key]


def _vignette(h, w, strength):
    key = (h, w, round(strength, 2))
    if key not in _vignette_cache:
        y, x = np.ogrid[:h, :w]
        d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        mask = 1 - strength * np.clip((d - 0.35) / 0.65, 0, 1) ** 1.6
        if len(_vignette_cache) > 16:
            _vignette_cache.clear()
        _vignette_cache[key] = mask.astype(np.float32)
    return _vignette_cache[key]
