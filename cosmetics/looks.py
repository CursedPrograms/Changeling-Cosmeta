"""looks.py - the makeup and the hair colour, painted onto a read Face (parse.py).

Every look is a colour blended into one region in LAB, so the light and texture underneath stay (a
lipstick keeps the lips' lines, a hair colour keeps every strand): the region's own brightness moves
part of the way towards the colour's, its tint all the way, its fine variation stays.

    hair       the hair mask (the parser)               colour, strength
    lips       upper and lower lip (the parser)          colour, strength, gloss
    eyeshadow  lid up to the brow (Face Mesh points)     colour, strength
    eyeliner   a line along the upper lid, with a wing   colour, strength (its thickness)
    blush      a soft patch on each cheekbone            colour, strength
    brows      the brows (the parser)                    colour, strength
    skin       smoother skin (an edge-keeping blur)      strength
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import facial_hair

# Face Mesh points: each upper lid from the outer corner in, each brow's lower edge from the inner end out
LIDS = ([33, 246, 161, 160, 159, 158, 157, 173, 133], [263, 466, 388, 387, 386, 385, 384, 398, 362])
BROWS_LOW = ([55, 65, 52, 53, 46], [285, 295, 282, 283, 276])
CHEEKS = (50, 280)          # on each cheekbone
TEMPLES = (127, 356)        # blush leans this way

HAIR_COLOURS = {"Platinum": "#e8dcc0", "Honey blonde": "#c8a060", "Copper": "#b0582a", "Auburn": "#7a2e1c",
                "Chocolate": "#4a2e20", "Jet black": "#141214", "Silver": "#b8b8bc", "Pastel pink": "#e8a0b8",
                "Lavender": "#a890d0", "Electric blue": "#2a50d0", "Emerald": "#1f8a5a"}
LIP_COLOURS = {"Nude": "#c08070", "Rose": "#c0506a", "Classic red": "#b01020", "Berry": "#7a1838",
               "Coral": "#e06048", "Plum": "#5a2040", "Cursed pink": "#e0408a"}
SHADOW_COLOURS = {"Bronze": "#8a5a30", "Smoky": "#3a3238", "Rose gold": "#c08070", "Plum": "#6a3050",
                  "Teal": "#2a7070", "Gold": "#c09a40"}
LINER_COLOURS = {"Black": "#101010", "Brown": "#3a2418", "Navy": "#18204a"}
BLUSH_COLOURS = {"Peach": "#f0a080", "Pink": "#f08aa0", "Berry": "#c05070", "Bronze": "#b07050"}
BROW_COLOURS = {"Soft black": "#2a2422", "Dark brown": "#3e2a20", "Light brown": "#7a5a40", "Blonde": "#b09060"}


def rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


@dataclass
class Part:
    on: bool = False
    colour: str = "#000000"
    strength: float = 0.6
    gloss: float = 0.0          # lips only


@dataclass
class Look:
    hair: Part = field(default_factory=lambda: Part(False, HAIR_COLOURS["Copper"], 0.75))
    lips: Part = field(default_factory=lambda: Part(False, LIP_COLOURS["Rose"], 0.6, 0.3))
    eyeshadow: Part = field(default_factory=lambda: Part(False, SHADOW_COLOURS["Bronze"], 0.5))
    eyeliner: Part = field(default_factory=lambda: Part(False, LINER_COLOURS["Black"], 0.5))
    blush: Part = field(default_factory=lambda: Part(False, BLUSH_COLOURS["Peach"], 0.4))
    brows: Part = field(default_factory=lambda: Part(False, BROW_COLOURS["Dark brown"], 0.5))
    skin: Part = field(default_factory=lambda: Part(False, "#000000", 0.5))
    beard: Part = field(default_factory=lambda: Part(False, "#3a2618", 0.7))      # colour the beard
    stubble: Part = field(default_factory=lambda: Part(False, "#1a1410", 0.5))    # paint stubble in
    shave: Part = field(default_factory=lambda: Part(False, "#000000", 0.8))      # shave it off
    transfer: Part = field(default_factory=lambda: Part(False, "#000000", 1.0))   # makeup copied from a photo


def _lab(img):
    return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


def _tint(lab, mask, colour, strength, light=0.6, keep=0.35):
    """Blend `colour` into `lab` where `mask` is: brightness `light` of the way to the colour's, tint all
    the way, the region's own variation kept at `keep`."""
    if strength <= 0 or mask.max() <= 0:
        return lab
    t = _lab(np.uint8([[rgb(colour)]]))[0, 0]
    sel = mask > 0.05
    mean = lab[sel].mean(0)
    out = lab.copy()
    out[..., 0] = lab[..., 0] + (t[0] - mean[0]) * light
    out[..., 1] = t[1] + (lab[..., 1] - mean[1]) * keep
    out[..., 2] = t[2] + (lab[..., 2] - mean[2]) * keep
    a = (np.clip(mask * strength, 0, 1))[..., None]
    return lab * (1 - a) + out * a


def _resample(points, n):
    """n points evenly along a polyline."""
    p = np.asarray(points, np.float32)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    at = np.linspace(0, seg[-1], n)
    return np.stack([np.interp(at, seg, p[:, 0]), np.interp(at, seg, p[:, 1])], 1)


def _poly_mask(shape, polys, feather):
    m = np.zeros(shape[:2], np.float32)
    for poly in polys:
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1.0, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), max(feather, 0.5)) if feather else m


def eyeshadow_mask(face):
    if face.points is None:
        return None
    pts, polys = face.points, []
    for lid, brow in zip(LIDS, BROWS_LOW):
        lid_p = pts[lid]
        brow_p = _resample(pts[brow][::-1], len(lid))          # outer to inner, like the lid
        crease = lid_p + (brow_p - lid_p) * 0.55               # up to just past the crease
        polys.append(np.concatenate([lid_p, crease[::-1]]))
    m = _poly_mask(face.image.shape, polys, face.size * 0.025)
    return m * (1 - face.mask("l_eye", "r_eye", feather=1.5))   # never on the eye itself


def eyeliner_mask(face, strength):
    if face.points is None:
        return None
    pts = face.points
    m = np.zeros(face.image.shape[:2], np.float32)
    thick = max(1.0, face.size * (0.004 + 0.010 * strength))
    for lid in LIDS:
        p = pts[lid]
        outer, inner = p[0], p[-1]
        along = (outer - inner) / (np.linalg.norm(outer - inner) + 1e-6)
        up = np.array([along[1], -along[0]]) if along[0] < 0 else np.array([-along[1], along[0]])
        if up[1] > 0:
            up = -up
        wing = outer + along * face.size * 0.05 * strength + up * face.size * 0.025 * strength
        line = np.vstack([p[::-1], wing])
        for i in range(len(line) - 1):                     # thicker towards the outer corner
            w = thick * (0.5 + 0.8 * i / (len(line) - 1))
            cv2.line(m, tuple(np.round(line[i]).astype(int)), tuple(np.round(line[i + 1]).astype(int)), 1.0,
                     max(1, int(round(w))), cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), 0.8)


def blush_mask(face):
    if face.points is None:
        return None
    pts = face.points
    m = np.zeros(face.image.shape[:2], np.float32)
    r = face.size * 0.11
    for c, t in zip(CHEEKS, TEMPLES):
        centre = pts[c] + (pts[t] - pts[c]) * 0.25
        angle = np.degrees(np.arctan2(*(pts[t] - pts[c])[::-1]))
        cv2.ellipse(m, tuple(np.round(centre).astype(int)), (int(r * 1.3), int(r * 0.8)), angle, 0, 360, 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), r * 0.45)
    return m * face.mask("skin", feather=2)


class Painter:
    """Masks are worked out once per photo; apply() is then quick enough for every slider move."""

    def __init__(self, face):
        self.face = face
        self.lab = _lab(face.image)
        f = face.size
        self.masks = {
            "hair": face.mask("hair", feather=max(1.0, f * 0.004)),
            "lips": face.mask("u_lip", "l_lip", feather=max(1.0, f * 0.006)),
            "eyeshadow": eyeshadow_mask(face),
            "blush": blush_mask(face),
            "brows": face.mask("l_brow", "r_brow", feather=max(1.0, f * 0.004)),
            "skin": face.mask("skin", "nose", feather=max(1.0, f * 0.01)),
        }
        area = facial_hair.area_mask(face)
        self.masks["beard_area"] = area
        self.masks["beard"] = facial_hair.beard_mask(face, self.lab, area) if area is not None else None
        self.transfer = None        # the copied makeup: a LAB change the size of the photo (transfer.py)
        self._smooth = None
        self._shaved = {}           # strength -> the shaved photo (LAB)
        self._stubble = None

    def apply(self, look, base=None):
        """The photo with `look` painted on. base: paint onto this picture instead (RGB, same size): a
        second layer goes onto the first layer's result, with the same masks."""
        lab = self.lab if base is None else _lab(base)
        m = self.masks
        if look.shave.on and m["beard"] is not None and look.shave.strength > 0:
            skin = self.face.mask("skin")
            if base is not None:
                lab = _lab(facial_hair.shave(base, m["beard"], look.shave.strength, skin))
            else:
                key = round(look.shave.strength, 2)
                if key not in self._shaved:
                    self._shaved = {key: _lab(facial_hair.shave(self.face.image, m["beard"], key, skin))}
                lab = self._shaved[key]
        if look.transfer.on and self.transfer is not None and look.transfer.strength > 0:
            lab = lab + self.transfer * look.transfer.strength
        if look.beard.on and m["beard"] is not None:
            lab = _tint(lab, m["beard"], look.beard.colour, look.beard.strength, light=0.8, keep=0.5)
        if look.stubble.on and m["beard_area"] is not None and look.stubble.strength > 0:
            if self._stubble is None:
                self._stubble = facial_hair.stubble_texture(self.face.image.shape, self.face.size)
            lab = _tint(lab, m["beard_area"] * self._stubble, look.stubble.colour, look.stubble.strength * 0.85,
                        light=0.7, keep=0.4)
        if look.skin.on and look.skin.strength > 0:
            if self._smooth is None:
                d = max(5, int(self.face.size * 0.03)) | 1
                self._smooth = _lab(cv2.bilateralFilter(self.face.image, d, 40, d))
            a = (m["skin"] * look.skin.strength * 0.85)[..., None]
            lab = lab * (1 - a) + self._smooth * a
        if look.hair.on:
            lab = _tint(lab, m["hair"], look.hair.colour, look.hair.strength, light=0.95, keep=0.45)
        if look.brows.on:
            lab = _tint(lab, m["brows"], look.brows.colour, look.brows.strength, light=0.6, keep=0.3)
        if look.eyeshadow.on and m["eyeshadow"] is not None:
            lab = _tint(lab, m["eyeshadow"], look.eyeshadow.colour, look.eyeshadow.strength * 0.8, light=0.45, keep=0.5)
        if look.blush.on and m["blush"] is not None:
            lab = _tint(lab, m["blush"], look.blush.colour, look.blush.strength * 0.55, light=0.2, keep=0.7)
        if look.lips.on:
            lab = _tint(lab, m["lips"], look.lips.colour, look.lips.strength, light=0.55, keep=0.3)
            if look.lips.gloss > 0:                         # a highlight where the lips already catch light
                L = lab[..., 0]
                sel = m["lips"] > 0.3
                if sel.any():
                    hi = np.clip((L - np.percentile(L[sel], 80)) / 15, 0, 1) * m["lips"]
                    lab[..., 0] = L + hi * look.lips.gloss * 25
        if look.eyeliner.on:
            liner = eyeliner_mask(self.face, look.eyeliner.strength)
            if liner is not None:
                lab = _tint(lab, liner, look.eyeliner.colour, 0.95, light=1.0, keep=0.1)
        out = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
        return np.clip(out * 255, 0, 255).astype(np.uint8)
