"""facial_hair.py - Cosmeta for beards: colour what's there, paint stubble in, shave stubble off.

The beard area comes from Face Mesh: the jaw line from cheek to cheek under the chin, up to the
nostrils and the cheeks, without the lips. Inside it, "beard" is whatever is clearly darker than the
face's own skin (so it works on any skin tone and any beard), softened at the edges.

    colour   tint the beard and moustache (grey to black, a dyed red, bleached)
    stubble  a fine stipple of dark points over the area: a shadow of a beard
    shave    the skin blended back over the beard (good for stubble and short beards; a full beard
             needs a generative edit: Cosmeta sends "clean shaven" to AetherCanvas AI)
"""
import cv2
import numpy as np

# Face Mesh: the jaw from one cheek round the chin to the other, and back over the nostrils
JAW = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]
TOP = [425, 358, 2, 129, 205]
BEARD_COLOURS = {"Black": "#151210", "Dark brown": "#3a2618", "Brown": "#5e4028", "Ginger": "#a0502a",
                 "Blonde": "#b89060", "Grey": "#8a8a88", "White": "#d8d6d0"}


def area_mask(face):
    """The whole beard area (float 0..1), lips left out."""
    if face.points is None:
        return None
    poly = np.concatenate([face.points[JAW], face.points[TOP]])
    m = np.zeros(face.image.shape[:2], np.float32)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1.0, cv2.LINE_AA)
    m *= 1 - face.mask("u_lip", "l_lip", "mouth", feather=2)
    return cv2.GaussianBlur(m, (0, 0), max(1.0, face.size * 0.02))


def beard_mask(face, lab, area):
    """The beard itself: darker than the face's skin, inside the area."""
    skin = face.mask("skin") > 0.5
    cheeks = skin & (area < 0.2)
    ref = np.median(lab[..., 0][cheeks if cheeks.any() else skin])
    dark = np.clip((ref - lab[..., 0] - 6) / 18, 0, 1)
    return cv2.GaussianBlur(dark * area, (0, 0), max(1.0, face.size * 0.004))


def stubble_texture(shape, size, seed=7):
    """A shadow of a beard: an even grey-blue shade with a fine grain in it (not dots on their own,
    which read as sandpaper)."""
    rng = np.random.default_rng(seed)
    grain = cv2.GaussianBlur(rng.random(shape[:2]).astype(np.float32), (0, 0), max(0.6, size * 0.0022))
    grain = (grain - grain.min()) / max(np.ptp(grain), 1e-6)
    return 0.65 + 0.35 * grain


def shave(img_rgb, mask, strength, skin):
    """Skin over the beard. The colour is the face's own skin from around it (a normalised blur of the
    skin that isn't beard, so nothing comes in from the background), with a little of the photo's own
    light and shade kept so it isn't flat."""
    src = img_rgb.astype(np.float32)
    clean = (skin * (1 - np.clip(mask * 3, 0, 1))).astype(np.float32)
    sigma = max(4.0, min(img_rgb.shape[:2]) * 0.02)
    num = cv2.GaussianBlur(src * clean[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(clean, (0, 0), sigma)[..., None]
    tone = num / np.maximum(den, 1e-4)
    light = cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), sigma * 2)
    tone *= (0.85 + 0.15 * light / max(light.mean(), 1))[..., None]
    a = np.clip(mask * strength * 1.4, 0, 1)[..., None]
    return np.clip(src * (1 - a) + tone * a, 0, 255).astype(np.uint8)
