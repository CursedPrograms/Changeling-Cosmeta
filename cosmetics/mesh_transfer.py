"""mesh_transfer.py - copy the makeup from a photo exactly where it is: the reference's face, warped onto yours.

Both faces are read with Face Mesh (478 points), and a ring of points just outside each face (taller over the
forehead) is added the same way to both. The reference is cut into the triangles of those points and each one is
warped onto your face's matching triangle, so a star over the left eye lands over your left eye, a clown's mouth
on your mouth, a skull's nose on your nose, whatever your face's shape and pose. Then:

    face paint  the reference's colours, outright: its white, its black, its pattern. Your skin's fine texture
                (pores, creases) stays under it, so it looks painted on rather than pasted.
    makeup      only what the makeup changed on the reference: each place compared with the reference's own
                bare skin (worked out around it), and that change added to your skin. Lipstick, liner, shadow
                and blush come across; your skin tone stays yours. The lips take the reference's lip colour.

Either way your eyes (the eyeball, between the lids) and your open mouth stay your own: the reference's are filled
in from around them before warping, so its teeth and eye whites don't land on your lips and lids. It works on the
full-size photo (no 256-pixel model), in about half a second.

The reference must show the face from the front-ish. A painted face Face Mesh can't read at first is cropped to
the face OpenCV's YuNet finds and read again.
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YUNET = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
INNER_MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


def mesh_points(mesh, rgb):
    """478 x 2 Face Mesh points for the biggest face in rgb, or None. A face the mesh can't read at first (heavy
    face paint hides what it looks for) is cropped to YuNet's box and read again."""
    found = mesh.process(rgb)
    h, w = rgb.shape[:2]
    if found.multi_face_landmarks:
        return np.array([(p.x * w, p.y * h) for p in found.multi_face_landmarks[0].landmark], np.float32)
    det = cv2.FaceDetectorYN.create(YUNET, "", (w, h), 0.4)
    _, faces = det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if faces is None:
        return None
    x, y, bw, bh = faces[np.argmax(faces[:, -1])][:4].astype(int)
    m = int(max(bw, bh) * 0.6)
    x0, y0 = max(x - m, 0), max(y - m, 0)
    crop = np.ascontiguousarray(rgb[y0:y + bh + m, x0:x + bw + m])
    found = mesh.process(crop)
    if not found.multi_face_landmarks:
        return None
    ch, cw = crop.shape[:2]
    return np.array([(p.x * cw + x0, p.y * ch + y0) for p in found.multi_face_landmarks[0].landmark], np.float32)


def extended(points):
    """The mesh's 468 points and a ring around the face: 1.12 x the face oval below the eyes, up to 1.4 x over the
    forehead (the mesh stops short of the hairline). Built the same way on both faces, so they still match."""
    pts = points[:468]
    oval = pts[FACE_OVAL]
    centre = pts[[33, 263, 152]].mean(0)                 # between the eyes and the chin
    up = pts[152] - pts[10]
    up = -up / (np.linalg.norm(up) + 1e-6)               # chin -> forehead
    ring = []
    for p in oval:
        v = p - centre
        k = 1.12 + 0.28 * max(0.0, float(np.dot(v / (np.linalg.norm(v) + 1e-6), up)))
        ring.append(centre + v * k)
    return np.vstack([pts, np.array(ring, np.float32)])


def triangles(points, shape):
    """Delaunay triangles of points (as index triples), on this face's own layout."""
    h, w = shape[:2]
    rect = (-w, -h, 3 * w, 3 * h)
    sub = cv2.Subdiv2D(rect)
    index = {}
    for i, (x, y) in enumerate(points):
        key = (float(x), float(y))
        if key in index:
            continue
        index[key] = i
        sub.insert(key)
    tris = []
    for t in sub.getTriangleList():
        corners = [(float(t[0]), float(t[1])), (float(t[2]), float(t[3])), (float(t[4]), float(t[5]))]
        ids = [index.get(c) for c in corners]
        if None not in ids:
            tris.append(ids)
    return np.array(tris, np.int32)


def warp(src, src_pts, dst_pts, tris, shape):
    """src (H x W x C float32) cut into tris at src_pts and laid onto dst_pts in an image of `shape`; with the
    coverage (H x W: 1 where a triangle landed)."""
    h, w = shape[:2]
    c = src.shape[2]
    out = np.zeros((h, w, c), np.float32)
    cover = np.zeros((h, w), np.float32)
    sh, sw = src.shape[:2]
    for t in tris:
        s, d = src_pts[t].astype(np.float32), dst_pts[t].astype(np.float32)
        x, y, bw, bh = cv2.boundingRect(d)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
        if x1 <= x0 or y1 <= y0:
            continue
        sx, sy, sbw, sbh = cv2.boundingRect(s)
        sx0, sy0 = max(sx - 2, 0), max(sy - 2, 0)
        sx1, sy1 = min(sx + sbw + 2, sw), min(sy + sbh + 2, sh)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        m = cv2.getAffineTransform((s - (sx0, sy0)).astype(np.float32), (d - (x0, y0)).astype(np.float32))
        patch = cv2.warpAffine(src[sy0:sy1, sx0:sx1], m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        if patch.ndim == 2:
            patch = patch[..., None]
        mask = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.fillConvexPoly(mask, np.round(d - (x0, y0)).astype(np.int32), 1.0, lineType=cv2.LINE_AA)
        region = out[y0:y1, x0:x1]
        take = mask[..., None] * (1 - cover[y0:y1, x0:x1, None])
        out[y0:y1, x0:x1] = region + patch * take
        cover[y0:y1, x0:x1] = np.minimum(cover[y0:y1, x0:x1] + mask, 1)
    return out, cover


def _poly(shape, pts, idx, grow=0.0):
    m = np.zeros(shape[:2], np.uint8)
    p = pts[idx]
    if grow:
        c = p.mean(0)
        p = c + (p - c) * (1 + grow)
    cv2.fillPoly(m, [np.round(p).astype(np.int32)], 255)
    return m


def _lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


class MeshTransfer:
    def __init__(self):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                                    min_detection_confidence=0.3)

    def delta(self, face, reference, paint=False):
        """The change to `face` (a read Face) that puts `reference`'s makeup on it: a full-size LAB difference
        (H x W x 3), and the result as a picture (RGB) for a quick look. reference: an RGB photo, or a Face."""
        ref_img = reference.image if hasattr(reference, "image") else reference
        ref_pts = mesh_points(self.mesh, ref_img)
        if ref_pts is None:
            raise ValueError("No face found in the makeup photo.")
        if face.points is None:
            raise ValueError("No face found in the photo.")
        img, pts = face.image, face.points
        size = face.size
        src = extended(ref_pts)
        dst = extended(pts)

        # the reference's own eyes and open mouth, filled in from around them first
        fill = np.maximum.reduce([_poly(ref_img.shape, ref_pts, INNER_MOUTH, 0.08),
                                  _poly(ref_img.shape, ref_pts, LEFT_EYE, 0.05),
                                  _poly(ref_img.shape, ref_pts, RIGHT_EYE, 0.05)])
        ref_filled = cv2.inpaint(ref_img, fill, max(3, int(face.size * 0.01)), cv2.INPAINT_TELEA)
        tris = triangles(dst, img.shape)
        warped, cover = warp(_lab(ref_filled), src, dst, tris, img.shape)

        T = _lab(img)
        # where it goes: your face (the parser's skin, nose, brows, lips, inside the ring), not your eyes or open mouth
        if face.labels is not None and face.labels.any():
            region = face.mask("skin", "nose", "l_brow", "r_brow", "u_lip", "l_lip")
        else:
            region = np.ones(img.shape[:2], np.float32)
        own = np.maximum.reduce([_poly(img.shape, pts, INNER_MOUTH), _poly(img.shape, pts, LEFT_EYE),
                                 _poly(img.shape, pts, RIGHT_EYE)]).astype(np.float32) / 255
        own = cv2.GaussianBlur(own, (0, 0), max(1.0, size * 0.004))
        weight = region * cover * (1 - own)
        weight = cv2.GaussianBlur(weight, (0, 0), max(1.0, size * 0.006))

        lips = cv2.GaussianBlur(_poly(img.shape, pts, LIPS).astype(np.float32) / 255 *
                                (1 - _poly(img.shape, pts, INNER_MOUTH).astype(np.float32) / 255), (0, 0),
                                max(1.0, size * 0.004))
        fine = max(1.0, size * 0.006)
        detail = T - cv2.GaussianBlur(T, (0, 0), fine)                 # your skin's own fine texture
        painted = cv2.GaussianBlur(warped, (0, 0), fine * 0.5) + detail * np.float32([1.0, 0.3, 0.3])
        if paint:
            out = painted
        else:
            # the reference's bare skin, worked out around its makeup: its skin, with the made-up places
            # (eyes' surroundings, lips, brows: on your face's layout now) left out and filled in by a wide blur
            made_up = np.maximum.reduce([_poly(img.shape, pts, LEFT_EYE, 1.0), _poly(img.shape, pts, RIGHT_EYE, 1.0),
                                         _poly(img.shape, pts, LIPS, 0.25)]).astype(np.float32) / 255
            if face.labels is not None and face.labels.any():
                made_up = np.maximum(made_up, face.mask("l_brow", "r_brow"))
            skin = cover * region * (1 - made_up)
            wide = max(2.0, size * 0.09)
            bare = cv2.GaussianBlur(warped * skin[..., None], (0, 0), wide) / \
                np.maximum(cv2.GaussianBlur(skin, (0, 0), wide), 1e-4)[..., None]
            change = warped - bare
            # only real makeup: a clear change of colour, or darker (liner, shadow); not the reference's own
            # highlights and shadows of its face's shape, which are small and mostly lightness
            strength = np.sqrt(change[..., 1] ** 2 + change[..., 2] ** 2) + np.maximum(-change[..., 0], 0) * 0.6
            keep = np.clip((strength - 4) / 8, 0, 1)
            keep = cv2.GaussianBlur(keep, (0, 0), fine)
            out = T + change * keep[..., None]
            out = out * (1 - lips[..., None]) + painted * lips[..., None]   # lips: the reference's lipstick
        d = (out - T) * weight[..., None]
        result = cv2.cvtColor(T + d, cv2.COLOR_LAB2RGB)
        return d, np.clip(result * 255, 0, 255).astype(np.uint8)
