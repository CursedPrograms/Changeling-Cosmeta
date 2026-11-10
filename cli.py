"""cli.py - Changeling Cosmeta without the window: a photo in, the same photo with a look on it out.

    python cli.py photo.jpg -o out.png --preset Glam
    python cli.py photo.jpg -o out.png --lips "Classic red" --eyeshadow Smoky:0.7 --hair "#e0408a"
    python cli.py photo.jpg -o out.png --reference makeup_refs/goth_white_face_black_lips_red_braids.jfif --paint
    python cli.py --list                     the presets and every named colour

A colour is a name from the window's swatches or a #hex, with :strength (0..1) after it if you like.
A preset and single parts mix: the parts you give win. --reference copies the makeup from a photo (the
window's Copy makeup, Warp model unless --model says otherwise); --paint copies it as face paint.
Changeling Studio's Chain runs this, in Cosmeta's own environment (venv).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# part -> (colour name, strength); the names are the window's swatches (cosmetics/looks.py)
PRESETS = {
    "Natural": {"lips": ("Nude", 0.5), "blush": ("Peach", 0.35), "brows": ("Dark brown", 0.35), "skin": (None, 0.4)},
    "Glam": {"lips": ("Classic red", 0.7), "eyeshadow": ("Rose gold", 0.5), "eyeliner": ("Black", 0.6),
             "blush": ("Pink", 0.4), "brows": ("Soft black", 0.45), "skin": (None, 0.5)},
    "Smoky": {"lips": ("Nude", 0.5), "eyeshadow": ("Smoky", 0.75), "eyeliner": ("Black", 0.7),
              "brows": ("Soft black", 0.5), "skin": (None, 0.4)},
    "Goth": {"lips": ("Plum", 0.85), "eyeshadow": ("Smoky", 0.85), "eyeliner": ("Black", 0.9),
             "brows": ("Soft black", 0.6), "hair": ("Jet black", 0.8)},
    "Cursed pink": {"lips": ("Cursed pink", 0.75), "eyeshadow": ("Plum", 0.45), "eyeliner": ("Black", 0.5),
                    "blush": ("Pink", 0.4), "hair": ("Pastel pink", 0.7)},
    "Copper hair": {"hair": ("Copper", 0.75)},
    "Platinum hair": {"hair": ("Platinum", 0.8)},
}
PARTS = ["hair", "lips", "eyeshadow", "eyeliner", "blush", "brows"]


def read_rgb(path):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    import cv2
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise SystemExit(f"Could not write image: {path}")
    buf.tofile(path)


def colours():
    from cosmetics import looks
    return {"hair": looks.HAIR_COLOURS, "lips": looks.LIP_COLOURS, "eyeshadow": looks.SHADOW_COLOURS,
            "eyeliner": looks.LINER_COLOURS, "blush": looks.BLUSH_COLOURS, "brows": looks.BROW_COLOURS}


def parse_colour(part, spec, default_strength):
    """'Classic red', 'classic red:0.8' or '#e0408a:0.6' -> (hex, strength)."""
    name, _, strength = spec.partition(":")
    name = name.strip()
    strength = float(strength) if strength else default_strength
    if name.startswith("#"):
        return name, strength
    for key, value in colours()[part].items():
        if key.lower() == name.lower():
            return value, strength
    raise SystemExit(f"--{part}: no colour called {name!r}; one of: {', '.join(colours()[part])}, or a #hex")


def make_look(args):
    from cosmetics.looks import Look
    look = Look()
    wanted = {}
    if args.preset:
        preset = next((v for k, v in PRESETS.items() if k.lower() == args.preset.lower()), None)
        if preset is None:
            raise SystemExit(f"--preset: one of {', '.join(PRESETS)}")
        for part, (name, strength) in preset.items():
            wanted[part] = (name, strength)
    for part in PARTS:
        spec = getattr(args, part)
        if spec:
            wanted[part] = (spec, None)
    if args.skin is not None:
        wanted["skin"] = (None, args.skin)
    for part, (spec, strength) in wanted.items():
        p = getattr(look, part)
        p.on = True
        if part == "skin":
            p.strength = strength
            continue
        p.colour, p.strength = parse_colour(part, spec, p.strength if strength is None else strength)
    return look


def main():
    ap = argparse.ArgumentParser(description="Changeling Cosmeta without the window: a look on a photo.")
    ap.add_argument("image", nargs="?", help="the photo")
    ap.add_argument("-o", "--output", help="where to save (default: <photo>_cosmetics.png)")
    ap.add_argument("--preset", help="a whole look: " + ", ".join(PRESETS))
    for part in PARTS:
        ap.add_argument(f"--{part}", help=f"{part} colour: a name or #hex, :strength optional")
    ap.add_argument("--skin", type=float, help="smoother skin, 0..1")
    ap.add_argument("--reference", help="copy the makeup from this photo (makeup_refs/ has some)")
    ap.add_argument("--paint", action="store_true", help="copy it as face paint (the whole face), not makeup")
    ap.add_argument("--amount", type=float, default=1.0, help="how much of the copied makeup, 0..1 (default 1)")
    ap.add_argument("--model", default="warp", choices=["warp", "gan", "stable-makeup"],
                    help="how to copy it (default warp: exact, quick)")
    ap.add_argument("--list", action="store_true", help="list the presets and colours and stop")
    args = ap.parse_args()

    if args.list:
        print("presets: " + ", ".join(PRESETS))
        for part, names in colours().items():
            print(f"{part}: " + ", ".join(names))
        return
    if not args.image:
        ap.error("give a photo")
    look = make_look(args)
    if not any(getattr(look, p).on for p in PARTS + ["skin"]) and not args.reference:
        ap.error("nothing to do: give --preset, a part (--lips ...), or --reference")

    from cosmetics.looks import Painter
    from cosmetics.parse import Parser
    photo = read_rgb(args.image)
    parser = Parser()
    print("  reading the face", flush=True)
    try:
        face = parser.read(photo)
    except ValueError as e:
        raise SystemExit(str(e))
    painter = Painter(face)
    if args.reference:
        from cosmetics.transfer_models import MODELS
        name = next(n for n in MODELS if n.lower().startswith({"warp": "warp", "gan": "gan",
                                                              "stable-makeup": "stable"}[args.model]))
        ok, why = MODELS[name].available()
        if not ok:
            raise SystemExit(why)
        print(f"  copying the {'face paint' if args.paint else 'makeup'} ({name})", flush=True)
        delta, _ = MODELS[name].make().delta(face, read_rgb(args.reference), args.paint, parser)
        painter.transfer = delta
        look.transfer.on = True
        look.transfer.strength = args.amount
    result = painter.apply(look)
    out = args.output or os.path.splitext(args.image)[0] + "_cosmetics.png"
    write_rgb(out, result)
    print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
