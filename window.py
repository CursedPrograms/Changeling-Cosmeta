"""window.py - Changeling Cosmeta's window, in Changeling's look (customtkinter, the Cursed pink).

Open a photo; the face is read once (which pixel is hair, skin, lips, brows; where the face's points are).
Then every control answers at once: hair colour, lips, eyes, cheeks, brows, skin, and the basic editor, on two
layers that stack (each with its own looks, editor, opacity and on/off). Hair length and photographic looks need a
generative model: those go to AetherCanvas AI when it's running. (Beards moved to Changeling Tonsor, the barber.)
Save writes the full-size result.
"""
import os
import threading
import time
import tkinter as tk
import webbrowser
from gallery import remember, show as show_gallery
from dataclasses import dataclass, field, fields
from tkinter import colorchooser, filedialog

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from cosmetics import aethercanvas
from cosmetics.adjust import Adjust
from cosmetics.facial_hair import BEARD_COLOURS
from cosmetics.looks import (BLEND_MODES, BLUSH_COLOURS, BROW_COLOURS, HAIR_COLOURS, LINER_COLOURS, LIP_COLOURS,
                             SHADOW_COLOURS, Look, Painter)

HERE = os.path.dirname(os.path.abspath(__file__))
GITHUB = "https://github.com/CursedPrograms/Changeling-Cosmeta"
KOFI = "https://ko-fi.com/cursedentertainment"
PINK = "#e0408a"
PREVIEW = 1100                  # the preview is worked on at most this many pixels across; Save is full size
IMAGE_TYPES = [("Pictures", "*.png *.jpg *.jpeg *.webp *.bmp *.jfif"), ("All files", "*.*")]
STUBBLE_COLOURS = {"Dark": "#1a1410", "Brown": "#3a2618", "Grey": "#6a6a68"}

# (group, label, Look field, colours or None, has gloss)
PARTS = [
    ("Hair", "Hair colour", "hair", HAIR_COLOURS, False),
    ("Face", "Lips", "lips", LIP_COLOURS, True),
    ("Face", "Eyeshadow", "eyeshadow", SHADOW_COLOURS, False),
    ("Face", "Eyeliner", "eyeliner", LINER_COLOURS, False),
    ("Face", "Blush", "blush", BLUSH_COLOURS, False),
    ("Face", "Brows", "brows", BROW_COLOURS, False),
    ("Face", "Smooth skin", "skin", None, False),
]
ADJUST = [("brightness", "Brightness"), ("contrast", "Contrast"), ("saturation", "Saturation"),
          ("warmth", "Warmth"), ("vignette", "Vignette"), ("sharpen", "Sharpen")]


@dataclass
class Layer:
    look: Look = field(default_factory=Look)
    adjust: Adjust = field(default_factory=Adjust)
    opacity: float = 1.0
    visible: bool = True


def read_rgb(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if ok:
        buf.tofile(path)
    return ok


def compose(painter, layers):
    """Both layers, the second over the first, each with its own looks, editor and opacity."""
    base = painter.face.image
    out = base
    for layer in layers:
        if not layer.visible or layer.opacity <= 0:
            continue
        painted = painter.apply(layer.look, base=None if out is base else out)
        painted = layer.adjust.apply(painted)
        if layer.opacity >= 1:
            out = painted
        else:
            out = cv2.addWeighted(painted, layer.opacity, out, 1 - layer.opacity, 0)
    return out


def set_logo(root, here):
    """The Changeling logo as the window's icon (after customtkinter puts its own in, ~200 ms)."""
    ico = os.path.join(here, "logo.ico")
    if os.name == "nt" and os.path.exists(ico):
        root.after(300, lambda: root.iconbitmap(ico))


def logo_image(here, size=56):
    png = os.path.join(here, "logo.png")
    return ctk.CTkImage(Image.open(png), size=(size, size)) if os.path.exists(png) else None


class Window:
    def __init__(self, make_parser):
        self.make_parser = make_parser
        self.parser = None
        self.photo = None            # the photo (RGB), or the AetherCanvas edit of it
        self.photo_path = None
        self.painter = None          # full size, for Save
        self.preview = None          # small, for the screen
        self.result = None           # the preview result (RGB)
        self.layers = [Layer(), Layer()]
        self.current = 0
        self.job = None
        self.state = {"busy": False, "stage": "", "message": ""}
        self.lock = threading.Lock()
        self._render_after = None
        self._loading = False        # while widgets are being set from a layer: no renders
        self.controls = {}

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme(os.path.join(HERE, "theme.json"))
        self.root = ctk.CTk()
        set_logo(self.root, HERE)
        self.root.title("Changeling Cosmeta")
        self.root.geometry("1380x860")
        self.root.minsize(1060, 680)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._build()
        self.root.bind("<Control-o>", lambda e: self.open())
        self.root.bind("<Control-s>", lambda e: self.save())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.after(100, self.watch)
        self._run(self._load_parser, "Loading the face reader")
        threading.Thread(target=self._check_aethercanvas, daemon=True).start()

    # ---- the window
    def _build(self):
        stage = ctk.CTkFrame(self.root, corner_radius=12)
        stage.grid(row=0, column=0, sticky="nsew", padx=(14, 7), pady=14)
        stage.grid_columnconfigure((0, 1), weight=1, uniform="half")
        stage.grid_rowconfigure(1, weight=1)
        for col, text in enumerate(("Before", "After")):
            ctk.CTkLabel(stage, text=text.upper(), text_color="gray50",
                         font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=col, pady=(8, 0))
        self.before = tk.Label(stage, bg="#16161a", bd=0, text="Open a photo  (Ctrl+O)", fg="gray60",
                               font=("Segoe UI", 12), cursor="hand2")
        self.before.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self.before.bind("<Button-1>", lambda e: self.open())
        self.after_view = tk.Label(stage, bg="#16161a", bd=0, fg="gray60", font=("Segoe UI", 12))
        self.after_view.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self.before.bind("<Configure>", lambda e: self.root.after_idle(self._refresh))

        side = ctk.CTkScrollableFrame(self.root, width=330, corner_radius=12)
        side.grid(row=0, column=1, sticky="ns", padx=(7, 14), pady=14)
        logo = logo_image(HERE)
        if logo is not None:
            ctk.CTkLabel(side, image=logo, text="").pack(anchor="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(side, text="CHANGELING", text_color="gray55",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(side, text="Cosmeta", font=ctk.CTkFont(size=24, weight="bold"), text_color=PINK).pack(anchor="w", padx=12)
        ctk.CTkLabel(side, text="makeup, hair colour and length", text_color="gray60").pack(anchor="w", padx=12)

        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(14, 4))
        ctk.CTkButton(row, text="Open photo  (Ctrl+O)", command=self.open).pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.save_btn = ctk.CTkButton(row, text="Save", width=70, command=self.save, state="disabled")
        self.save_btn.pack(side="left")
        ctk.CTkButton(row, text="Gallery", width=80, command=self.gallery).pack(side="left", padx=(4, 4))
        self.bar = ctk.CTkProgressBar(side, mode="determinate")
        self.bar.set(0)
        self.bar.pack(fill="x", padx=12, pady=(6, 2))
        self.status = ctk.CTkLabel(side, text="", text_color="gray65", wraplength=300, justify="left")
        self.status.pack(anchor="w", padx=12)

        # the layers
        self._section(side, "Layer")
        self.layer_pick = ctk.CTkSegmentedButton(side, values=["Layer 1", "Layer 2"], command=self.pick_layer)
        self.layer_pick.set("Layer 1")
        self.layer_pick.pack(fill="x", padx=12, pady=4)
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", padx=12)
        self.layer_on = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(row, text="Shown", variable=self.layer_on, command=self._layer_changed).pack(side="left")
        ctk.CTkButton(row, text="Clear layer", width=90, command=self.clear_layer).pack(side="right")
        self.opacity = self._slider(side, "Opacity", 0, 100, 100, self._layer_changed)

        # makeup copied from a photo (the reference's face warped onto this one: cosmetics/mesh_transfer.py)
        self._section(side, "Copy makeup from a photo")
        box = ctk.CTkFrame(side, fg_color=("gray86", "gray17"), corner_radius=8)
        box.pack(fill="x", padx=12, pady=3)
        row = ctk.CTkFrame(box, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(6, 2))
        self.ref_thumb = ctk.CTkLabel(row, text="", width=56, height=56, fg_color=("gray80", "gray22"), corner_radius=6)
        self.ref_thumb.pack(side="left")
        self.ref_btn = ctk.CTkButton(row, text="Pick a makeup photo", command=self.pick_reference)
        self.ref_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))
        from cosmetics.transfer_models import DEFAULT, MODELS
        ctk.CTkLabel(box, text="Model", text_color="gray70").pack(anchor="w", padx=8, pady=(4, 0))
        self.transfer_pick = ctk.CTkSegmentedButton(box, values=[n.split(" (")[0] for n in MODELS],
                                                    command=self.pick_transfer_model)
        self.transfer_pick.set(DEFAULT.split(" (")[0])
        self.transfer_pick.pack(fill="x", padx=8, pady=(2, 0))
        self.transfer_note = ctk.CTkLabel(box, text="Warp: exact, patterns and all, a fifth of a second.",
                                          text_color="gray55", wraplength=290, justify="left")
        self.transfer_note.pack(anchor="w", padx=8)
        self.transfer_name = DEFAULT
        self.paint_mode = ctk.CTkSegmentedButton(box, values=["Makeup", "Face paint"],
                                                 command=lambda _: self.copy_makeup())
        self.paint_mode.set("Makeup")
        self.paint_mode.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(box, text="Makeup takes only what the makeup changed, on your own skin tone. Face paint "
                     "takes its colours outright: a clown's white, a skull, corpse paint, festival colours.",
                     text_color="gray55",
                     wraplength=290, justify="left").pack(anchor="w", padx=8)
        on = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(box, text="Copied makeup", variable=on, command=self._look_changed).pack(anchor="w", padx=8, pady=(4, 2))
        strength = self._slider(box, "Strength", 0, 150, 100, self._look_changed)
        self.controls["transfer"] = (on, strength, None, None)
        self.transfer_model = None          # (model name, the loaded model)
        self.ref_image = None

        group = None
        for g, label, key, colours, gloss in PARTS:
            if g != group:
                self._section(side, g)
                group = g
            self._part(side, label, key, colours, gloss)

        self._section(side, "Editor")
        for key, label in ADJUST:
            lo, hi = Adjust.RANGES[key]
            s = self._slider(side, label, lo, hi, 0, self._adjust_changed)
            self.controls["adjust:" + key] = s

        # the generative edits
        self._section(side, "AI looks: makeup, hair  (AetherCanvas AI)")
        self.ac_note = ctk.CTkLabel(side, text="Looking for AetherCanvas AI ...", text_color="gray55",
                                    wraplength=300, justify="left")
        self.ac_note.pack(anchor="w", padx=12)
        self.ac_buttons = []
        for title, items in aethercanvas.EDITS.items():
            ctk.CTkLabel(side, text=title, text_color="gray70").pack(anchor="w", padx=12, pady=(6, 0))
            grid = ctk.CTkFrame(side, fg_color="transparent")
            grid.pack(fill="x", padx=12)
            for i, (name, instruction) in enumerate(items):
                b = ctk.CTkButton(grid, text=name, height=26, state="disabled",
                                  command=lambda t=instruction, n=name: self.generative(n, t))
                b.grid(row=i // 3, column=i % 3, sticky="ew", padx=2, pady=2)
                grid.grid_columnconfigure(i % 3, weight=1)
                self.ac_buttons.append(b)
        self.undo_btn = ctk.CTkButton(side, text="Back to the original photo", command=self.back_to_original,
                                      state="disabled", fg_color="gray25")
        self.undo_btn.pack(fill="x", padx=12, pady=(8, 4))

        links = ctk.CTkFrame(side, fg_color="transparent")
        links.pack(fill="x", padx=12, pady=(14, 2))
        for text, url in (("GitHub", GITHUB), ("Ko-fi", KOFI)):
            l = ctk.CTkLabel(links, text=text, text_color=("#c2186b", "#f29cc0"), cursor="hand2")
            l.pack(side="left", padx=(0, 14))
            l.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
        ctk.CTkLabel(side, text="© Cursed Entertainment 2026", text_color="gray45").pack(anchor="w", padx=12, pady=(0, 10))

    def _section(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), text_color="gray50",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=12, pady=(16, 1))

    def _slider(self, parent, text, lo, hi, value, on_change, unit="%"):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(4, 0))
        ctk.CTkLabel(row, text=text).pack(side="left")
        shown = ctk.CTkLabel(row, text=f"{value}", text_color="gray60")
        shown.pack(side="right")

        def moved(v):
            shown.configure(text=f"{int(v)}")
            on_change()
        s = ctk.CTkSlider(parent, from_=lo, to=hi, number_of_steps=hi - lo, command=moved)
        s.set(value)
        s.pack(fill="x", padx=12, pady=(0, 2))
        s.shown = shown
        return s

    def _part(self, parent, label, key, colours, gloss):
        box = ctk.CTkFrame(parent, fg_color=("gray86", "gray17"), corner_radius=8)
        box.pack(fill="x", padx=12, pady=3)
        on = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(box, text=label, variable=on, command=self._look_changed).pack(anchor="w", padx=8, pady=(6, 2))
        chip = None
        if colours:
            sw = ctk.CTkFrame(box, fg_color="transparent")
            sw.pack(fill="x", padx=8)
            for i, (name, colour) in enumerate(colours.items()):
                b = ctk.CTkButton(sw, text="", width=22, height=22, corner_radius=11, fg_color=colour,
                                  hover_color=colour, border_width=0,
                                  command=lambda k=key, c=colour, v=on: self._pick_colour(k, c, v))
                b.grid(row=i // 9, column=i % 9, padx=2, pady=2)
                tip = name
                b.bind("<Enter>", lambda e, t=tip: self.status.configure(text=t))
            more = ctk.CTkButton(sw, text="+", width=22, height=22, corner_radius=11, fg_color="gray30",
                                 command=lambda k=key, v=on: self._custom_colour(k, v))
            n = len(colours)
            more.grid(row=n // 9, column=n % 9, padx=2, pady=2)
            chip = ctk.CTkLabel(box, text="", width=60, height=6, corner_radius=3, fg_color=getattr(Look(), key).colour)
            chip.pack(anchor="w", padx=10, pady=(2, 0))
        if colours and key in ("hair", "lips", "eyeshadow", "blush", "brows"):     # how the colour goes on
            row = ctk.CTkFrame(box, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=(4, 0))
            ctk.CTkLabel(row, text="Blend").pack(side="left")
            menu = ctk.CTkOptionMenu(row, values=BLEND_MODES, width=130, command=lambda _: self._look_changed())
            menu.set("Natural")
            menu.pack(side="right")
            self.blend_menus = getattr(self, "blend_menus", {})
            self.blend_menus[key] = menu
        strength = self._slider(box, "Strength", 0, 100, int(getattr(Look(), key).strength * 100), self._look_changed)
        g = self._slider(box, "Gloss", 0, 100, int(Look().lips.gloss * 100), self._look_changed) if gloss else None
        self.controls[key] = (on, strength, g, chip)

    # ---- the layers and the controls
    @property
    def layer(self):
        return self.layers[self.current]

    def pick_layer(self, name):
        self.current = 0 if name == "Layer 1" else 1
        self._load_widgets()

    def clear_layer(self):
        self.layers[self.current] = Layer()
        self._load_widgets()
        self._schedule()

    def _load_widgets(self):
        """Every control shows the chosen layer's values."""
        self._loading = True
        layer = self.layer
        self.layer_on.set(layer.visible)
        self.opacity.set(layer.opacity * 100)
        self.opacity.shown.configure(text=f"{int(layer.opacity * 100)}")
        for key, (on, strength, gloss, chip) in ((k, v) for k, v in self.controls.items() if not k.startswith("adjust:")):
            part = getattr(layer.look, key)
            on.set(part.on)
            strength.set(part.strength * 100)
            strength.shown.configure(text=f"{int(part.strength * 100)}")
            if gloss is not None:
                gloss.set(part.gloss * 100)
                gloss.shown.configure(text=f"{int(part.gloss * 100)}")
            if chip is not None:
                chip.configure(fg_color=part.colour)
            if key in getattr(self, "blend_menus", {}):
                self.blend_menus[key].set(part.blend)
        for f in fields(Adjust):
            s = self.controls["adjust:" + f.name]
            s.set(getattr(layer.adjust, f.name))
            s.shown.configure(text=f"{getattr(layer.adjust, f.name)}")
        self._loading = False

    def _look_changed(self, *_):
        if self._loading:
            return
        for key, (on, strength, gloss, chip) in ((k, v) for k, v in self.controls.items() if not k.startswith("adjust:")):
            part = getattr(self.layer.look, key)
            part.on = on.get()
            part.strength = strength.get() / 100
            if gloss is not None:
                part.gloss = gloss.get() / 100
            if key in getattr(self, "blend_menus", {}):
                part.blend = self.blend_menus[key].get()
        self._schedule()

    def _adjust_changed(self, *_):
        if self._loading:
            return
        for f in fields(Adjust):
            setattr(self.layer.adjust, f.name, int(self.controls["adjust:" + f.name].get()))
        self._schedule()

    def _layer_changed(self, *_):
        if self._loading:
            return
        self.layer.visible = self.layer_on.get()
        self.layer.opacity = self.opacity.get() / 100
        self._schedule()

    def _pick_colour(self, key, colour, on):
        getattr(self.layer.look, key).colour = colour
        on.set(True)
        chip = self.controls[key][3]
        if chip is not None:
            chip.configure(fg_color=colour)
        self._look_changed()

    def _custom_colour(self, key, on):
        picked = colorchooser.askcolor(color=getattr(self.layer.look, key).colour, parent=self.root)[1]
        if picked:
            self._pick_colour(key, picked, on)

    # ---- the picture
    def _schedule(self):
        if self._render_after is not None:
            self.root.after_cancel(self._render_after)
        self._render_after = self.root.after(60, self.render)

    def render(self):
        self._render_after = None
        if self.preview is None:
            return
        self.result = compose(self.preview, self.layers)
        self._show(self.after_view, self.result)

    def _show(self, label, image):
        if image is None:
            label.configure(image="")
            label.image = None
            return
        w, h = max(label.winfo_width(), 64), max(label.winfo_height(), 64)
        ih, iw = image.shape[:2]
        k = min(w / iw, h / ih)
        small = cv2.resize(image, (max(int(iw * k), 1), max(int(ih * k), 1)), interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(small))
        label.configure(image=photo, text="")
        label.image = photo

    def _refresh(self):
        self._show(self.before, self.photo)
        self._show(self.after_view, self.result)

    # ---- the jobs, on their own thread
    def _run(self, work, stage):
        with self.lock:
            self.state.update(busy=True, stage=stage, message="")
        self.job = threading.Thread(target=self._guard, args=(work,), daemon=True)
        self.job.start()

    def _guard(self, work):
        try:
            message = work()
        except Exception as e:
            message = f"Couldn't: {e}"
        with self.lock:
            self.state.update(busy=False, message=message or "")

    def watch(self):
        with self.lock:
            s = dict(self.state)
        if s["busy"]:
            if self.bar.cget("mode") != "indeterminate":
                self.bar.configure(mode="indeterminate")
                self.bar.start()
            self.status.configure(text=s["stage"] + " ...")
        elif self.job is not None and not self.job.is_alive():
            self.job = None
            self.bar.stop()
            self.bar.configure(mode="determinate")
            self.bar.set(1 if self.preview is not None else 0)
            self.status.configure(text=s["message"])
            self.save_btn.configure(state="normal" if self.preview is not None else "disabled")
            self._refresh()
            self.render()
        self.root.after(100, self.watch)

    def _load_parser(self):
        self.parser = self.make_parser()
        if self.photo is not None:
            return self._read(self.photo)
        return "Ready: open a photo."

    def _read(self, image):
        face = self.parser.read(image)
        self.painter = Painter(face)
        self.preview = Painter(face.resized(PREVIEW))
        return "Ready: switch looks on, pick colours."

    def _check_aethercanvas(self):
        up = aethercanvas.available()
        text = ("AetherCanvas AI is running: these send the photo to it (a minute or two each)." if up else
                "Start AetherCanvas AI (github.com/CursedPrograms/AetherCanvas-AI) for these.")
        self.root.after(0, lambda: (self.ac_note.configure(text=text),
                                    [b.configure(state="normal" if up else "disabled") for b in self.ac_buttons]))

    def open(self, path=None):
        path = path or filedialog.askopenfilename(title="A photo with a face", filetypes=IMAGE_TYPES)
        if not path:
            return
        image = read_rgb(path)
        if image is None:
            self.status.configure(text=f"Can't read {os.path.basename(path)}")
            return
        self.photo, self.photo_path, self.original = image, path, image
        self.preview = self.painter = self.result = None
        self.undo_btn.configure(state="disabled")
        self._refresh()
        if self.parser is not None and self.job is None:
            self._run(lambda: self._read(image), "Reading the face")

    def pick_reference(self):
        """A photo whose makeup to copy (opens in makeup_refs/)."""
        if self.painter is None or self.job is not None:
            self.status.configure(text="Open a photo first.")
            return
        path = filedialog.askopenfilename(title="A photo with the makeup to copy", filetypes=IMAGE_TYPES,
                                          initialdir=os.path.join(HERE, "makeup_refs"))
        if not path:
            return
        ref_image = read_rgb(path)
        if ref_image is None:
            self.status.configure(text=f"Can't read {os.path.basename(path)}")
            return
        h, w = ref_image.shape[:2]
        s = min(h, w)
        square = ref_image[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        self.ref_thumb.configure(image=ctk.CTkImage(Image.fromarray(square), size=(56, 56)), text="")
        self.ref_image, self.ref_name = ref_image, os.path.basename(path)
        self.copy_makeup()

    def pick_transfer_model(self, short):
        """The Model switch: Warp, DMT or Stable-Makeup. One that isn't here says why and the last one stays."""
        from cosmetics.transfer_models import MODELS
        name = next(n for n in MODELS if n.split(" (")[0] == short)
        ok, why = MODELS[name].available()
        notes = {"Warp": "Warp: exact, patterns and all, a fifth of a second.",
                 "GAN": "GAN (classic): a research model (DMT) at 256 pixels; colours and style, not drawn patterns.",
                 "Stable-Makeup": "Stable-Makeup: a diffusion model redraws the makeup; seconds a photo, GPU."}
        if not ok:
            self.transfer_pick.set(self.transfer_name.split(" (")[0])
            self.transfer_note.configure(text=why)
            return
        self.transfer_name = name
        self.transfer_note.configure(text=notes[short])
        self.copy_makeup()

    def copy_makeup(self):
        """Copy the picked photo's makeup (again: the mode changed) onto this layer."""
        if self.ref_image is None or self.painter is None or self.job is not None:
            return
        ref_image, name, paint = self.ref_image, self.ref_name, self.paint_mode.get() == "Face paint"
        model_name = self.transfer_name

        def work():
            from cosmetics.transfer_models import MODELS
            if self.transfer_model is None or self.transfer_model[0] != model_name:
                with self.lock:
                    self.state["stage"] = f"Loading {model_name}"
                self.transfer_model = (model_name, MODELS[model_name].make())
            delta, _ = self.transfer_model[1].delta(self.painter.face, ref_image, paint, self.parser)
            self.painter.transfer = delta
            ph, pw = self.preview.face.image.shape[:2]
            self.preview.transfer = cv2.resize(delta, (pw, ph), interpolation=cv2.INTER_AREA)
            self.layer.look.transfer.on = True
            return f"{'Face paint' if paint else 'Makeup'} copied from {name} ({model_name.split(' (')[0]}): shown on this layer."
        self._run(work, "Copying the makeup")
        self.controls["transfer"][0].set(True)

    def generative(self, name, instruction):
        if self.photo is None or self.job is not None:
            return
        image = self.photo

        def work():
            edited = aethercanvas.edit(image, instruction)
            self.photo = edited
            self._read(edited)
            return f"Done: {name}. Your looks are on top of it."
        self._run(work, f"{name}: AetherCanvas AI is editing")
        self.undo_btn.configure(state="normal")

    def back_to_original(self):
        if getattr(self, "original", None) is None or self.job is not None:
            return
        self.photo = self.original
        self.undo_btn.configure(state="disabled")
        self._run(lambda: self._read(self.original), "Reading the face")

    def gallery(self):
        """Everything made with the Changelings (saving still asks where; the gallery lists it too)."""
        show_gallery(self.root, "Cosmeta")

    def save(self):
        if self.painter is None:
            return
        stem, _ = os.path.splitext(os.path.basename(self.photo_path or "photo.png"))
        path = filedialog.asksaveasfilename(title="Save the result", defaultextension=".png",
                                            initialdir=os.path.dirname(self.photo_path or HERE),
                                            initialfile=f"{stem}_cosmetics.png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        self.status.configure(text="Saving at full size ...")
        self.root.update_idletasks()
        if write_rgb(path, compose(self.painter, self.layers)):
            self.status.configure(text=f"Saved {os.path.basename(path)}")
            remember(path, "Cosmeta")

    def run(self, path=None):
        if path:
            self.root.after(300, lambda: self.open(path))
        self.root.mainloop()
