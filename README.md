[![Twitter: @NorowaretaGemu](https://img.shields.io/badge/X-@NorowaretaGemu-blue.svg?style=flat)](https://x.com/NorowaretaGemu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div align="center">
  <a href="https://ko-fi.com/cursedentertainment">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="ko-fi" style="width: 20%;"/>
  </a>
</div>

<p align="center"><img src="https://raw.githubusercontent.com/CursedPrograms/Changeling-Studio/main/site/logo.png" alt="Changeling" width="120"></p>

# Changeling Cosmetics

**Website:** https://cursedprograms.github.io/Changeling-Cosmetics/

![Three people, each with makeup, new hair colours, beard colour, a shave, stubble and the editor](assets/cosmetics_samples.jpg)

*Part of the Changeling family, all together in [Changeling Studio](https://github.com/CursedPrograms/Changeling-Studio):
[CopyCat](https://github.com/CursedPrograms/Changeling-CopyCat) ·
[CopyCat Realtime](https://github.com/CursedPrograms/Changeling-CopyCat-Realtime) ·
[Tiresias](https://github.com/CursedPrograms/Changeling-Tiresias) (gender) ·
[Chronos](https://github.com/CursedPrograms/Changeling-Chronos) (age) ·
[Natio](https://github.com/CursedPrograms/Changeling-Natio) (where in the world) ·
**Cosmetics** (makeup, hair, beards) ·
[Echo](https://github.com/CursedPrograms/Changeling-Echo) (voices) ·
[Tonsor](https://github.com/CursedPrograms/Changeling-Tonsor) (hairstyles) ·
realtime: [Tiresias](https://github.com/CursedPrograms/Changeling-Tiresias-Realtime) ·
[Chronos](https://github.com/CursedPrograms/Changeling-Chronos-Realtime) ·
[Natio](https://github.com/CursedPrograms/Changeling-Natio-Realtime) ·
[Cosmetics](https://github.com/CursedPrograms/Changeling-Cosmetics-Realtime).*

FaceApp-style cosmetics for photos: makeup, hair colour, beards and stubble, painted onto the face itself so
the light and texture underneath stay (a lipstick keeps the lips' lines, a hair colour keeps every strand).
Two layers that stack, a basic editor, and photographic makeup looks, hair length or a full beard through
[AetherCanvas AI](https://github.com/CursedPrograms/AetherCanvas-AI). Every face above is an AI-generated person who
doesn't exist.

### Copy makeup from a photo

![Three makeup photos (a clown's face paint, a goth's face paint, dark glam makeup) copied onto three women](assets/transfer_samples.jpg)

The top row is the makeup; each row below is a person wearing it: face paint takes the whole face, white base and
all; makeup keeps her own skin tone. It copies colours and style, not drawn patterns (cracks, stars, motifs). All
the faces are AI-generated people who don't exist.

## Use it

Start `run.bat` (or `.\run.ps1`, `./run.sh`). The first time it sets itself up: Python 3.11 with MediaPipe,
OpenCV, ONNX Runtime and customtkinter, and the face parser (`models/face_parsing_int8.onnx`, about 90 MB). No GPU
needed. `run.bat photo.jpg` opens with that photo.

![Changeling Cosmetics's window: the photo, the result, and the controls on the right](assets/window.png)

| Control | Does |
| --- | --- |
| **Open photo** (<kbd>Ctrl</kbd>+<kbd>O</kbd>), **Save** (<kbd>Ctrl</kbd>+<kbd>S</kbd>) | the face is read once (a few seconds); Save writes the full-size result |
| **Copy makeup from a photo** | pick a photo of a face wearing makeup (`makeup_refs/`). **Makeup**: its lips, eyes, blush and brows come over, your own skin tone stays. **Face paint**: everything comes over, a clown's or a goth's white base too. It copies colours and style, not drawn patterns (cracks, stars, motifs) |
| **Layer 1 / Layer 2** | two layers that stack, each with its own looks, editor, **Opacity** and **Shown**; **Clear layer** starts one again |
| **Hair colour** | 11 colours (platinum to electric blue) or any colour (+) |
| **Lips** | lipstick, with **Gloss** |
| **Eyeshadow, Eyeliner, Blush, Brows** | each with its own colours and strength |
| **Smooth skin** | an edge-keeping blur on the skin only |
| **Beard colour, Stubble, Shave** | colour a beard or moustache, paint stubble in, shave stubble or a short beard off |
| **Editor** | brightness, contrast, saturation, warmth, vignette, sharpen |
| **AI looks** (AetherCanvas AI) | makeup: natural, glam, smoky eyes, winged liner, red, nude or berry lips, glow, soft glam, gothic, bridal, freckles, no makeup; hair length: bangs, curly, straight, ponytail, bun, braids; full beard, goatee, moustache, clean shaven: sent to AetherCanvas AI when it's running (a minute or two each); **Back to the original photo** undoes them |

Every control answers at once: the preview is worked on a smaller copy, and Save paints the full-size photo.

## How it works

1. **Read the face, once.** A SegFormer face parser ([jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing),
   int8 ONNX) labels every pixel: hair, skin, nose, lips, brows, eyes, ears, neck, clothes. MediaPipe Face Mesh
   finds the face's 468 points (for the eyelids, cheekbones and the beard area).
2. **Paint in LAB.** Each look is a colour blended into its region: the region's brightness moves part of the way
   to the colour's, its tint all the way, its own fine variation stays. Gloss brightens where the lips already
   catch the light.
3. **Facial hair.** The beard area is the jaw from cheek to cheek up to the nostrils, without the lips; the beard is
   whatever in it is clearly darker than the face's own skin. Shave puts the face's own skin tone (a normalised
   blur of the skin around it) back over it; stubble is an even shade with a fine grain.
4. **Copied makeup.** DMT (Disentangled Makeup Transfer, a GAN, converted to ONNX: `models/dmt.onnx`, 48 MB) puts the
   reference's makeup on a 256-pixel copy of the face; only its colour change comes back, onto the full-size photo, on
   the lips, eyes, brows and cheeks. Lids and brows only darken, and the skin keeps its own lightness: the model
   learned from pale "after" photos, so its foundation would lighten skin, which is a skin-tone change, not makeup.
5. **Two layers.** Layer 2 is painted onto Layer 1's result with the same masks, each with its editor and opacity.

`cosmetics/adjust.py` (the editor) is the same file in every Changeling.

## Licenses

The code is MIT; the makeup transfer model comes from an MIT-licensed Space (huggingface.co/spaces/rashid996958/makeup-transfer). The face parser was trained on CelebAMask-HQ, which is for non-commercial research; MediaPipe is
Apache 2.0.

---

<br>
<div align="center">
© Cursed Entertainment 2026
</div>
<br>
<div align="center">
<a href="https://cursed-entertainment.itch.io/" target="_blank">
    <img src="https://raw.githubusercontent.com/CursedPrograms/cursedentertainment/main/images/logos/logo-wide-grey.png"
        alt="CursedEntertainment Logo" style="width:250px;">
</a>
</div>
<br>
<div align="center">
  <a href="https://github.com/SynthWomb" target="_blank">
    <img src="https://raw.githubusercontent.com/SynthWomb/synth.womb/main/logos/synthwomb07.png" alt="SynthWomb" style="width:200px;"/>
  </a>
</div>
