"""transfer_models.py - the ways to copy makeup from a photo, for the window's Model switch.

    Warp (exact)         mesh_transfer.py: the reference's face warped onto yours, triangle by triangle. Patterns
                         land exactly where they were; a fifth of a second; nothing to download. The default.
    DMT (GAN)            transfer.py: Disentangled Makeup Transfer, a GAN at 256 x 256: colours and style, not
                         drawn patterns. Only if models/dmt.onnx is on this PC.
    Stable-Makeup (AI)   Stable-Makeup (Zhang et al. 2024): a diffusion model that redraws the makeup onto your
                         face. About 8 GB of models in F:\\AIModels\\StableMakeup (stable_makeup_models.py fetches
                         them); seconds a photo on the graphics card.

Each one: available() -> (True, "") or (False, why not), and make() -> something with
delta(face, reference_rgb, paint, parser) -> (LAB change the size of the photo, the result as RGB).
"""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Warp:
    name = "Warp (exact)"

    @staticmethod
    def available():
        return True, ""

    @staticmethod
    def make():
        from .mesh_transfer import MeshTransfer
        m = MeshTransfer()

        class Run:
            def delta(self, face, ref, paint, parser):
                return m.delta(face, ref, paint=paint)
        return Run()


class _DMT:
    name = "DMT (GAN)"

    @staticmethod
    def available():
        from .transfer import MODEL
        if os.path.exists(MODEL):
            return True, ""
        return False, "DMT's model (models/dmt.onnx) isn't on this PC."

    @staticmethod
    def make():
        from .transfer import MakeupTransfer, read_reference
        m = MakeupTransfer()

        class Run:
            def delta(self, face, ref, paint, parser):
                return m.delta(face, read_reference(parser, ref), paint=paint)
        return Run()


class _StableMakeup:
    name = "Stable-Makeup (AI)"

    @staticmethod
    def available():
        from . import stable_makeup_models as sm
        missing = sm.missing()
        if missing:
            have = len(sm.FILES) - len(missing)
            return False, (f"Stable-Makeup's models are still coming: {have} of {len(sm.FILES)} files in "
                           f"{sm.ROOT} (python -m cosmetics.stable_makeup_models fetches the rest).")
        if not os.path.exists(os.path.join(HERE, "cosmetics", "stable_makeup.py")):
            return False, "Stable-Makeup's models are here; its code isn't in this version yet."
        return True, ""

    @staticmethod
    def make():
        from .stable_makeup import StableMakeup     # noqa: F401 - added once the models are here
        m = StableMakeup()

        class Run:
            def delta(self, face, ref, paint, parser):
                return m.delta(face, ref, paint=paint)
        return Run()


MODELS = {m.name: m for m in (_Warp, _DMT, _StableMakeup)}
DEFAULT = _Warp.name
