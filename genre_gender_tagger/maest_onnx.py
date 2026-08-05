"""MAEST Discogs519 ONNX drop-in (feature extractor + classifier).

Duck-types the HF pair used by genre_gender_tagger:
  - feature_extractor(clips, sampling_rate=..., return_tensors=\"pt\"|\"np\")
  - model(input_values=...) -> object with .logits (numpy array; torch only if available)
  - model.config.id2label[i] -> \"Genre---Style\"
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from maest_fe_np import SAMPLE_RATE as MAEST_SR
from maest_fe_np import maest_input_values

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
ONNX_NAME = "maest_discogs519.onnx"
LABELS_NAME = "maest_discogs519.id2label.json"


def resolve_maest_onnx(models_dir: Path | None = None) -> Path | None:
    root = models_dir or MODELS
    candidates = [
        root / ONNX_NAME,
        HERE.parent / "_onnx_spike" / "onnx_out" / ONNX_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_maest_labels(models_dir: Path | None = None) -> Path | None:
    root = models_dir or MODELS
    candidates = [
        root / LABELS_NAME,
        HERE.parent / "_onnx_spike" / "onnx_out" / LABELS_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_id2label(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data["id2label"]
    return {i: str(lab) for i, lab in enumerate(labels)}


class MaestFeatureExtractorNp:
    """HF AutoFeatureExtractor stand-in (numpy mel + Discogs normalize)."""

    sampling_rate = MAEST_SR

    def __call__(
        self,
        raw_speech,
        sampling_rate: int | None = None,
        return_tensors: str | None = None,
        **_kwargs,
    ):
        if sampling_rate is not None and int(sampling_rate) != MAEST_SR:
            raise ValueError(
                f"MAEST expects {MAEST_SR} Hz, got {sampling_rate}"
            )
        iv = maest_input_values(raw_speech)
        if return_tensors == "pt":
            try:
                import torch
            except ImportError:
                return {"input_values": iv}
            return {"input_values": torch.from_numpy(iv)}
        if return_tensors == "np" or return_tensors is None:
            return {"input_values": iv}
        raise ValueError(f"unsupported return_tensors={return_tensors!r}")


class MaestModelOnnx:
    """HF AutoModelForAudioClassification stand-in via onnxruntime."""

    def __init__(self, onnx_path: Path, id2label: dict[int, str], device: str = ""):
        try:
            from ort_util import create_ort_session
        except ImportError:
            import sys

            root = Path(__file__).resolve().parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from ort_util import create_ort_session

        self.session = create_ort_session(onnx_path, device=device)
        self.config = SimpleNamespace(id2label=id2label, num_labels=len(id2label))
        try:
            self.device = self.session.get_providers()[0]
        except Exception:
            self.device = device or "onnx"

    def eval(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def __call__(self, input_values=None, **_kwargs):
        if input_values is None:
            raise ValueError("input_values required")
        if hasattr(input_values, "detach"):
            arr = input_values.detach().float().cpu().numpy()
        else:
            arr = np.asarray(input_values, dtype=np.float32)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        logits = self.session.run(["logits"], {"input_values": arr})[0]
        logits = np.asarray(logits, dtype=np.float32)
        # Prefer numpy (frozen ONNX builds have no torch). Torch wrapper
        # only when available for HF-compat callers.
        try:
            import torch
        except ImportError:
            return SimpleNamespace(logits=logits)
        return SimpleNamespace(logits=torch.from_numpy(logits))


def try_load_maest_onnx(
    *,
    device: str = "",
    models_dir: Path | None = None,
    status=print,
) -> tuple[MaestFeatureExtractorNp, MaestModelOnnx] | None:
    """Return (FE, model) if STEM_ONNX and assets exist; else None."""
    if os.environ.get("STEM_ONNX", "1").strip() == "0":
        return None
    onnx_path = resolve_maest_onnx(models_dir)
    labels_path = resolve_maest_labels(models_dir)
    if onnx_path is None or labels_path is None:
        return None
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return None
    id2label = load_id2label(labels_path)
    status(f"  MAEST onnxruntime: {onnx_path.name}")
    fe = MaestFeatureExtractorNp()
    model = MaestModelOnnx(onnx_path, id2label, device=device)
    return fe, model
