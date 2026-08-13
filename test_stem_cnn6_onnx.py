"""Unittest suite for the Classify (Stem CNN6 ONNX) runner's silence handling.

Run from anywhere:

    python test_stem_cnn6_onnx.py

or:

    python -m unittest test_stem_cnn6_onnx -v

Model-backed tests skip automatically when ``models/stem_cnn6.onnx`` is not
present; nothing is downloaded by the tests.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import classify_backend as cb  # noqa: E402
from stem_cnn6_onnx import (  # noqa: E402
    FINE_CLASSES,
    N_FINE,
    SOURCES,
    StemCnn6OnnxModel,
    resolve_stem_cnn6_onnx,
)

SAMPLE_RATE = 32000
CLIP_SAMPLES = 320000
MODE = cb.STEM_MODES["4 (bass/drums/other/vocals)"]


def _vocals_only_run(chunks):
    """Stub _run_onnx_batch: every chunk is confidently VOCALS."""
    out = np.zeros((len(chunks), N_FINE), dtype=np.float32)
    out[:, FINE_CLASSES.index("VOCALS")] = 1.0
    return out


def _fake_model(predict_fn):
    """A StemCnn6OnnxModel shell without a real session; only _run_onnx_batch is stubbed."""
    model = object.__new__(StemCnn6OnnxModel)
    model._run_onnx_batch = predict_fn  # type: ignore[attr-defined]
    return model


def _stereo(mono):
    return np.stack([mono, mono]).astype(np.float32)


def _tone(seconds=10.0, freq=440.0):
    n = int(SAMPLE_RATE * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    return (0.3 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _energies(out, item=0, t_len=None):
    """Per-synthetic-stem RMS, mirroring classify_backend's classify_batch."""
    if t_len is None:
        t_len = out.shape[-1]
    return {
        name: float(np.sqrt(np.mean(out[item, k, :, :t_len] ** 2) + 1e-12))
        for k, name in enumerate(SOURCES)
    }


class SilentSkipTests(unittest.TestCase):
    """Stubbed model — the exact regression: silent stem -> skip + None label."""

    def test_silent_stem_is_skipped_with_no_fine_label(self):
        model = _fake_model(_vocals_only_run)
        silent = _stereo(np.zeros(CLIP_SAMPLES, dtype=np.float32))
        out = model.separate_numpy(silent)
        # The old code fell back to a plain mean here and returned an arbitrary
        # class (e.g. VOCALS); now it must yield no fine label at all.
        self.assertEqual(model._last_fine_labels, [None])
        label, _top, _share, _margin, _reason = cb.classify_to_category(
            _energies(out), MODE, 0.4, 0.2
        )
        self.assertEqual(label, "skip")

    def test_real_stem_is_classified(self):
        model = _fake_model(_vocals_only_run)
        out = model.separate_numpy(_stereo(_tone()))
        self.assertEqual(model._last_fine_labels, ["VOCALS"])
        label, _top, top_share, _margin, _reason = cb.classify_to_category(
            _energies(out), MODE, 0.4, 0.2
        )
        self.assertEqual(label, "vocals")
        self.assertGreater(top_share, 0.9)


class ModelSilentTests(unittest.TestCase):
    """End-to-end with the real ONNX model (CPU). Skips when the model is absent."""

    @classmethod
    def setUpClass(cls):
        cls.onnx = resolve_stem_cnn6_onnx()
        cls.model = None
        cls._skip_reason = ""
        if cls.onnx is None:
            return
        try:
            cls.model = StemCnn6OnnxModel(cls.onnx, prefer_gpu=False)
        except Exception as exc:  # noqa: BLE001 — skip instead of fail
            cls.onnx = None
            cls._skip_reason = str(exc)

    def setUp(self):
        if self.onnx is None:
            if self._skip_reason:
                self.skipTest(f"backend load failed: {self._skip_reason}")
            self.skipTest("stem_cnn6.onnx not present")

    def test_silent_stem_is_skipped_with_no_fine_label(self):
        silent = _stereo(np.zeros(CLIP_SAMPLES, dtype=np.float32))
        out = self.model.separate_numpy(silent)
        self.assertEqual(self.model._last_fine_labels, [None])
        label, _top, _share, _margin, _reason = cb.classify_to_category(
            _energies(out), MODE, 0.4, 0.2
        )
        self.assertEqual(label, "skip")

    def test_real_stem_gets_a_fine_label(self):
        out = self.model.separate_numpy(_stereo(_tone()))
        self.assertEqual(len(self.model._last_fine_labels), 1)
        self.assertIn(
            self.model._last_fine_labels[0], FINE_CLASSES,
            "a non-silent stem must produce a real fine label (not None)",
        )
        # A non-silent stem's synthetic output must carry dominant energy in
        # exactly one stem (vs. silent stems whose energies are all equal).
        energies = _energies(out)
        ranked = sorted(energies.values(), reverse=True)
        self.assertGreater(ranked[0], ranked[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
