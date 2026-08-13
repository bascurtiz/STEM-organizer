"""Unittest suite for instrument_tagger (stdlib only — no pytest needed).

Run from anywhere:

    python instrument_tagger/test_instrument_tagger.py

or from inside the tagger directory:

    python -m unittest test_instrument_tagger -v

Model-backed tests (load backend / classify) skip automatically when the
Stem CNN6 ONNX model is not present; nothing is downloaded by the tests.
"""

import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

TAGGER_DIR = Path(__file__).resolve().parent
if str(TAGGER_DIR) not in sys.path:
    sys.path.insert(0, str(TAGGER_DIR))

import instrument_tagger as it  # noqa: E402


def _quiet(_msg=None):
    pass


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 32000) -> None:
    """Write a mono 16-bit PCM WAV from float32 samples in [-1, 1]."""
    x = np.asarray(samples, dtype=np.float32)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())


def _make_silent_wav(path: Path, seconds: float = 30.0, sample_rate: int = 32000) -> None:
    """A fully-silent stem (no signal anywhere)."""
    _write_wav(path, np.zeros(int(sample_rate * seconds), dtype=np.float32), sample_rate)


def _make_silent_intro_wav(
    path: Path,
    silence_seconds: float = 15.0,
    tone_seconds: float = 15.0,
    sample_rate: int = 32000,
    freq: float = 440.0,
) -> None:
    """A stem with a long silent intro, then a real tone (content later)."""
    n_sil = int(sample_rate * silence_seconds)
    n_tone = int(sample_rate * tone_seconds)
    t = np.linspace(0.0, tone_seconds, n_tone, endpoint=False)
    tone = (0.3 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    _write_wav(path, np.concatenate([np.zeros(n_sil, dtype=np.float32), tone]), sample_rate)


def _onnx_path() -> Path | None:
    try:
        p = it._resolve_onnx()
    except Exception:
        return None
    return p if (p is not None and p.is_file()) else None


def _fake_backend(predict_fn):
    """A StemCnn6BackendOnnx shell without a real session; only _run_pooled is stubbed."""
    backend = object.__new__(it.StemCnn6BackendOnnx)
    backend._run_pooled = predict_fn  # type: ignore[attr-defined]
    return backend


class ImportTests(unittest.TestCase):
    def test_import_does_not_run_cli(self):
        """Importing in a fresh interpreter must print nothing and exit fast."""
        code = (
            "import sys, time\n"
            "sys.path.insert(0, {dir!r})\n"
            "t0 = time.time()\n"
            "import instrument_tagger\n"
            "print(time.time() - t0)\n"
        ).format(dir=str(TAGGER_DIR))
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 1, proc.stdout)
        self.assertLess(float(lines[0]), 60.0, "import too slow")

    def test_main_is_not_auto_invoked(self):
        self.assertTrue(callable(it.main))


class ConstantTests(unittest.TestCase):
    def test_core_constants_resolved(self):
        self.assertEqual(it.SAMPLE_RATE, 32000)
        self.assertEqual(it.CLIP_SAMPLES, 320000)
        self.assertEqual(it.MAX_AUDIO_SECONDS, 10.0)
        self.assertEqual(len(it.STEM_CLASSES), 11)
        self.assertEqual(it.N_CLASSES, 11)
        self.assertGreater(it.SILENCE_PEAK_FLOOR, 0.0)
        self.assertGreater(it.SILENCE_RMS_FLOOR, 0.0)
        self.assertGreater(it.CHUNK_OVERLAP, 0.0)
        self.assertLess(it.CHUNK_OVERLAP, 1.0)

    def test_silent_floors_detect_digital_silence(self):
        # Digital silence (all zeros) must be below both floors.
        self.assertLess(0.0, it.SILENCE_PEAK_FLOOR)
        self.assertLess(0.0, it.SILENCE_RMS_FLOOR)


class LoadTests(unittest.TestCase):
    """Whole-file silence detection in load_mono_32k (no model needed)."""

    def test_fully_silent_wav_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "silent.wav"
            _make_silent_wav(wav, seconds=30.0)
            audio = it.load_mono_32k(wav)
        self.assertEqual(audio.size, 0, "fully-silent stem must be skipped (empty)")

    def test_silent_intro_wav_returns_full_signal(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "intro.wav"
            _make_silent_intro_wav(wav, silence_seconds=15.0, tone_seconds=15.0)
            audio = it.load_mono_32k(wav)
        # Full 30 s must be returned (not truncated to the first 10 s), and the
        # later half must carry the tone — so the chunker can find real content.
        self.assertGreater(audio.size, it.CLIP_SAMPLES)
        self.assertEqual(audio.shape[0], 30 * it.SAMPLE_RATE)
        tail_peak = float(np.max(np.abs(audio[int(audio.shape[0] * 0.6) :])))
        self.assertGreater(tail_peak, 0.1, "tone should live after the silent intro")


class ChunkTests(unittest.TestCase):
    """Chunking + RMS weighting (the core of the silent-intro fix)."""

    def _chunks(self, audio):
        backend = object.__new__(it.StemCnn6BackendOnnx)
        return backend._chunk_mono(audio)  # noqa: SLF001

    def test_chunk_mono_empty(self):
        chunks, weights = self._chunks(None)
        self.assertEqual(chunks, [])
        self.assertEqual(weights, [])
        chunks, weights = self._chunks(np.zeros(0, dtype=np.float32))
        self.assertEqual(chunks, [])
        self.assertEqual(weights, [])

    def test_chunk_mono_silent_weights_all_zero(self):
        silent = np.zeros(it.CLIP_SAMPLES * 3, dtype=np.float32)
        chunks, weights = self._chunks(silent)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(sum(weights), 0.0, "every silent chunk must be weight 0")

    def test_chunk_mono_silent_intro_weights_later_chunks(self):
        audio = np.zeros(it.SAMPLE_RATE * 30, dtype=np.float32)
        t = np.linspace(0.0, 15.0, it.SAMPLE_RATE * 15, endpoint=False)
        audio[it.SAMPLE_RATE * 15 :] = 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
        chunks, weights = self._chunks(audio)
        self.assertGreater(len(chunks), 2)
        # Early chunks (silent intro) are weight 0; later chunks (tone) are > 0.
        self.assertEqual(sum(weights[: len(weights) // 2]), 0.0)
        self.assertGreater(sum(weights[len(weights) // 2 :]), 0.0)


class PredictBatchTests(unittest.TestCase):
    """predict_batch weighted averaging, stubbed so no model is needed."""

    def _drums_only_run(self, chunks):
        out = np.zeros((len(chunks), it.N_CLASSES), dtype=np.float32)
        out[:, it.STEM_CLASSES.index("DRUMS")] = 1.0
        return out

    def test_silent_input_is_zeroed(self):
        backend = _fake_backend(self._drums_only_run)
        probs = backend.predict_batch([np.zeros(it.CLIP_SAMPLES * 3, dtype=np.float32)])
        self.assertEqual(probs.shape, (1, it.N_CLASSES))
        self.assertEqual(float(probs.max()), 0.0)
        result = it.probs_to_result(probs[0])
        self.assertTrue(result["silent"])
        self.assertEqual(result["label"], "")

    def test_silent_intro_input_is_classified(self):
        backend = _fake_backend(self._drums_only_run)
        audio = np.zeros(it.SAMPLE_RATE * 30, dtype=np.float32)
        t = np.linspace(0.0, 15.0, it.SAMPLE_RATE * 15, endpoint=False)
        audio[it.SAMPLE_RATE * 15 :] = 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
        probs = backend.predict_batch([audio])
        result = it.probs_to_result(probs[0])
        self.assertFalse(result["silent"])
        self.assertEqual(result["label"], "DRUMS")
        self.assertGreater(result["score"], 0.0)


class ProbsToResultTests(unittest.TestCase):
    def test_zero_probs_are_silent(self):
        result = it.probs_to_result(np.zeros(it.N_CLASSES, dtype=np.float32))
        self.assertTrue(result["silent"])
        self.assertEqual(result["label"], "")
        self.assertEqual(result["score"], 0.0)

    def test_one_hot_probs_produce_label(self):
        probs = np.zeros(it.N_CLASSES, dtype=np.float32)
        idx = it.STEM_CLASSES.index("VOCALS")
        probs[idx] = 1.0
        result = it.probs_to_result(probs)
        self.assertFalse(result["silent"])
        self.assertEqual(result["label"], "VOCALS")
        self.assertAlmostEqual(result["score"], 1.0)


class ModelClassifyTests(unittest.TestCase):
    """End-to-end over synthetic stems with the real ONNX model (CPU)."""

    @classmethod
    def setUpClass(cls):
        cls.onnx = _onnx_path()
        if cls.onnx is None:
            return
        try:
            cls.backend = it.StemCnn6BackendOnnx(cls.onnx, device="cpu")
        except Exception as exc:  # noqa: BLE001 — skip instead of fail
            cls.onnx = None
            cls.backend = None
            cls._skip_reason = str(exc)

    def setUp(self):
        if self.onnx is None:
            self.skipTest(
                "stem_cnn6.onnx not present"
                if not getattr(self, "_skip_reason", "")
                else f"backend load failed: {self._skip_reason}"
            )

    def test_classify_silent_stem_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "silent.wav"
            _make_silent_wav(wav, seconds=30.0)
            result = it.classify_file(str(wav), self.backend)
        self.assertTrue(result["silent"])
        self.assertEqual(result["label"], "")
        self.assertEqual(result["score"], 0.0)

    def test_classify_silent_intro_stem_is_classified(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "intro.wav"
            _make_silent_intro_wav(wav, silence_seconds=15.0, tone_seconds=15.0)
            result = it.classify_file(str(wav), self.backend)
        self.assertFalse(result["silent"])
        self.assertIn(result["label"], it.STEM_CLASSES)
        self.assertGreater(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
