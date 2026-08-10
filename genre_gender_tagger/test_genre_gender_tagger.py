"""Unittest suite for genre_gender_tagger (stdlib only — no pytest needed).

Run from anywhere:

    python genre_gender_tagger/test_genre_gender_tagger.py

or from inside the tagger directory:

    python -m unittest test_genre_gender_tagger -v

Model-backed tests (load backend / classify) skip automatically when the
gender ONNX models are not present; nothing is downloaded by the tests.
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

TAGGER_DIR = Path(__file__).resolve().parent
if str(TAGGER_DIR) not in sys.path:
    sys.path.insert(0, str(TAGGER_DIR))

import numpy as np  # noqa: E402  (imported by the tagger at module level too)

import genre_gender_tagger as gg  # noqa: E402


def _quiet(_msg=None):
    pass


def _make_sine_wav(path, seconds=3.0, sample_rate=16000, freq=220.0):
    """Write a short mono 16-bit sine WAV (valid audio for feature extraction)."""
    n = int(sample_rate * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    x = (0.3 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((x * 32767.0).astype(np.int16).tobytes())


def _gender_models_present():
    d = gg._writable_gender_model_dir()
    return (
        (d / gg.GENDER_EFFNET_ONNX_NAME).is_file()
        and (d / gg.GENDER_HEAD_ONNX_NAME).is_file()
    )


class ImportTests(unittest.TestCase):
    def test_import_does_not_run_cli(self):
        """Importing in a fresh interpreter must print nothing and exit fast."""
        code = (
            "import sys, time\n"
            "sys.path.insert(0, {dir!r})\n"
            "t0 = time.time()\n"
            "import genre_gender_tagger\n"
            "print(time.time() - t0)\n"
        ).format(dir=str(TAGGER_DIR))
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # Only the elapsed-seconds line may be printed — no CLI banner/startup.
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 1, proc.stdout)
        self.assertLess(float(lines[0]), 60.0, "import too slow")

    def test_main_is_not_auto_invoked(self):
        self.assertTrue(callable(gg.main))


class ConstantTests(unittest.TestCase):
    def test_core_constants_resolved(self):
        self.assertEqual(gg.SAMPLE_RATE, 16000)
        self.assertEqual(gg.CLIP_LENGTH, 30)
        self.assertEqual(gg.NUMBER_OF_CLIPS, 3)
        self.assertEqual(gg.GENDER_LABELS, ("female", "male"))
        self.assertGreater(gg.GENDER_FRAME_SIZE, 0)
        self.assertGreater(gg.GENDER_N_MELS, 0)

    def test_file_discovery_constants(self):
        self.assertIn(".wav", gg.AUDIO_EXTENSIONS)
        self.assertIn(".flac", gg.AUDIO_EXTENSIONS)
        self.assertIn(".mp3", gg.AUDIO_EXTENSIONS)
        self.assertIn(".m4a", gg.AUDIO_EXTENSIONS)
        self.assertIsInstance(gg.INCLUDE_SUBFOLDERS, bool)

    def test_metadata_mappings(self):
        self.assertIn("genre", gg._MP4_STD)
        self.assertIn("comment", gg._MP4_STD)
        self.assertIn("style", gg._MP4_FREEFORM)
        self.assertIn("gender", gg._MP4_FREEFORM)
        self.assertIn("reverb", gg._MP4_FREEFORM)

    def test_mutagen_classes_imported(self):
        import mutagen.id3
        import mutagen.mp4
        import mutagen.wave

        self.assertIs(gg.ID3, mutagen.id3.ID3)
        self.assertIs(gg.MP3, mutagen.mp3.MP3)
        self.assertIs(gg.MP4, mutagen.mp4.MP4)
        self.assertIs(gg.WAVE, mutagen.wave.WAVE)

    def test_runtime_defaults_exist(self):
        self.assertIsNone(gg.torch)
        self.assertIsInstance(gg.ORT_CUDA_DETAIL, str)


class ModelTests(unittest.TestCase):
    def test_model_dir_resolution(self):
        d = gg._writable_gender_model_dir()
        self.assertTrue(d.is_dir(), f"model dir missing: {d}")
        # all tagger models live in one root models/ folder
        for name in (
            gg.GENDER_EFFNET_ONNX_NAME,
            gg.GENDER_HEAD_ONNX_NAME,
            "maest_discogs519.onnx",
        ):
            if name:
                self.assertTrue((d / name).is_file(), f"missing {d / name}")

    def test_ensure_gender_onnx_models(self):
        if not _gender_models_present():
            self.skipTest("gender ONNX models not present")
        result = gg.ensure_gender_onnx_models(status=_quiet)
        self.assertEqual(len(result), 2)
        for p in result:
            self.assertTrue(Path(p).is_file())

    def test_load_gender_ort_backend(self):
        if not _gender_models_present():
            self.skipTest("gender ONNX models not present")
        backend = gg.load_gender_ort_backend(status=_quiet)
        self.assertTrue(hasattr(backend, "predict_batch"))
        chunk = np.zeros((1, 128, 96), dtype=np.float32)
        probs = backend.predict_batch(chunk)
        self.assertEqual(probs.shape, (1, 2))
        self.assertEqual(probs.dtype, np.float32)


class ClassifyTests(unittest.TestCase):
    def test_classify_gender_file(self):
        if not _gender_models_present():
            self.skipTest("gender ONNX models not present")
        backend = gg.load_gender_ort_backend(status=_quiet)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "tone.wav"
            _make_sine_wav(wav)
            result = gg.classify_gender_file(str(wav), backend)
        for key in ("gender", "confidence", "female", "male"):
            self.assertIn(key, result)
        self.assertIn(result["gender"], ("female", "male"))
        self.assertAlmostEqual(
            result["female"] + result["male"], 1.0, places=3
        )
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


class TagTests(unittest.TestCase):
    def setUp(self):
        self._save = {
            "TAG_WRITE_MODE": gg.TAG_WRITE_MODE,
            "GENDER_TAG_FIELD": gg.GENDER_TAG_FIELD,
            "REVERB_TAG_MODE": gg.REVERB_TAG_MODE,
            "OVERWRITE_TAGS": gg.OVERWRITE_TAGS,
        }

    def tearDown(self):
        for name, value in self._save.items():
            setattr(gg, name, value)

    def test_metadata_roundtrip_wav(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "t.wav"
            _make_sine_wav(wav)

            self.assertFalse(gg.has_genre_tags(str(wav)))
            self.assertFalse(gg.has_gender_tags(str(wav)))

            # combined genre/style -> "Rock/Pop" in TCON
            gg.TAG_WRITE_MODE = "combined"
            self.assertTrue(gg.write_metadata(str(wav), "Rock", "Pop"))
            self.assertEqual(gg.read_tag_field(str(wav), "genre"), "Rock/Pop")
            self.assertTrue(gg.has_genre_tags(str(wav)))

            # split mode -> genre and style separate
            gg.TAG_WRITE_MODE = "split"
            self.assertTrue(gg.write_metadata(str(wav), "Rock", "Pop"))
            self.assertEqual(gg.read_tag_field(str(wav), "genre"), "Rock")
            self.assertEqual(gg.read_tag_field(str(wav), "style"), "Pop")

            # gender + reverb combined into comment field
            gg.GENDER_TAG_FIELD = "comment"
            gg.REVERB_TAG_MODE = "combined"
            self.assertTrue(
                gg.write_gender_metadata(str(wav), "female", "medium")
            )
            self.assertEqual(
                gg.read_tag_field(str(wav), "comment"), "female/medium"
            )
            self.assertTrue(gg.has_gender_tags(str(wav)))


class GenreTests(unittest.TestCase):
    """Genre-side pipeline via the MAEST ONNX model (pure numpy, no torch)."""

    @classmethod
    def setUpClass(cls):
        try:
            from maest_onnx import try_load_maest_onnx

            cls.fe, cls.model = try_load_maest_onnx(status=_quiet) or (None, None)
        except Exception:
            cls.fe, cls.model = None, None

    def setUp(self):
        if self.fe is None or self.model is None:
            self.skipTest("maest_discogs519.onnx / id2label not present")

    def test_create_clips_shape(self):
        audio = np.zeros(gg.SAMPLE_RATE * gg.CLIP_LENGTH * 2, dtype=np.float32)
        clips = gg.create_clips(audio)
        self.assertEqual(len(clips), gg.NUMBER_OF_CLIPS)
        for clip in clips:
            self.assertEqual(len(clip), gg.SAMPLE_RATE * gg.CLIP_LENGTH)

    def test_load_and_extract_returns_features(self):
        # longer than one clip -> multi-clip slicing path
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "t.wav"
            _make_sine_wav(wav, seconds=gg.CLIP_LENGTH + 5)
            index, inputs = gg.load_and_extract(
                (0, str(wav)),
                feature_extractor=self.fe,
                use_maest_onnx=True,
                torch_mod=None,
            )
        self.assertEqual(index, 0)
        iv = inputs["input_values"]
        self.assertEqual(iv.shape[0], gg.NUMBER_OF_CLIPS)
        self.assertEqual(iv.dtype, np.float32)

    def test_run_gpu_batch_onnx(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "t.wav"
            _make_sine_wav(wav, seconds=gg.CLIP_LENGTH + 5)
            index, inputs = gg.load_and_extract(
                (0, str(wav)),
                feature_extractor=self.fe,
                use_maest_onnx=True,
                torch_mod=None,
            )
            scores, mapping = gg.run_gpu_batch(
                [(index, inputs)],
                model=self.model,
                device="cpu",
                is_gpu=False,
                model_dtype=np.float32,
                use_maest_onnx=True,
            )
        self.assertEqual(len(mapping), gg.NUMBER_OF_CLIPS)
        self.assertEqual(scores.shape[0], gg.NUMBER_OF_CLIPS)
        self.assertTrue(
            np.allclose(scores.sum(axis=1), 1.0, atol=1e-4),
            "scores must be probabilities",
        )

    def test_classify_one_file_onnx(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "t.wav"
            _make_sine_wav(wav, seconds=gg.CLIP_LENGTH + 5)
            scores, n_clips = gg.classify_one_file(
                str(wav),
                model=self.model,
                feature_extractor=self.fe,
                use_maest_onnx=True,
                device="cpu",
                is_gpu=False,
                model_dtype=np.float32,
            )
        self.assertEqual(n_clips, gg.NUMBER_OF_CLIPS)
        self.assertEqual(scores.shape[0], n_clips)
        label = self.model.config.id2label[int(np.argmax(scores[0]))]
        self.assertIsInstance(label, str)
        self.assertTrue(label)

    def test_log_genre_from_scores(self):
        idx = next(
            i
            for i in range(self.model.config.num_labels)
            if "---" in self.model.config.id2label[i]
        )
        scores = np.zeros(self.model.config.num_labels, dtype=np.float32)
        scores[idx] = 1.0
        with contextlib.redirect_stdout(io.StringIO()):
            gg._log_genre_from_scores(
                0, scores, model=self.model, files=["fake.wav"]
            )

    def test_store_gpu_scores_explicit_storage(self):
        storage = {}
        scores = np.array([[0.5, 0.5], [0.4, 0.6]], dtype=np.float32)
        gg._store_gpu_scores(
            scores, [7, 7], progress=None, score_storage=storage
        )
        self.assertEqual(len(storage[7]), 2)


class FileIterationTests(unittest.TestCase):
    def setUp(self):
        self._save = {"INCLUDE_SUBFOLDERS": gg.INCLUDE_SUBFOLDERS}
        self._files_from = os.environ.pop("GG_FILES_FROM", None)

    def tearDown(self):
        for name, value in self._save.items():
            setattr(gg, name, value)
        if self._files_from is not None:
            os.environ["GG_FILES_FROM"] = self._files_from

    def test_iter_audio_files_recursive_and_flat(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.wav").write_bytes(b"x")
            (root / "b.mp3").write_bytes(b"x")
            (root / "c.txt").write_bytes(b"x")
            (root / "d.m4a").write_bytes(b"x")
            sub = root / "sub"
            sub.mkdir()
            (sub / "e.flac").write_bytes(b"x")

            gg.INCLUDE_SUBFOLDERS = True
            names = sorted(p.name for p in gg.iter_audio_files(td))
            self.assertEqual(names, ["a.wav", "b.mp3", "d.m4a", "e.flac"])

            gg.INCLUDE_SUBFOLDERS = False
            names = sorted(p.name for p in gg.iter_audio_files(td))
            self.assertEqual(names, ["a.wav", "b.mp3", "d.m4a"])

    def test_list_audio_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.wav").write_bytes(b"x")
            (root / "b.flac").write_bytes(b"x")
            gg.INCLUDE_SUBFOLDERS = True
            with contextlib.redirect_stdout(io.StringIO()):
                files = gg.list_audio_files(td)
            self.assertEqual(sorted(Path(f).name for f in files), ["a.wav", "b.flac"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
