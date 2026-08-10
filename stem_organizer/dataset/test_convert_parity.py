"""Parity tests: ffmpeg fast path vs numpy pipeline in convert_flac.

Covers:
- Integer-PCM no-op (s16, s24): bit-exact between ffmpeg and soundfile.
- Resample (48k -> 44.1k): sr/length/format correct, RMSE < tolerance.
- Mono -> stereo: bit-exact duplication.
- >2ch source: falls back to numpy.
- Float / lossy sources: fallback to numpy, dither applied, correct output.
- Gain-reduction path (float overs): numpy fallback, peak <= ceiling.
"""
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stem_organizer.dataset.convert_flac import (
    _FfmpegUnsupported,
    _process_file_ffmpeg,
    process_file,
)


def _mk_s16_wav(path, sr=44100, seed=0, seconds=2, peak=0.9, channels=2):
    """Write an s16 WAV fixture (range -32768..32767, scaled to peak)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    sig = (0.6 * np.sin(2 * np.pi * 440 * t)
           + 0.3 * np.sin(2 * np.pi * 880 * t)
           + 0.05 * rng.normal(size=len(t)))
    sig = sig * peak / max(abs(sig).max(), 1e-12)
    # stereo: same signal both channels + slight decorrelation
    stereo = np.stack([sig, sig + 0.01 * rng.uniform(-1, 1, len(sig))], axis=1)
    if channels == 1:
        stereo = stereo[:, :1]
    mono = stereo.mean(axis=1, keepdims=True) if channels == 1 else stereo
    sf.write(str(path), np.clip(mono, -1, 1), sr, subtype="PCM_16", format="WAV")


def _mk_s24_wav(path, sr=44100, channels=2):
    """Write a 24-bit WAV fixture."""
    rng = np.random.default_rng(7)
    t = np.arange(int(sr * 2)) / sr
    sig = 0.7 * np.sin(2 * np.pi * 330 * t) + 0.05 * rng.normal(size=len(t))
    if channels == 1:
        sig = sig[:, None]
    else:
        sig = np.stack([sig, sig + 0.01 * rng.uniform(-1, 1, len(sig))], axis=1)
    sf.write(str(path), sig, sr, subtype="PCM_24", format="WAV")


def _mk_float_wav(path, sr=44100, peak=1.2, channels=2):
    """Write a float WAV fixture (may exceed 0 dBFS for gain testing)."""
    rng = np.random.default_rng(13)
    t = np.arange(int(sr * 2)) / sr
    sig = (0.5 * np.sin(2 * np.pi * 220 * t)
           + 0.3 * np.sin(2 * np.pi * 660 * t)
           + 0.02 * rng.normal(size=len(t)))
    sig = sig * peak / max(abs(sig).max(), 1e-12)
    if channels == 2:
        sig = np.stack([sig, sig + 0.01 * rng.uniform(-1, 1, len(sig))], axis=1)
    else:
        sig = sig[:, None]
    sf.write(str(path), sig, sr, subtype="FLOAT", format="WAV")


class FfmpegPathTests(unittest.TestCase):
    """Tests that exercise the ffmpeg fast path (integer-PCM, <=2ch, no reduction)."""

    def test_s16_noop_bit_exact(self):
        """s16 44.1k -> same-sr FLAC: ffmpeg output is bit-exact with soundfile write."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "in.wav"
            _mk_s16_wav(src)
            ref = sf.read(str(src), dtype="int16")[0]
            dst = d / "out.flac"
            row = process_file(src, dst, headroom_db=1.0, dither=True,
                               target_samplerate=44100, target_channels=2)
            self.assertEqual(row["status"], "ok")
            got = sf.read(str(dst), dtype="int16")[0]
            maxdiff = int(np.abs(got.astype(np.int64) - ref.astype(np.int64)).max())
            self.assertEqual(maxdiff, 0, f"s16 no-op differs by {maxdiff} LSB")

    def test_s24_noop_bit_exact(self):
        """s24 44.1k -> same-sr FLAC: ffmpeg output is bit-exact with soundfile."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "in24.wav"
            _mk_s24_wav(src)
            ref = sf.read(str(src), dtype="float64")[0]
            dst = d / "out24.flac"
            row = process_file(src, dst, headroom_db=1.0, dither=True,
                               target_samplerate=44100, target_channels=2)
            self.assertEqual(row["status"], "ok")
            info = sf.info(str(dst))
            self.assertEqual(info.subtype, "PCM_24")
            self.assertEqual(info.samplerate, 44100)
            got = sf.read(str(dst), dtype="float64")[0]
            maxdiff = float(np.abs(got - ref).max())
            self.assertAlmostEqual(maxdiff, 0.0, places=7,
                                   msg="s24 no-op float diff")

    def test_s16_resample(self):
        """s16 48k -> 44.1k: ffmpeg resample is sr-correct and tolerably close."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "in48.wav"
            _mk_s16_wav(src, sr=48000)
            dst = d / "out44.flac"
            row = process_file(src, dst, headroom_db=1.0, dither=True,
                               target_samplerate=44100, target_channels=2)
            self.assertEqual(row["status"], "ok")
            info = sf.info(str(dst))
            self.assertEqual(info.samplerate, 44100)
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.subtype, "PCM_16")
            # Length: resample preserves duration, so frames ~ 2*44100
            self.assertAlmostEqual(info.frames, 88200, delta=20,
                                   msg="resampled frame count")
            # Compare with scipy resample_poly (the old path) via reference
            from scipy.signal import resample_poly
            g = math.gcd(48000, 44100)
            up, down = 44100 // g, 48000 // g
            raw, _ = sf.read(str(src), dtype="float64")
            ref = np.stack([resample_poly(raw[:, c], up, down) for c in range(raw.shape[1])], axis=1)
            got = sf.read(str(dst), dtype="float64")[0]
            m = min(len(got), len(ref))
            diff = got[:m] - ref[:m]
            rmse = np.sqrt(np.mean(diff ** 2))
            self.assertLess(rmse, 0.01, f"resample RMSE {rmse:.6f} (tolerance 0.01)")
            maxd = float(np.abs(diff).max())
            self.assertLess(maxd, 0.06, f"resample maxdiff {maxd:.4f} (tolerance 0.06)")

    def test_mono_to_stereo(self):
        """Mono s16 -> stereo FLAC: ffmpeg duplicates, bit-exact."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "mono.wav"
            _mk_s16_wav(src, channels=1)
            dst = d / "out_stereo.flac"
            row = process_file(src, dst, headroom_db=1.0, dither=True,
                               target_samplerate=44100, target_channels=2)
            self.assertEqual(row["status"], "ok")
            info = sf.info(str(dst))
            self.assertEqual(info.channels, 2)
            got = sf.read(str(dst), dtype="int16")[0]
            self.assertTrue(np.array_equal(got[:, 0], got[:, 1]),
                            "mono->stereo channels not identical")

    def test_stereo_to_mono(self):
        """Stereo s16 -> mono FLAC: ffmpeg mean-downmixes."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "stereo.wav"
            _mk_s16_wav(src, channels=2, seed=7)
            ref = sf.read(str(src), dtype="float64")[0].mean(axis=1)
            dst = d / "out_mono.flac"
            row = process_file(src, dst, headroom_db=1.0, dither=True,
                               target_samplerate=44100, target_channels=1)
            self.assertEqual(row["status"], "ok")
            info = sf.info(str(dst))
            self.assertEqual(info.channels, 1)
            got = sf.read(str(dst), dtype="float64")[0]
            got = got[:, 0] if got.ndim > 1 else got
            m = min(len(got), len(ref))
            maxdiff = float(np.abs(got[:m] - ref[:m]).max())
            # mean downmix differs at float->int rounding by <= 1 LSB
            self.assertLess(maxdiff, 5e-5, f"stereo->mono maxdiff {maxdiff:.6f}")

    def test_fast_path_rejects_float(self):
        """Direct ffmpeg path raises for float sources."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "f32.wav"
            _mk_float_wav(src, peak=0.9)
            with self.assertRaises(_FfmpegUnsupported):
                _process_file_ffmpeg(src, d / "x.flac", headroom_db=1.0,
                                     dither=True, target_samplerate=44100,
                                     target_channels=2)

    def test_fast_path_rejects_5ch(self):
        """Direct ffmpeg path raises for >2ch sources."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "x.wav"
            rng = np.random.default_rng(4)
            sf.write(str(src), rng.uniform(-0.5, 0.5, (100, 5)), 44100,
                     subtype="PCM_16", format="WAV")
            with self.assertRaises(_FfmpegUnsupported):
                _process_file_ffmpeg(src, d / "x.flac", headroom_db=1.0,
                                     dither=True, target_samplerate=44100,
                                     target_channels=2)


class NumpyFallbackTests(unittest.TestCase):
    """Tests that the numpy pipeline handles cases the ffmpeg path rejects."""

    def test_float_to_24bit(self):
        """Float source -> numpy fallback -> 24-bit FLAC with dither."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "f32.wav"
            _mk_float_wav(src, peak=0.9)
            dst = d / "out.flac"
            row = process_file(src, dst, headroom_db=1.0, dither=True,
                               target_samplerate=44100, target_channels=2)
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["output_bit_depth"], 24)
            info = sf.info(str(dst))
            self.assertEqual(info.subtype, "PCM_24")
            self.assertEqual(info.samplerate, 44100)

    def test_float_overs_gain_reduction(self):
        """Float overs -> gain reduction in numpy fallback -> peak <= ceiling."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "f32_ov.wav"
            _mk_float_wav(src, peak=1.5)  # +3.5 dBFS
            dst = d / "out.flac"
            row = process_file(src, dst, headroom_db=1.0, dither=True,
                               target_samplerate=44100, target_channels=2)
            self.assertEqual(row["status"], "ok")
            self.assertNotEqual(row["gain_db"], "0.00", "gain should be non-zero")
            got = sf.read(str(dst), dtype="float64")[0]
            peak_after = float(np.abs(got).max())
            target_ceiling = 10 ** (-1.0 / 20.0)
            self.assertLessEqual(peak_after, target_ceiling + 1e-6,
                                 f"peak {peak_after:.6f} exceeds ceiling {target_ceiling:.6f}")

    def test_5ch_numpy_fallback(self):
        """>2ch source -> process_file falls back to numpy -> stereo output."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "5ch.wav"
            rng = np.random.default_rng(44)
            sf.write(str(src), rng.uniform(-0.5, 0.5, (44100, 5)), 44100,
                     subtype="PCM_24", format="WAV")
            dst = d / "out.flac"
            row = process_file(src, dst, headroom_db=1.0, dither=True,
                               target_samplerate=44100, target_channels=2)
            self.assertEqual(row["status"], "ok")
            info = sf.info(str(dst))
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.subtype, "PCM_24")


if __name__ == "__main__":
    unittest.main()