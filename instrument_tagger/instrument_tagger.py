#!/usr/bin/env python3
"""
Stem CNN6 instrument classifier (11-class, raw-waveform ONNX).

Replaces the former PaSST OpenMIC-2018 runner. The model bakes its own
STFT + LogMel frontend into the ONNX graph, so this runner feeds raw 32 kHz
mono waveforms straight to ONNX Runtime — no external mel frontend, no torch,
no hear21passt.

Classes (must match train_vocal_classifier.CLASSES and the ONNX output column
order — column i == STEM_CLASSES[i]):
    BASS, DRUMS, FLUTE, FX, GUITAR, KEYS, ORGAN, STRINGS, SYNTH, VOCALS, WINDS

Phase-1 CLI: classify files / folder → JSON lines on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

# Avoid Windows cp1252 crashes on non-ASCII paths / names in print().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ----------------------------------------------------------
# Constants — Stem CNN6 (32 kHz, 10 s clips, raw-waveform input)
# ----------------------------------------------------------

SAMPLE_RATE = 32000
MAX_AUDIO_SECONDS = 10.0
CLIP_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)  # 320000 — training segment

# Fully-silent stems (no signal anywhere in the file) are ambiguous — skip them
# instead of emitting an arbitrary label (the model never saw silence in
# training). Peak amplitude below this floor counts as silence.
SILENCE_PEAK_FLOOR = 1e-4

# Cap chunks/files per Session.run() to bound peak memory. The exported graph
# has a dynamic batch axis, so pooling many clips into one call is much faster
# than one call per clip — especially on GPU.
MAX_ONNX_BATCH = 16

# Long audio is chunked into 10 s clips with 50% overlap (matches the Classify
# path); chunks quieter than this RMS floor contribute ~0 weight to the average.
CHUNK_OVERLAP = 0.5
SILENCE_RMS_FLOOR = 1e-4

HERE = Path(__file__).resolve().parent
# Single model source: the root models/ folder (beside the exe when frozen).
MODEL_DIR = HERE.parent / "models"
ONNX_FILENAME = "stem_cnn6.onnx"

# Class index map — MUST match train_vocal_classifier.CLASSES and the ONNX
# output column order. Single source of truth is the training script; this is
# a verbatim copy so the runner has no dependency on the training project.
STEM_CLASSES = (
    "BASS",
    "DRUMS",
    "FLUTE",
    "FX",
    "GUITAR",
    "KEYS",
    "ORGAN",
    "STRINGS",
    "SYNTH",
    "VOCALS",
    "WINDS",
)
N_CLASSES = len(STEM_CLASSES)

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac",
    ".aif",
    ".aiff",
}


def _status(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _print_json(obj: dict) -> None:
    """Emit one JSON line on stdout; survive Windows console encoding."""
    try:
        print(json.dumps(obj, ensure_ascii=False), flush=True)
    except UnicodeEncodeError:
        print(json.dumps(obj, ensure_ascii=True), flush=True)


def load_mono_32k(filename: str | Path) -> np.ndarray:
    """Full-length mono float32 @ 32 kHz; empty if fully silent.

    Loads the entire file (downmixed to mono, resampled to 32 kHz) so the
    backend can chunk it and classify the *whole* stem — a stem with a silent
    intro is still classified from its later content. A fully-silent stem (no
    signal anywhere) returns an empty array so the caller can skip it.
    Peak-normalises only if the signal exceeds 1.0 (matches the training-time
    preprocessing exactly).
    """
    import librosa

    audio: np.ndarray | None = None
    sr = SAMPLE_RATE
    full_peak = 0.0

    try:
        info = sf.info(str(filename))
        if info.samplerate > 0:
            sr = int(info.samplerate)
        else:
            info = None
    except Exception:
        info = None

    if info is not None:
        # Stream the whole file (memory-light) down to mono while tracking peak.
        try:
            parts: list[np.ndarray] = []
            with sf.SoundFile(str(filename)) as f:
                for block in f.blocks(blocksize=65536, always_2d=True):
                    mono = block.mean(axis=1).astype(np.float32, copy=False)
                    if mono.size:
                        bp = float(np.max(np.abs(mono)))
                        if bp > full_peak:
                            full_peak = bp
                    parts.append(mono)
            audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        except Exception:
            audio = None

    if audio is None:
        # sf can't decode some codecs (m4a/aac/opus) — librosa/audioread uses
        # ffmpeg (bundled app ffmpeg is on PATH).
        audio, sr = librosa.load(str(filename), sr=None, mono=True)
        full_peak = float(np.max(np.abs(audio))) if audio.size else 0.0

    if full_peak < SILENCE_PEAK_FLOOR:
        return np.zeros(0, dtype=np.float32)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq"
        )
    # Peak normalize lightly — matches trainer.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


class StemCnn6BackendOnnx:
    """Raw-waveform → ONNX → 11-class probabilities.

    The ONNX graph embeds the full STFT + LogMel + CNN6 stack, so this backend
    only needs to: stack clips into a batch, run the session, clip outputs to
    [0, 1]. No mel frontend, no torch.

    Public API mirrors the old PaSST backend (``predict`` / ``predict_batch`` /
    ``name`` / ``device``) so ``instrument_enrich.py`` is unchanged.
    """

    name = "stem-cnn6"

    def __init__(self, onnx_path: Path, device: str = ""):
        try:
            from ort_util import create_ort_session
        except ImportError:
            root = Path(__file__).resolve().parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from ort_util import create_ort_session

        self.session = create_ort_session(onnx_path, device=device or "")
        self.device = device or "onnx"
        # Expose active EP for logging (cuda vs cpu).
        try:
            providers = self.session.get_providers()
            if "CUDAExecutionProvider" in providers:
                self.device = "cuda"
            elif providers:
                self.device = str(providers[0]).replace("ExecutionProvider", "").lower()
        except Exception:
            pass

    def predict(self, audio: np.ndarray) -> np.ndarray:
        return self.predict_batch([audio])[0]

    def _chunk_mono(self, audio) -> tuple[list[np.ndarray], list[float]]:
        """Split a mono waveform into 10 s clips + per-clip RMS weights.

        Long audio is chunked with 50% overlap (matches the Classify path).
        Chunks quieter than ``SILENCE_RMS_FLOOR`` get weight 0, so a silent
        intro can't dilute the chunks that actually carry signal. Empty/None
        input returns ([], []).
        """
        if audio is None or getattr(audio, "size", 0) == 0:
            return [], []
        mono = np.asarray(audio, dtype=np.float32)
        t_len = mono.shape[0]
        if t_len <= CLIP_SAMPLES:
            starts = [0]
        else:
            stride = max(1, int(CLIP_SAMPLES * (1.0 - CHUNK_OVERLAP)))
            n_chunks = (t_len - CLIP_SAMPLES) // stride + 1
            if (n_chunks - 1) * stride + CLIP_SAMPLES < t_len:
                n_chunks += 1
            starts = [ci * stride for ci in range(n_chunks)]

        chunks: list[np.ndarray] = []
        weights: list[float] = []
        for start in starts:
            end = min(start + CLIP_SAMPLES, t_len)
            real = mono[start:end]
            rms = float(np.sqrt(np.mean(real**2) + 1e-12)) if real.size else 0.0
            weights.append(rms if rms >= SILENCE_RMS_FLOOR else 0.0)
            if real.shape[0] < CLIP_SAMPLES:
                padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)
                padded[: real.shape[0]] = real
                chunks.append(padded)
            else:
                chunks.append(real)
        return chunks, weights

    def predict_batch(self, audios: list[np.ndarray]) -> np.ndarray:
        """Classify full stems → (N, 11) float32, chunking long audio.

        Each input is split into 10 s clips (50% overlap), pooled into shared
        ONNX calls capped at ``MAX_ONNX_BATCH``, then RMS-weighted back into one
        probability vector per input. Fully-silent inputs (or inputs whose every
        chunk is silent) return an all-zero row — no valid prediction.
        """
        n = len(audios)
        if n == 0:
            return np.zeros((0, N_CLASSES), dtype=np.float32)

        per_audio_chunks: list[list[np.ndarray]] = []
        per_audio_weights: list[list[float]] = []
        pooled: list[np.ndarray] = []
        for audio in audios:
            chunks, weights = self._chunk_mono(audio)
            per_audio_chunks.append(chunks)
            per_audio_weights.append(weights)
            pooled.extend(chunks)

        pooled_probs = self._run_pooled(pooled)  # (total_chunks, N_CLASSES)

        out = np.zeros((n, N_CLASSES), dtype=np.float32)
        offset = 0
        for i in range(n):
            cnt = len(per_audio_chunks[i])
            if cnt == 0:
                continue
            p = pooled_probs[offset : offset + cnt]
            offset += cnt
            w = np.asarray(per_audio_weights[i], dtype=np.float32)
            wsum = float(w.sum())
            if wsum > 0:
                out[i] = (p * w[:, np.newaxis]).sum(axis=0) / wsum
            # else: every chunk was silent — leave the row zeroed (skip).
        return out

    def _run_pooled(self, chunks: list[np.ndarray]) -> np.ndarray:
        """Run pooled clips through ONNX in sub-batches bounded by MAX_ONNX_BATCH."""
        if not chunks:
            return np.zeros((0, N_CLASSES), dtype=np.float32)
        out_rows: list[np.ndarray] = []
        for start in range(0, len(chunks), MAX_ONNX_BATCH):
            sub = chunks[start : start + MAX_ONNX_BATCH]
            inp = np.stack(sub, axis=0)
            probs = self.session.run(["probs"], {"audio": inp})[0]
            out_rows.append(np.clip(probs, 0.0, 1.0).astype(np.float32))
        return np.concatenate(out_rows, axis=0)


def _batch_size() -> int:
    raw = os.environ.get("PASST_BATCH_SIZE", "16").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def _audio_workers() -> int:
    raw = os.environ.get("PASST_AUDIO_WORKERS", "2").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 2


# Back-compat aliases — instrument_enrich.py imports these names.
_passt_batch_size = _batch_size
_passt_audio_workers = _audio_workers


def _resolve_onnx() -> Path | None:
    """Resolve the Stem CNN6 ONNX weight. Returns None if not present.

    The model file is not bundled in source control; it is either built
    locally (place at models/stem_cnn6.onnx) or downloaded
    by the installer into the same path.
    """
    candidates = [
        MODEL_DIR / ONNX_FILENAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_backend(status=_status) -> StemCnn6BackendOnnx:
    """Load Stem CNN6 ONNX. Raises SystemExit with a helpful message if absent.

    The old PaSST runner had a torch/hear21passt fallback here; the Stem CNN6
    is ONNX-only (its frontend is baked into the graph), so there is no
    fallback path — onnxruntime is the only backend.
    """
    onnx_path = _resolve_onnx()
    if onnx_path is None:
        raise SystemExit(
            f"ERROR: Stem CNN6 ONNX weight not found.\n"
            f"  Expected at: {MODEL_DIR / ONNX_FILENAME}\n"
            f"  Place the exported stem_cnn6.onnx there (export from the\n"
            f"  train-vocal-classifier project, or restore via the installer).\n"
        )
    status("  loading Stem CNN6 (onnxruntime)...")
    backend = StemCnn6BackendOnnx(onnx_path, device="")
    status(f"  device: {backend.device}")
    # Warm up the session so the first real predict isn't slow.
    try:
        backend.predict(np.zeros(CLIP_SAMPLES, dtype=np.float32))
    except Exception:
        pass
    return backend


def probs_to_result(
    probs: np.ndarray,
    *,
    top_k: int = 5,
    threshold: float = 0.0,
) -> dict:
    """11-class softmax probabilities → primary label + top-k list.

    Unlike the old PaSST runner there is no synth-demote hack: the model's
    softmax already produces a single clean distribution over mutually
    exclusive classes, so argmax is the correct primary label.
    """
    probs = np.asarray(probs, dtype=np.float32)
    if probs.size == 0 or float(probs.max()) <= 0.0:
        # Zeroed row: the clip was empty or near-silent, so there is no valid
        # prediction. Return an empty result so the rename pipeline skips it.
        return {
            "label": "",
            "score": 0.0,
            "score_raw": 0.0,
            "top": [],
            "above": [],
            "n_patches": 0,
            "model": "stem-cnn6",
            "demoted_synth": False,
            "second_score": 0.0,
            "silent": True,
        }
    order = np.argsort(-probs)
    best_i = int(order[0])

    top = []
    for i in order[: max(1, top_k)]:
        score = float(probs[i])
        if score < threshold and top:
            break
        top.append([STEM_CLASSES[int(i)], score])

    above = [
        [STEM_CLASSES[int(i)], float(probs[i])]
        for i in order
        if float(probs[i]) >= threshold
    ]

    p1 = float(probs[best_i])
    others = [int(i) for i in order if int(i) != best_i]
    p2 = float(probs[others[0]]) if others else 0.0

    return {
        "label": STEM_CLASSES[best_i],
        "score": p1,
        "score_raw": p1,
        "top": top,
        "above": above,
        "n_patches": 1,
        "model": "stem-cnn6",
        # Field kept for cache/back-compat shape parity with the old PaSST
        # result; always False under this model.
        "demoted_synth": False,
        # Margin vs runner-up — handy for confidence gating downstream.
        "second_score": p2,
        "silent": False,
    }


def classify_file(
    filename: str | Path,
    backend: StemCnn6BackendOnnx,
    *,
    top_k: int = 5,
    threshold: float = 0.0,
) -> dict:
    audio = load_mono_32k(filename)
    probs = backend.predict(audio)
    result = probs_to_result(probs, top_k=top_k, threshold=threshold)
    result["path"] = str(Path(filename).resolve())
    return result


def iter_audio_files(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify audio with Stem CNN6 (11-class instruments).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="Single audio file")
    src.add_argument("--folder", type=Path, help="Folder (recursive)")
    src.add_argument(
        "--files-from",
        type=Path,
        help="Text file with one audio path per line (use - for stdin)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Top-k labels in output (default 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Min softmax score for above[] list (default 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max files from folder (0 = all)",
    )
    args = parser.parse_args(argv)

    files: list[Path] = []
    if args.file is not None:
        if not args.file.is_file():
            _status(f"ERROR: not a file: {args.file}")
            return 1
        files = [args.file]
    elif args.files_from is not None:
        if str(args.files_from) == "-":
            raw_lines = sys.stdin.read().splitlines()
        else:
            if not args.files_from.is_file():
                _status(f"ERROR: not a file: {args.files_from}")
                return 1
            raw_lines = args.files_from.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        for line in raw_lines:
            line = line.lstrip("\ufeff").strip().strip('"')
            if not line:
                continue
            path = Path(line)
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(path)
            if args.limit and len(files) >= args.limit:
                break
        if not files:
            _status("ERROR: no valid audio paths in --files-from list")
            return 1
    else:
        if not args.folder.is_dir():
            _status(f"ERROR: not a folder: {args.folder}")
            return 1
        for i, path in enumerate(iter_audio_files(args.folder), 1):
            files.append(path)
            if args.limit and i >= args.limit:
                break
        if not files:
            _status(f"ERROR: no audio files under {args.folder}")
            return 1

    _status(f"Instrument tagger (Stem CNN6) — {len(files)} file(s)")
    backend = load_backend(status=_status)
    batch_size = _batch_size()
    audio_workers = _audio_workers()
    _status(f"  backend: {backend.name}")
    _status(f"  batch={batch_size} decode_workers={audio_workers}")
    _status("")

    from concurrent.futures import ThreadPoolExecutor

    def _safe_load(path: Path):
        try:
            return path, load_mono_32k(path), None
        except Exception as exc:
            return path, None, str(exc)

    errors = 0
    predict_batch = getattr(backend, "predict_batch", None)

    def _load_chunk(paths: list[Path]):
        return list(pool.map(_safe_load, paths))

    def _emit_chunk(loaded) -> None:
        nonlocal errors
        ok_paths: list[Path] = []
        ok_audios: list[np.ndarray] = []
        for path, audio, err in loaded:
            if err is not None or audio is None:
                errors += 1
                _print_json({"path": str(path.resolve()), "error": err or "load failed"})
                continue
            ok_paths.append(path)
            ok_audios.append(audio)
        if not ok_audios:
            return
        try:
            if callable(predict_batch):
                probs_batch = predict_batch(ok_audios)
            else:
                probs_batch = np.stack([backend.predict(a) for a in ok_audios], axis=0)
        except Exception as exc:
            for path, audio in zip(ok_paths, ok_audios):
                try:
                    probs = backend.predict(audio)
                    result = probs_to_result(
                        probs, top_k=args.top, threshold=args.threshold
                    )
                    result["path"] = str(path.resolve())
                    _print_json(result)
                except Exception as exc2:
                    errors += 1
                    _print_json({"path": str(path.resolve()), "error": str(exc2)})
            _status(f"  [warn] batch infer failed ({exc}); fell back per-file")
            return
        for path, probs in zip(ok_paths, probs_batch):
            result = probs_to_result(probs, top_k=args.top, threshold=args.threshold)
            result["path"] = str(path.resolve())
            _print_json(result)

    with ThreadPoolExecutor(max_workers=audio_workers) as pool:
        starts = list(range(0, len(files), batch_size))
        next_fut = None
        for i, start in enumerate(starts):
            chunk = files[start : start + batch_size]
            if next_fut is not None:
                loaded = next_fut.result()
                next_fut = None
            else:
                loaded = _load_chunk(chunk)
            if i + 1 < len(starts):
                next_chunk = files[starts[i + 1] : starts[i + 1] + batch_size]
                next_fut = pool.submit(_load_chunk, next_chunk)
            _emit_chunk(loaded)

    _status(f"done. ok={len(files) - errors} err={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
