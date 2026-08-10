#!/usr/bin/env python3
"""
PANNs (Cnn14) AudioSet tagger — vocal / speech focus.

Pretrained on AudioSet (527 classes). No fine-tuning required.
Emits JSON lines on stdout (same protocol as instrument_tagger).

Default focus classes: Singing, Speech, Rapping, Humming, Choir
(+ related AudioSet siblings when present in the label list).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


def _bootstrap_sibling_site_packages() -> None:
    """Ensure ``site-packages`` (beside STEM-organizer.exe) is on ``sys.path``."""
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        try:
            path = path.resolve()
        except OSError:
            pass
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    explicit = os.environ.get("STEM_SITE_PACKAGES", "").strip()
    if explicit:
        add(Path(explicit))

    for part in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        part = part.strip()
        if part:
            add(Path(part))

    here = Path(__file__).resolve().parent
    for folder in (here, *here.parents):
        if folder.name.lower() == "_internal":
            continue
        add(folder / "site-packages")
        if any(
            (folder / name).is_file()
            for name in (
                "STEM-organizer.exe",
                "stem-organizer.exe",
                "STEM_organizer.exe",
            )
        ):
            add(folder / "site-packages")
            break

    ordered = sorted(
        candidates,
        key=lambda p: (
            0 if (p.is_dir() and (p / "panns_inference").is_dir()) else 1,
            0 if (p.is_dir() and (p / "torch").is_dir()) else 1,
            0 if p.is_dir() else 1,
        ),
    )
    for site in ordered:
        if not site.is_dir():
            continue
        entry = str(site)
        if entry not in sys.path:
            sys.path.insert(0, entry)
        break


_bootstrap_sibling_site_packages()

import numpy as np
import soundfile as sf

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ----------------------------------------------------------
# Constants — PANNs Cnn14 expects 32 kHz mono
# ----------------------------------------------------------

SAMPLE_RATE = 32000
# Clip-level default: first 30 s (AudioSet clips are ~10 s; longer is fine).
MAX_AUDIO_SECONDS = 30.0

HERE = Path(__file__).resolve().parent
# Single model source: the root models/ folder (beside the exe when frozen).
MODEL_DIR = HERE.parent / "models"

# Official assets (panns_inference uses wget for these — broken on Windows PATH).
_LABELS_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
    "class_labels_indices.csv"
)
_CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
)
_CHECKPOINT_NAME = "Cnn14_mAP=0.431.pth"
_LABELS_NAME = "class_labels_indices.csv"
# panns_inference re-downloads if file is smaller than ~300 MB.
_MIN_CHECKPOINT_BYTES = 300_000_000

# Primary focus set the user asked about (exact AudioSet display names).
FOCUS_LABELS = (
    "Singing",
    "Speech",
    "Rapping",
    "Humming",
    "Choir",
)

# Speech is a loose AudioSet bucket (spoken word, narration, etc.). When it barely
# beats a music-focused runner-up on renormalized focus shares, prefer the runner-up
# — same idea as synthesizer demotion in instrument_tagger (rename model).
SPEECH_DEMOTE_MAX_GAP = 0.20

# Extra related AudioSet labels reported under vocal{} when present.
RELATED_LABELS = (
    "Male singing",
    "Female singing",
    "Child singing",
    "Synthetic singing",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Whispering",
    "Shout",
    "Yodeling",
    "Chant",
    "Vocal music",
    "A capella",
)

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


def _fmt_summary_elapsed(seconds: float) -> str:
    """m:ss or h:mm:ss (same as Genre / Gender summaries)."""
    total = max(0, int(round(float(seconds or 0))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _print_vocal_summary(
    *,
    elapsed: float,
    files: int,
    tagged: int | None,
    skipped: int | None,
    errors: int,
    peak_vram_gb: float | None = None,
    extra_lines: list | None = None,
) -> None:
    """=== Vocal type Summary === — same footer as Genre / Gender."""
    n = max(0, int(files or 0))
    minutes = max(float(elapsed or 0) / 60.0, 1e-9)
    print(flush=True)
    print("=== Vocal type Summary ===", flush=True)
    print(f"  Total time: {_fmt_summary_elapsed(elapsed)}", flush=True)
    print(f"  Files: {n:,}", flush=True)
    if n > 0:
        print(f"  Sec/file: {float(elapsed) / n:.3f}", flush=True)
        print(f"  Files/min: {n / minutes:.2f}", flush=True)
    if tagged is not None:
        if skipped is not None:
            print(f"  Tagged: {tagged:,} | Skipped: {skipped:,}", flush=True)
        else:
            print(f"  Tagged: {tagged:,}", flush=True)
    if int(errors or 0) > 0:
        print(f"  Errors: {int(errors):,}", flush=True)
    if peak_vram_gb is not None:
        print(f"  Peak VRAM: {float(peak_vram_gb):.2f} GB", flush=True)
    if extra_lines:
        for line in extra_lines:
            text = str(line).strip()
            if text:
                print(f"  {text}", flush=True)
    print(flush=True)
    print("DONE", flush=True)


def _print_json(obj: dict) -> None:
    try:
        print(json.dumps(obj, ensure_ascii=False), flush=True)
    except UnicodeEncodeError:
        print(json.dumps(obj, ensure_ascii=True), flush=True)


def _panns_home() -> Path:
    return Path.home() / "panns_data"


def _download_file(
    url: str,
    dest: Path,
    *,
    status=_status,
    timeout: int = 600,
) -> None:
    """Download ``url`` → ``dest`` with urllib (no wget)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    status(f"  downloading {dest.name} ...")
    status(f"    {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
            total = resp.headers.get("Content-Length")
            total_n = int(total) if total and total.isdigit() else 0
            done = 0
            last_pct = -1
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total_n > 0:
                    pct = int(100 * done / total_n)
                    if pct >= last_pct + 5:
                        status(f"    {pct}% ({done // (1024 * 1024)} MB)")
                        last_pct = pct
                elif done and done % (50 * 1024 * 1024) < 256 * 1024:
                    status(f"    {done // (1024 * 1024)} MB...")
        tmp.replace(dest)
        status(f"  saved: {dest} ({dest.stat().st_size // (1024 * 1024)} MB)")
    except Exception:
        try:
            if tmp.is_file():
                tmp.unlink()
        except OSError:
            pass
        raise


def ensure_panns_labels(*, status=_status) -> Path:
    """Ensure the AudioSet labels CSV exists (no checkpoint download).

    On the ONNX path we only need the labels; the 327 MB Cnn14 .pth is skipped.
    """
    home = _panns_home()
    home.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    labels_home = home / _LABELS_NAME
    labels_local = MODEL_DIR / _LABELS_NAME
    if not labels_home.is_file():
        if labels_local.is_file():
            import shutil

            shutil.copy2(labels_local, labels_home)
            status(f"  labels: copied → {labels_home}")
        else:
            _download_file(_LABELS_URL, labels_home, status=status)
            try:
                import shutil

                shutil.copy2(labels_home, labels_local)
            except Exception:
                pass
    elif not labels_local.is_file():
        try:
            import shutil

            shutil.copy2(labels_home, labels_local)
        except Exception:
            pass
    return labels_local


def ensure_panns_assets(*, status=_status) -> Path:
    """Ensure labels CSV + Cnn14 checkpoint exist (Windows-safe, no wget).

    ``panns_inference.config`` imports fail if ``~/panns_data/class_labels_indices.csv``
    is missing — and its only download path is ``os.system('wget …')``.
    Torch-fallback path only; the ONNX path uses :func:`ensure_panns_labels`.
    """
    ensure_panns_labels(status=status)
    home = _panns_home()

    ckpt_local = MODEL_DIR / _CHECKPOINT_NAME
    ckpt_home = home / _CHECKPOINT_NAME

    def _ckpt_ok(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size >= _MIN_CHECKPOINT_BYTES
        except OSError:
            return False

    if not _ckpt_ok(ckpt_local) and _ckpt_ok(ckpt_home):
        import shutil

        shutil.copy2(ckpt_home, ckpt_local)
    if not _ckpt_ok(ckpt_local):
        _download_file(_CHECKPOINT_URL, ckpt_local, status=status)
    if _ckpt_ok(ckpt_local) and not _ckpt_ok(ckpt_home):
        try:
            import shutil

            shutil.copy2(ckpt_local, ckpt_home)
        except Exception:
            pass

    return ckpt_local


def _default_checkpoint() -> Path:
    """Prefer local models\\Cnn14… then ~/panns_data."""
    local = MODEL_DIR / _CHECKPOINT_NAME
    if local.is_file():
        return local
    if MODEL_DIR.is_dir():
        for path in sorted(MODEL_DIR.glob("Cnn14*.pth")):
            return path
    return _panns_home() / _CHECKPOINT_NAME


def _resolve_cnn14_onnx(checkpoint: Path | None, ensured: Path) -> Path | None:
    """Find a cnn14.onnx next to the (resolved) .pth checkpoint or in models/."""
    candidates: list[Path] = []
    base = MODEL_DIR / "cnn14.onnx"
    candidates.append(base)
    if checkpoint is not None and checkpoint.is_file():
        candidates.append(checkpoint.with_suffix(".onnx"))
    if ensured is not None and ensured.is_file():
        candidates.append(ensured.with_suffix(".onnx"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_audioset_labels() -> list[str]:
    """Load AudioSet display_name labels from class_labels_indices.csv.

    Mirrors ``panns_inference.labels`` (527 entries, CSV order) so the ONNX
    backend needs no panns_inference import at runtime.
    """
    for path in (MODEL_DIR / _LABELS_NAME, _panns_home() / _LABELS_NAME):
        if not path.is_file():
            continue
        labels: list[str] = []
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                labels.append(row["display_name"])
        if labels:
            return labels
    raise FileNotFoundError(
        f"{_LABELS_NAME} not found in {MODEL_DIR} or {_panns_home()}"
    )


def load_mono_32k(
    filename: str | Path,
    *,
    max_seconds: float = MAX_AUDIO_SECONDS,
) -> np.ndarray:
    """Mono float32 @ 32 kHz."""
    import librosa

    max_src = None
    try:
        info = sf.info(str(filename))
        if info.samplerate > 0 and max_seconds > 0:
            max_src = int(max_seconds * info.samplerate) + info.samplerate
    except Exception:
        max_src = None

    try:
        data, sr = sf.read(
            str(filename),
            always_2d=True,
            dtype="float32",
            frames=max_src if max_src else -1,
        )
        audio = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=SAMPLE_RATE,
                res_type="soxr_hq",
            )
    except Exception:
        # sf can't decode some codecs (m4a/aac/opus) — librosa/audioread uses
        # ffmpeg (bundled app ffmpeg is on PATH).
        audio, _ = librosa.load(str(filename), sr=SAMPLE_RATE, mono=True)
    if max_seconds > 0:
        cap = int(max_seconds * SAMPLE_RATE)
        if audio.shape[0] > cap:
            audio = audio[:cap]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


class _PannsBackend:
    """Cnn14 audio tagging via panns_inference."""

    name = "panns-cnn14"

    def __init__(self, tagger, labels: list[str], device: str):
        self.tagger = tagger
        self.labels = list(labels)
        self.device = device
        self._label_index = {name: i for i, name in enumerate(self.labels)}

    def predict(self, audio: np.ndarray) -> np.ndarray:
        return self.predict_batch([audio])[0]

    def predict_batch(self, audios: list[np.ndarray]) -> np.ndarray:
        """Sequential torch predicts stacked to (N, C) — no fused batch API."""
        n_lab = len(self.labels)
        if not audios:
            return np.zeros((0, n_lab), dtype=np.float32)
        out = np.zeros((len(audios), n_lab), dtype=np.float32)
        for i, audio in enumerate(audios):
            if audio is None or getattr(audio, "size", 0) == 0:
                continue
            batch = np.asarray(audio, dtype=np.float32)[None, :]
            clipwise, _embedding = self.tagger.inference(batch)
            probs = np.asarray(clipwise[0], dtype=np.float32)
            n = min(n_lab, probs.shape[0])
            out[i, :n] = probs[:n]
        return out


def _pad_waveforms(audios: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Zero-pad to max length in chunk. Returns (B, T) and empty-slot indices."""
    empty_idx: list[int] = []
    cleaned: list[np.ndarray] = []
    max_t = 1
    for i, audio in enumerate(audios):
        if audio is None or getattr(audio, "size", 0) == 0:
            empty_idx.append(i)
            cleaned.append(np.zeros(1, dtype=np.float32))
        else:
            a = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
            cleaned.append(a)
            if a.shape[0] > max_t:
                max_t = int(a.shape[0])
    stacked = np.zeros((len(cleaned), max_t), dtype=np.float32)
    for i, a in enumerate(cleaned):
        stacked[i, : a.shape[0]] = a
    return stacked, empty_idx


class _PannsBackendOnnx:
    """ONNX Runtime drop-in for _PannsBackend.

    Same public API (``predict`` / ``predict_batch`` / ``labels`` / ``name``).
    The full Cnn14 pipeline (Spectrogram + LogMel + BN + conv blocks) is in the
    ONNX graph, so input is a raw 32 kHz mono waveform and output is 527
    AudioSet probs — no torchlibrosa / librosa needed at runtime.
    """

    name = "panns-cnn14"

    def __init__(self, onnx_path, labels: list[str], device: str = ""):
        try:
            from ort_util import create_ort_session
        except ImportError:
            import sys
            from pathlib import Path as _P

            root = _P(__file__).resolve().parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from ort_util import create_ort_session

        self.session = create_ort_session(onnx_path, device=device or "")
        self.labels = list(labels)
        self.device = device or "onnx"
        try:
            providers = self.session.get_providers()
            if "CUDAExecutionProvider" in providers:
                self.device = "cuda"
            elif providers:
                self.device = str(providers[0]).replace("ExecutionProvider", "").lower()
        except Exception:
            pass
        self._label_index = {name: i for i, name in enumerate(self.labels)}

    def predict(self, audio: np.ndarray) -> np.ndarray:
        return self.predict_batch([audio])[0]

    def predict_batch(self, audios: list[np.ndarray]) -> np.ndarray:
        """Pad to max T, one ORT ``session.run``. Returns (N, C) float32."""
        n_lab = len(self.labels)
        if not audios:
            return np.zeros((0, n_lab), dtype=np.float32)
        batch, empty_idx = _pad_waveforms(audios)
        probs = self.session.run(
            ["probs"], {"audio": np.ascontiguousarray(batch, dtype=np.float32)}
        )[0]
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim == 1:
            probs = probs[None, :]
        if probs.shape[1] != n_lab:
            out = np.zeros((probs.shape[0], n_lab), dtype=np.float32)
            n = min(n_lab, probs.shape[1])
            out[:, :n] = probs[:, :n]
            probs = out
        for i in empty_idx:
            probs[i] = 0.0
        return probs


def _panns_batch_size() -> int:
    raw = os.environ.get("PANNS_BATCH_SIZE", "4").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def _panns_audio_workers() -> int:
    raw = os.environ.get("PANNS_AUDIO_WORKERS", "2").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def load_backend(
    *,
    checkpoint: Path | None = None,
    device: str | None = None,
    status=_status,
) -> _PannsBackend | _PannsBackendOnnx:
    # ONNX path (default): single cnn14.onnx, full pipeline in-graph. Avoids
    # importing panns_inference / torch AND downloading the 327 MB .pth. Only
    # the labels CSV is needed; ensure just that (no checkpoint download).
    if os.environ.get("STEM_ONNX", "1").strip() != "0":
        onnx_path = _resolve_cnn14_onnx(checkpoint, None)
        if onnx_path is not None and onnx_path.is_file():
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                pass
            else:
                try:
                    ensure_panns_labels(status=status)
                except Exception as exc:
                    raise SystemExit(
                        "\nERROR: could not obtain AudioSet labels CSV.\n"
                        f"  detail: {exc!r}\n"
                        f"  labels: {_LABELS_URL}\n"
                    ) from exc
                labels = _load_audioset_labels()
                status("  Cnn14 (onnxruntime) ready")
                return _PannsBackendOnnx(onnx_path, labels, device=device or "")

    import torch

    # Torch fallback — needs the full assets incl. the 327 MB .pth checkpoint.
    try:
        status("  ensuring PANNs assets (labels + Cnn14, no wget)...")
        ensured = ensure_panns_assets(status=status)
    except Exception as exc:
        raise SystemExit(
            "\nERROR: could not download PANNs assets.\n"
            f"  detail: {exc!r}"
            "  Manual fix: put these under %USERPROFILE%\\panns_data\\\n"
            "    - class_labels_indices.csv\n"
            "    - Cnn14_mAP=0.431.pth\n"
            f"  labels: {_LABELS_URL}\n"
            f"  weights: {_CHECKPOINT_URL}\n"
        ) from exc

    try:
        from panns_inference import AudioTagging, labels as panns_labels
    except ImportError as exc:
        raise SystemExit(
            "\nERROR: panns_inference not installed.\n"
            "  Frozen build: run install-deps.bat beside STEM-organizer.exe\n"
            "  From source: run panns_tagger\\install-deps.bat\n"
            "    (or root install-deps.bat).\n"
            f"  python: {sys.executable}\n"
            f"  detail: {exc!r}\n"
        ) from exc
    except FileNotFoundError as exc:
        raise SystemExit(
            "\nERROR: PANNs labels CSV still missing after download attempt.\n"
            f"  detail: {exc!r}\n"
            f"  expected: {_panns_home() / _LABELS_NAME}\n"
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint is not None and checkpoint.is_file():
        ckpt_path = checkpoint
    else:
        ckpt_path = ensured if ensured.is_file() else _default_checkpoint()
    if not ckpt_path.is_file():
        raise SystemExit(f"\nERROR: Cnn14 checkpoint missing: {ckpt_path}\n")

    import contextlib
    import io

    # AudioTagging prints "Checkpoint path: …" — keep Vocal LOG clean.
    with contextlib.redirect_stdout(io.StringIO()):
        tagger = AudioTagging(checkpoint_path=str(ckpt_path), device=device)

    labels = list(panns_labels)
    return _PannsBackend(tagger, labels, device)


def _vocal_scores(probs: np.ndarray, label_index: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in (*FOCUS_LABELS, *RELATED_LABELS):
        idx = label_index.get(name)
        if idx is None:
            continue
        out[name] = float(probs[idx])
    return out


def _softmax_renorm(scores: dict[str, float], keys: tuple[str, ...]) -> dict[str, float]:
    """Renormalize a label subset to shares that sum to 1.0 (RMS-style).

    PANNs clipwise outputs are multi-label sigmoids (do not sum to 1). Softmax
    over ``log(p)`` on the focus subset is equivalent to ``p / sum(p)`` — the
    same exclusive-share math Classify uses for RMS energy / total.
    """
    vals = np.array([float(scores.get(k, 0.0)) for k in keys], dtype=np.float64)
    if vals.size == 0:
        return {}
    # Clamp then log-softmax ≡ L1 normalize (stable when some scores are ~0).
    vals = np.clip(vals, 1e-12, None)
    log_v = np.log(vals)
    log_v -= float(np.max(log_v))
    ex = np.exp(log_v)
    total = float(ex.sum()) + 1e-12
    return {k: float(ex[i] / total) for i, k in enumerate(keys)}


def _calibrate_focus_confidence(top1: float, top2: float | None) -> float:
    """p1/(p1+p2) on focus shares — matches genre / rename confidence reading."""
    t1 = float(top1)
    if top2 is None:
        return t1
    t2 = float(top2)
    denom = t1 + t2
    return (t1 / denom) if denom > 0.0 else 0.0


def _pick_focus_label(vocal: dict[str, float]) -> tuple[str, float, bool]:
    """Argmax focus share, demoting Speech when a runner-up is within gap."""
    if not vocal:
        return "Singing", 0.0, False
    ranked = sorted(vocal.items(), key=lambda kv: -kv[1])
    label = ranked[0][0]
    share = float(ranked[0][1])
    demoted = False
    if label == "Speech" and len(ranked) > 1:
        runner_label, runner_share = ranked[1]
        if share - float(runner_share) <= SPEECH_DEMOTE_MAX_GAP:
            label = runner_label
            share = float(runner_share)
            demoted = True
    return label, share, demoted


def probs_to_result(
    probs: np.ndarray,
    backend: _PannsBackend,
    *,
    top_k: int = 10,
    focus_only: bool = False,
) -> dict:
    """Clipwise probs → primary focus label + softmax-renormed vocal shares."""
    raw = _vocal_scores(probs, backend._label_index)
    focus_keys = tuple(k for k in FOCUS_LABELS if k in raw)
    vocal = _softmax_renorm(raw, focus_keys) if focus_keys else {}

    primary_label, primary_share, demoted_speech = _pick_focus_label(vocal)
    others = [float(s) for name, s in vocal.items() if name != primary_label]
    runner_share = max(others) if others else None
    primary_score = _calibrate_focus_confidence(primary_share, runner_share)

    order = np.argsort(-probs)
    top: list[list] = []
    for i in order[: max(1, top_k)]:
        name = backend.labels[int(i)]
        if focus_only and name not in FOCUS_LABELS and name not in RELATED_LABELS:
            continue
        if focus_only and name in vocal:
            top.append([name, float(vocal[name])])
        else:
            top.append([name, float(probs[int(i)])])

    return {
        "label": primary_label,
        "score": float(primary_score),
        "score_share": float(primary_share),
        "vocal": vocal,
        "vocal_raw": {k: raw[k] for k in FOCUS_LABELS if k in raw},
        "vocal_related": {
            k: v for k, v in raw.items() if k not in FOCUS_LABELS
        },
        "top": top,
        "model": backend.name,
        "demoted_speech": demoted_speech,
    }


def _segment_windows(
    audio: np.ndarray,
    *,
    segment_sec: float,
    hop_sec: float | None = None,
) -> list[tuple[int, int, float, float]]:
    """Return (start_sample, end_sample, start_sec, end_sec) windows."""
    if segment_sec <= 0 or audio.size == 0:
        return []
    seg = max(1, int(round(segment_sec * SAMPLE_RATE)))
    hop = seg if hop_sec is None or hop_sec <= 0 else max(1, int(round(hop_sec * SAMPLE_RATE)))
    windows: list[tuple[int, int, float, float]] = []
    n = int(audio.shape[0])
    start = 0
    while start < n:
        end = min(n, start + seg)
        if end - start < max(1, seg // 4) and windows:
            # Drop a tiny trailing stub unless it's the only content.
            break
        windows.append(
            (
                start,
                end,
                start / float(SAMPLE_RATE),
                end / float(SAMPLE_RATE),
            )
        )
        if end >= n:
            break
        start += hop
    return windows


def classify_file(
    filename: str | Path,
    backend: _PannsBackend,
    *,
    top_k: int = 10,
    focus_only: bool = False,
    segment_sec: float = 0.0,
    hop_sec: float | None = None,
    max_seconds: float = MAX_AUDIO_SECONDS,
) -> dict:
    audio = load_mono_32k(filename, max_seconds=max_seconds)
    probs = backend.predict(audio)
    result = probs_to_result(
        probs, backend, top_k=top_k, focus_only=focus_only
    )
    result["path"] = str(Path(filename).resolve())
    result["duration_sec"] = float(audio.shape[0] / SAMPLE_RATE) if audio.size else 0.0

    if segment_sec and segment_sec > 0:
        segs = []
        for start_i, end_i, t0, t1 in _segment_windows(
            audio, segment_sec=segment_sec, hop_sec=hop_sec
        ):
            chunk = audio[start_i:end_i]
            # Pad very short tails so Cnn14 still runs stably.
            min_samples = int(0.5 * SAMPLE_RATE)
            if chunk.shape[0] < min_samples:
                chunk = np.pad(chunk, (0, min_samples - chunk.shape[0]))
            s_probs = backend.predict(chunk)
            s_res = probs_to_result(
                s_probs, backend, top_k=min(5, top_k), focus_only=focus_only
            )
            segs.append(
                {
                    "start": round(t0, 3),
                    "end": round(t1, 3),
                    "label": s_res["label"],
                    "score": s_res["score"],
                    "vocal": s_res["vocal"],
                    "top": s_res["top"],
                }
            )
        result["segments"] = segs
    return result


def iter_audio_files(folder: Path, *, recursive: bool = True):
    if recursive:
        paths = sorted(folder.rglob("*"))
    else:
        paths = sorted(folder.iterdir())
    for path in paths:
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def _read_existing_vocal_tag(path: Path) -> str:
    """Return existing vocal-type tag text (COMMENT or VOCAL_TYPE), else ''."""
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return ""
    try:
        audio = MutagenFile(str(path), easy=True)
        if audio is None:
            return ""
        for key in ("comment", "vocal_type", "vocal type"):
            vals = audio.get(key)
            if vals:
                return str(vals[0]).strip()
        # Non-easy: try common ID3 / Vorbis keys
        audio2 = MutagenFile(str(path))
        if audio2 is None:
            return ""
        tags = getattr(audio2, "tags", None)
        if tags is None:
            return ""
        if hasattr(tags, "get"):
            for key in ("COMMENT", "comment", "VOCAL_TYPE", "----:com.apple.iTunes:VOCAL_TYPE"):
                val = tags.get(key)
                if val is None:
                    continue
                if isinstance(val, list) and val:
                    return str(val[0]).strip()
                text = getattr(val, "text", None)
                if text:
                    return str(text[0]).strip()
                return str(val).strip()
    except Exception:
        return ""
    return ""


def _already_tagged(path: Path) -> bool:
    existing = _read_existing_vocal_tag(path)
    if not existing:
        return False
    low = existing.lower()
    for name in FOCUS_LABELS:
        if low.startswith(name.lower()) or f"/{name.lower()}" in low:
            return True
    return False


def write_vocal_metadata(
    path: Path,
    label: str,
    score: float = 0.0,
    *,
    field: str = "comment",
) -> None:
    """Write primary vocal label to COMMENT or VOCAL_TYPE (custom).

    Percentage stays in LOG only — tags get the bare label (e.g. ``Singing``).
    """
    from mutagen import File as MutagenFile
    from mutagen.flac import FLAC
    from mutagen.id3 import COMM, ID3, TXXX, ID3NoHeaderError
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4, MP4FreeForm
    from mutagen.wave import WAVE

    value = str(label).strip()
    if not value:
        raise ValueError("empty vocal label")
    suffix = path.suffix.lower()

    from file_writable import ensure_writable

    ensure_writable(path)

    if suffix == ".flac":
        audio = FLAC(str(path))
        if field == "vocal":
            audio["VOCAL_TYPE"] = [value]
        else:
            audio["COMMENT"] = [value]
        audio.save()
        return

    if suffix in (".mp3", ".wav"):
        try:
            if suffix == ".mp3":
                audio = MP3(str(path))
            else:
                audio = WAVE(str(path))
            tags = audio.tags
            if tags is None:
                audio.add_tags()
                tags = audio.tags
        except ID3NoHeaderError:
            tags = ID3()
            if suffix == ".mp3":
                audio = MP3(str(path))
                audio.tags = tags
            else:
                audio = WAVE(str(path))
                audio.add_tags()
                tags = audio.tags
        if field == "vocal":
            tags.delall("TXXX:VOCAL_TYPE")
            tags.add(TXXX(encoding=3, desc="VOCAL_TYPE", text=[value]))
        else:
            tags.delall("COMM")
            tags.add(COMM(encoding=3, lang="eng", desc="", text=[value]))
        audio.save()
        return

    if suffix in (".m4a", ".mp4", ".aac"):
        audio = MP4(str(path))
        if field == "vocal":
            audio["----:com.apple.iTunes:VOCAL_TYPE"] = [
                MP4FreeForm(value.encode("utf-8"))
            ]
        else:
            audio["\xa9cmt"] = [value]
        audio.save()
        return

    # Fallback easy API
    audio = MutagenFile(str(path), easy=True)
    if audio is None:
        raise RuntimeError(f"unsupported format for tags: {suffix}")
    if field == "vocal":
        audio["vocal_type"] = value
    else:
        audio["comment"] = value
    audio.save()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Classify audio with PANNs Cnn14 (AudioSet). "
            "Focus: Singing / Speech / Rapping / Humming / Choir."
        ),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="Single audio file")
    src.add_argument("--folder", type=Path, help="Folder (recursive by default)")
    src.add_argument(
        "--files-from",
        type=Path,
        help="Text file with one audio path per line (use - for stdin)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top-k overall AudioSet labels (default 10)",
    )
    parser.add_argument(
        "--focus-only",
        action="store_true",
        help="Restrict top[] to vocal/focus labels",
    )
    parser.add_argument(
        "--segment-sec",
        type=float,
        default=0.0,
        help="If >0, also emit per-segment vocal probs (window length seconds)",
    )
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=0.0,
        help="Segment hop seconds (default = segment-sec, no overlap)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=MAX_AUDIO_SECONDS,
        help=f"Max audio to load (default {MAX_AUDIO_SECONDS})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional Cnn14 .pth path (default: models\\ or auto-download)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device (default auto)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max files from folder / list (0 = all)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="With --folder, only scan the top-level directory",
    )
    parser.add_argument(
        "--write-meta",
        action="store_true",
        help="Write primary label to COMMENT or VOCAL_TYPE",
    )
    parser.add_argument(
        "--tag-field",
        choices=("comment", "vocal"),
        default="comment",
        help="Where to write tags when --write-meta (default comment)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-tag files that already have a vocal-type tag",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Batch mode: emit __gg_processed__ progress for STEM LOG "
            "(no per-file [i/total] status lines)"
        ),
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
        for i, path in enumerate(
            iter_audio_files(args.folder, recursive=not args.no_recursive), 1
        ):
            files.append(path)
            if args.limit and i >= args.limit:
                break
        if not files:
            _status(f"ERROR: no audio files under {args.folder}")
            return 1

    device = None if args.device == "auto" else args.device
    hop = args.hop_sec if args.hop_sec > 0 else None
    batch_size = _panns_batch_size()
    audio_workers = _panns_audio_workers()
    use_segments = bool(args.segment_sec and args.segment_sec > 0)

    _status(f"  PANNs tagger (Cnn14 / AudioSet) — {len(files)} file(s)")
    _status(f"  focus: {', '.join(FOCUS_LABELS)}")
    backend = load_backend(
        checkpoint=args.checkpoint,
        device=device,
        status=_status,
    )
    actual_dev = getattr(backend, "device", device or "auto")
    _status(
        f"  backend: {backend.name}  device={actual_dev}  "
        f"labels={len(backend.labels)}"
    )
    if not use_segments:
        _status(f"  batch={batch_size} decode_workers={audio_workers}")
    _status("")

    errors = 0
    skipped = 0
    tagged = 0
    total = len(files)
    started = time.perf_counter()
    batch = bool(args.batch)

    def _emit_progress(done: int) -> None:
        pct = (100.0 * done / total) if total else 0.0
        print(f"__progress__\t{pct:.1f}\t?\t{done}\t{total}\tpanns", flush=True)
        if batch and total:
            print(f"__gg_processed__\t{done}\t{total}", flush=True)

    def _handle_result(path: Path, result: dict, index: int) -> None:
        nonlocal tagged, errors
        result["index"] = index
        result["total"] = total
        if args.write_meta:
            try:
                write_vocal_metadata(
                    path,
                    str(result.get("label", "")),
                    float(result.get("score", 0.0)),
                    field=args.tag_field,
                )
                result["tagged"] = True
                tagged += 1
            except Exception as tag_exc:
                result["tag_error"] = str(tag_exc)
        if batch:
            if result.get("error") or result.get("tag_error"):
                _print_json(result)
        else:
            _print_json(result)

    def _result_from_probs(
        path: Path, audio: np.ndarray, probs: np.ndarray
    ) -> dict:
        result = probs_to_result(
            probs, backend, top_k=args.top, focus_only=args.focus_only
        )
        result["path"] = str(path.resolve())
        result["duration_sec"] = (
            float(audio.shape[0] / SAMPLE_RATE) if audio is not None and audio.size else 0.0
        )
        return result

    # Segment mode stays sequential (rare / UI checkbox hidden).
    if use_segments:
        for i, path in enumerate(files, 1):
            print(
                f"__progress__\t{(100.0 * (i - 1) / total) if total else 0.0:.1f}"
                f"\t?\t{i - 1}\t{total}\tpanns",
                flush=True,
            )
            if args.write_meta and not args.overwrite and _already_tagged(path):
                skipped += 1
                if batch:
                    print(f"__gg_processed__\t{i}\t{total}", flush=True)
                else:
                    _print_json(
                        {
                            "path": str(path.resolve()),
                            "skipped": True,
                            "reason": "already tagged",
                            "index": i,
                            "total": total,
                        }
                    )
                continue
            try:
                result = classify_file(
                    path,
                    backend,
                    top_k=args.top,
                    focus_only=args.focus_only,
                    segment_sec=args.segment_sec,
                    hop_sec=hop,
                    max_seconds=args.max_seconds,
                )
                _handle_result(path, result, i)
                if batch:
                    print(f"__gg_processed__\t{i}\t{total}", flush=True)
            except Exception as exc:
                errors += 1
                _print_json(
                    {
                        "path": str(path.resolve()),
                        "error": str(exc),
                        "index": i,
                        "total": total,
                    }
                )
                if batch:
                    print(f"__gg_processed__\t{i}\t{total}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor

        # Pre-filter skips so batches stay full of work.
        to_run: list[tuple[int, Path]] = []
        for i, path in enumerate(files, 1):
            if args.write_meta and not args.overwrite and _already_tagged(path):
                skipped += 1
                if not batch:
                    _print_json(
                        {
                            "path": str(path.resolve()),
                            "skipped": True,
                            "reason": "already tagged",
                            "index": i,
                            "total": total,
                        }
                    )
                continue
            to_run.append((i, path))

        done_count = skipped
        if skipped:
            _emit_progress(done_count)

        def _safe_load(item: tuple[int, Path]):
            idx, path = item
            try:
                return idx, path, load_mono_32k(path, max_seconds=args.max_seconds), None
            except Exception as exc:
                return idx, path, None, str(exc)

        def _load_chunk(items: list[tuple[int, Path]]):
            return list(pool.map(_safe_load, items))

        def _bump() -> None:
            nonlocal done_count
            done_count += 1
            _emit_progress(done_count)

        def _infer_chunk(loaded) -> None:
            nonlocal errors
            ok_idx: list[int] = []
            ok_paths: list[Path] = []
            ok_audios: list[np.ndarray] = []
            for idx, path, audio, err in loaded:
                if err is not None or audio is None:
                    errors += 1
                    _print_json(
                        {
                            "path": str(path.resolve()),
                            "error": err or "load failed",
                            "index": idx,
                            "total": total,
                        }
                    )
                    _bump()
                    continue
                ok_idx.append(idx)
                ok_paths.append(path)
                ok_audios.append(audio)
            if not ok_audios:
                return
            predict_batch = getattr(backend, "predict_batch", None)
            try:
                if callable(predict_batch):
                    probs_batch = predict_batch(ok_audios)
                else:
                    probs_batch = np.stack(
                        [backend.predict(a) for a in ok_audios], axis=0
                    )
            except Exception as exc:
                _status(f"  [warn] batch infer failed ({exc}); fell back per-file")
                for idx, path, audio in zip(ok_idx, ok_paths, ok_audios):
                    try:
                        probs = backend.predict(audio)
                        result = _result_from_probs(path, audio, probs)
                        _handle_result(path, result, idx)
                    except Exception as exc2:
                        errors += 1
                        _print_json(
                            {
                                "path": str(path.resolve()),
                                "error": str(exc2),
                                "index": idx,
                                "total": total,
                            }
                        )
                    _bump()
                return
            for idx, path, audio, probs in zip(
                ok_idx, ok_paths, ok_audios, probs_batch
            ):
                result = _result_from_probs(path, audio, probs)
                _handle_result(path, result, idx)
                _bump()

        with ThreadPoolExecutor(max_workers=audio_workers) as pool:
            starts = list(range(0, len(to_run), batch_size))
            next_fut = None
            for i, start in enumerate(starts):
                chunk = to_run[start : start + batch_size]
                if next_fut is not None:
                    loaded = next_fut.result()
                    next_fut = None
                else:
                    loaded = _load_chunk(chunk)
                if i + 1 < len(starts):
                    next_chunk = to_run[
                        starts[i + 1] : starts[i + 1] + batch_size
                    ]
                    next_fut = pool.submit(_load_chunk, next_chunk)
                _infer_chunk(loaded)

    print(f"__progress__\t100.0\t0\t{total}\t{total}\tpanns", flush=True)
    if batch and total:
        print(f"__gg_processed__\t{total}\t{total}", flush=True)
    elapsed = time.perf_counter() - started
    peak = None
    try:
        import torch

        if torch.cuda.is_available():
            peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except Exception:
        peak = None

    extras: list[str] = []
    if not args.write_meta:
        extras.append("METADATA UNTOUCHED (--write-meta off)")
    _print_vocal_summary(
        elapsed=elapsed,
        files=total,
        tagged=tagged if args.write_meta else None,
        skipped=skipped if args.write_meta else (skipped if skipped else None),
        errors=errors,
        peak_vram_gb=peak,
        extra_lines=extras or None,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
