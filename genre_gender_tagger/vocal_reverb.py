"""Lightweight dry/wet vocal reverb classifier (mel-CNN).

Ships as models/vocal_reverb.pt — no Whisper / Hugging Face at runtime.
Train with train_vocal_reverb.py from reverb_data/dry and reverb_data/wet.
"""
from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

try:
    import torch
    from torch import nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

MODEL_NAME = "vocal_reverb.pt"
CLASS_NAMES = ("dry", "wet")

DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "n_mels": 64,
    "n_fft": 1024,
    "hop_length": 256,
    "clip_seconds": 4.0,
    "channels": (32, 64, 128),
}


if torch is not None:

    class MelReverbNet(nn.Module):
        """Small Conv2d stack over log-mel → dry/wet logits."""

        def __init__(
            self,
            n_mels: int = 64,
            channels: tuple[int, ...] = (32, 64, 128),
            n_classes: int = 2,
        ) -> None:
            super().__init__()
            layers: list = []
            in_ch = 1
            for out_ch in channels:
                layers.extend(
                    [
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.GELU(),
                        nn.MaxPool2d(2),
                    ]
                )
                in_ch = out_ch
            self.features = nn.Sequential(*layers)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_ch, n_classes),
            )
            self.n_mels = n_mels

        def forward(self, x):
            # x: (B, 1, n_mels, time)
            return self.head(self.features(x))

else:
    MelReverbNet = None  # type: ignore[misc, assignment]


class VocalReverbRouterOnnx:
    """ONNX Runtime drop-in replacement for VocalReverbRouter.

    Same public API (``predict``, ``predict_many``, ``cfg``, ``classes``,
    ``device``) so callers in genre_gender_tagger.py are unchanged. The
    mel-spectrogram + crop prep is reused verbatim from the torch path
    (pure numpy/librosa); only the Conv2d forward is replaced by an ORT session.

    The model expects input ``(B, 1, n_mels, target_frames)`` float32 and
    returns ``(B, n_classes)`` logits. The channel dim is added here.
    """

    def __init__(self, checkpoint: Path, onnx_path: Path | None = None,
                 device: str | None = None, status=print):
        import json

        if onnx_path is None:
            onnx_path = Path(checkpoint).with_suffix(".onnx")
        self.onnx_path = Path(onnx_path)

        # Config/classes come from a sidecar JSON next to the .onnx (so the
        # .pt need not ship in a torch-free build). Fall back to the .pt only
        # if the sidecar is absent (transition safety).
        sidecar = self.onnx_path.with_suffix(".config.json")
        cfg = dict(DEFAULT_CONFIG)
        classes = CLASS_NAMES
        if sidecar.is_file():
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            cfg.update(saved.get("config") or {})
            classes = tuple(saved.get("classes") or CLASS_NAMES)
        elif torch is not None and Path(checkpoint).is_file():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            cfg.update(saved.get("config") or {})
            classes = tuple(saved.get("classes") or CLASS_NAMES)
        else:
            raise FileNotFoundError(
                f"Neither {sidecar.name} nor {Path(checkpoint).name} found beside "
                f"{self.onnx_path}. Re-run export_vocal_reverb.py to generate the sidecar."
            )
        # channels may be a list (JSON) -> normalize to tuple
        if isinstance(cfg.get("channels"), list):
            cfg["channels"] = tuple(cfg["channels"])
        self.cfg = cfg
        self.classes = classes

        status(f"  loading {self.onnx_path.name} (onnxruntime) ...")

        try:
            from ort_util import create_ort_session
        except ImportError:
            # Same folder as this script when frozen next to genre_gender_tagger/
            import sys
            from pathlib import Path as _P

            root = _P(__file__).resolve().parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from ort_util import create_ort_session

        self.session = create_ort_session(self.onnx_path, device=device or "cpu")
        # 'device' kept for API parity; ONNX provider choice is the real knob.
        self.device = self.session.get_providers()[0]

        self.target_frames = frames_for_clip(cfg)
        self._n_crops = 3

    # --- mel/crop prep is identical to the torch router (shared helper) ---
    def _crops_for_file(self, filename: str) -> list[np.ndarray]:
        return _reverb_crops_for_file(self, filename)

    def _result_from_logits(self, logits: np.ndarray) -> dict:
        # logits: (n_classes,) float — mean over crops already applied by caller.
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        class_probs = dict(zip(self.classes, probs.tolist(), strict=True))
        dry_p = float(class_probs.get("dry", 0.0))
        wet_p = float(class_probs.get("wet", 0.0))
        if wet_p >= dry_p:
            label, confidence = "wet", wet_p
        else:
            label, confidence = "dry", dry_p
        return {
            "reverb": label,
            "reverb_confidence": confidence,
            "wet": wet_p,
            "dry": dry_p,
        }

    def _forward(self, crops: np.ndarray) -> np.ndarray:
        """crops: (N, n_mels, target_frames) -> logits (N, n_classes)."""
        mel = crops[:, np.newaxis, :, :].astype(np.float32)  # (N,1,n_mels,time)
        return self.session.run(["logits"], {"mel": mel})[0]

    def predict(self, filename: str) -> dict:
        crops = self._crops_for_file(filename)
        logits = self._forward(np.stack(crops, axis=0)).mean(axis=0)
        return self._result_from_logits(logits)

    def predict_many(
        self,
        filenames: list[str],
        *,
        gpu_batch_size: int = 64,
        num_workers: int = 8,
    ) -> list[dict | BaseException]:
        """Parallel CPU decode/mel, then batched ONNX forwards.

        Mirrors VocalReverbRouter.predict_many semantics: one entry per input
        path (result dict, or the exception raised).
        """
        from concurrent.futures import ThreadPoolExecutor

        n = len(filenames)
        out: list[dict | BaseException | None] = [None] * n
        if n == 0:
            return []

        def _prep(item):
            index, path = item
            try:
                return index, self._crops_for_file(path), None
            except BaseException as exc:
                return index, None, exc

        workers = max(1, min(num_workers, n))
        prepared = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, crops, err in pool.map(_prep, enumerate(filenames)):
                if err is not None:
                    out[index] = err
                else:
                    prepared.append((index, crops))
        if not prepared:
            return [e if e is not None else RuntimeError("no crops") for e in out]

        crop_tensors, owners = [], []
        for index, crops in prepared:
            for crop in crops:
                crop_tensors.append(crop)
                owners.append(index)

        logit_sum: dict[int, np.ndarray] = {}
        logit_count: dict[int, int] = {}
        bs = max(1, int(gpu_batch_size))
        for start in range(0, len(crop_tensors), bs):
            chunk = crop_tensors[start:start + bs]
            own = owners[start:start + bs]
            logits = self._forward(np.stack(chunk, axis=0))
            for j, index in enumerate(own):
                if index in logit_sum:
                    logit_sum[index] = logit_sum[index] + logits[j]
                    logit_count[index] += 1
                else:
                    logit_sum[index] = logits[j]
                    logit_count[index] = 1

        for index, total in logit_sum.items():
            out[index] = self._result_from_logits(total / logit_count[index])

        return [
            e if e is not None else RuntimeError("reverb predict missing")
            for e in out
        ]


def load_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    """Mono float32 @ sample_rate. soundfile first, librosa fallback for bad FLACs."""
    path = Path(path)
    audio = None
    sr = sample_rate
    try:
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        audio = data.mean(axis=1)
    except Exception:
        # Corrupt / sync-lost FLACs often fail in libsndfile; librosa/audioread
        # can still decode some of them. If both fail, let caller handle.
        audio, sr = librosa.load(str(path), sr=None, mono=True, dtype=np.float32)

    if audio is None or audio.size == 0:
        raise ValueError(f"empty audio: {path}")

    if int(sr) != int(sample_rate):
        audio = librosa.resample(
            audio, orig_sr=int(sr), target_sr=sample_rate, res_type="soxr_hq"
        )
    return np.asarray(audio, dtype=np.float32)


def probe_audio(path: str | Path) -> None:
    """Raise if the file cannot be opened/decoded (short read)."""
    path = Path(path)
    try:
        with sf.SoundFile(str(path)) as handle:
            n = min(int(handle.frames), 4096)
            if n > 0:
                handle.read(frames=n, dtype="float32", always_2d=True)
            return
    except Exception:
        pass
    # Fallback decode of a short slice
    audio, _sr = librosa.load(str(path), sr=None, mono=True, duration=0.25)
    if audio is None or np.asarray(audio).size == 0:
        raise ValueError(f"unreadable or empty: {path}")


def audio_to_logmel(
    audio: np.ndarray,
    *,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Return log-mel (n_mels, time) float32."""
    if audio.size == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # Stabilize empty / silent clips
    if not np.isfinite(logmel).all():
        logmel = np.nan_to_num(logmel, nan=-80.0, posinf=0.0, neginf=-80.0)
    return logmel


def crop_or_pad_logmel(
    logmel: np.ndarray, target_frames: int, *, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Random crop (train) or center crop / pad (infer) to target_frames."""
    n_mels, n_frames = logmel.shape
    if n_frames == target_frames:
        return logmel
    if n_frames > target_frames:
        if rng is not None:
            start = int(rng.integers(0, n_frames - target_frames + 1))
        else:
            start = max(0, (n_frames - target_frames) // 2)
        return logmel[:, start : start + target_frames]
    pad = target_frames - n_frames
    left = pad // 2
    right = pad - left
    return np.pad(logmel, ((0, 0), (left, right)), mode="constant", constant_values=-80.0)


def frames_for_clip(cfg: dict) -> int:
    sr = int(cfg["sample_rate"])
    hop = int(cfg["hop_length"])
    seconds = float(cfg["clip_seconds"])
    return max(1, int(round(seconds * sr / hop)))


def _reverb_crops_for_file(router, filename: str) -> list[np.ndarray]:
    """Decode + log-mel crops for one file (CPU).

    Shared by :class:`VocalReverbRouter` (torch) and
    :class:`VocalReverbRouterOnnx` — both expose ``cfg``, ``target_frames``,
    ``_n_crops``.
    """
    cfg = router.cfg
    audio = load_mono(filename, int(cfg["sample_rate"]))
    logmel = audio_to_logmel(
        audio,
        sample_rate=int(cfg["sample_rate"]),
        n_mels=int(cfg["n_mels"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
    )
    n_frames = logmel.shape[1]
    if n_frames <= router.target_frames:
        return [crop_or_pad_logmel(logmel, router.target_frames)]
    starts = np.linspace(
        0,
        n_frames - router.target_frames,
        num=router._n_crops,
        dtype=np.int64,
    )
    return [
        logmel[:, int(start) : int(start) + router.target_frames]
        for start in starts
    ]


class VocalReverbRouter:
    """Load vocal_reverb.pt and predict dry/wet for an audio file."""

    def __init__(self, checkpoint: Path, device: str | None = None, status=print):
        if torch is None:
            raise RuntimeError("torch is required for vocal_reverb.pt fallback")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        status(f"  loading {Path(checkpoint).name} ({self.device}) ...")
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(saved.get("config") or {})
        self.cfg = cfg
        self.classes = tuple(saved.get("classes") or CLASS_NAMES)
        channels = tuple(cfg.get("channels") or DEFAULT_CONFIG["channels"])
        self.model = MelReverbNet(
            n_mels=int(cfg["n_mels"]),
            channels=channels,
            n_classes=len(self.classes),
        ).to(self.device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.target_frames = frames_for_clip(cfg)
        self._n_crops = 3

    def _crops_for_file(self, filename: str) -> list[np.ndarray]:
        """Decode + log-mel crops for one file (CPU)."""
        return _reverb_crops_for_file(self, filename)

    def _result_from_logits(self, logits):
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        class_probs = dict(zip(self.classes, probs, strict=True))
        dry_p = float(class_probs.get("dry", 0.0))
        wet_p = float(class_probs.get("wet", 0.0))
        if wet_p >= dry_p:
            label = "wet"
            confidence = wet_p
        else:
            label = "dry"
            confidence = dry_p
        return {
            "reverb": label,
            "reverb_confidence": confidence,
            "wet": wet_p,
            "dry": dry_p,
        }

    def predict(self, filename: str) -> dict:
        with torch.inference_mode():
            crops = self._crops_for_file(filename)
            x = torch.from_numpy(np.stack(crops, axis=0)).unsqueeze(1)
            if self.device == "cuda":
                x = x.pin_memory().to(self.device, non_blocking=True)
            else:
                x = x.to(self.device)
            logits = self.model(x).mean(dim=0)
            return self._result_from_logits(logits)

    def predict_many(
        self,
        filenames: list[str],
        *,
        gpu_batch_size: int = 64,
        num_workers: int = 8,
    ) -> list[dict | BaseException]:
        """Parallel CPU decode/mel, then batched GPU (or CPU) forwards.

        Returns one entry per input path: result dict, or the exception raised.
        """
        from concurrent.futures import ThreadPoolExecutor

        n = len(filenames)
        out: list[dict | BaseException | None] = [None] * n
        if n == 0:
            return []

        def _prep(item: tuple[int, str]):
            index, path = item
            try:
                return index, self._crops_for_file(path), None
            except BaseException as exc:
                return index, None, exc

        workers = max(1, min(num_workers, n))
        prepared: list[tuple[int, list[np.ndarray]]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, crops, err in pool.map(_prep, enumerate(filenames)):
                if err is not None:
                    out[index] = err
                else:
                    prepared.append((index, crops))

        if not prepared:
            return [e if e is not None else RuntimeError("no crops") for e in out]

        crop_tensors: list[np.ndarray] = []
        owners: list[int] = []
        for index, crops in prepared:
            for crop in crops:
                crop_tensors.append(crop)
                owners.append(index)

        logit_sum: dict[int, object] = {}
        logit_count: dict[int, int] = {}
        bs = max(1, int(gpu_batch_size))

        with torch.inference_mode():
            for start in range(0, len(crop_tensors), bs):
                chunk = crop_tensors[start : start + bs]
                own = owners[start : start + bs]
                x = torch.from_numpy(np.stack(chunk, axis=0)).unsqueeze(1)
                if self.device == "cuda":
                    x = x.pin_memory().to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device)
                logits = self.model(x)
                for j, index in enumerate(own):
                    row = logits[j].detach()
                    if index in logit_sum:
                        logit_sum[index] = logit_sum[index] + row
                        logit_count[index] += 1
                    else:
                        logit_sum[index] = row
                        logit_count[index] = 1

        for index, total in logit_sum.items():
            out[index] = self._result_from_logits(total / logit_count[index])

        return [
            e if e is not None else RuntimeError("reverb predict missing")
            for e in out
        ]


def ensure_vocal_reverb(model_dir: Path, status=print) -> Path:
    """Return path to vocal_reverb.pt or exit with train instructions."""
    model_dir = Path(model_dir)
    path = model_dir / MODEL_NAME
    if path.exists() and path.stat().st_size > 0:
        return path

    data_hint = Path(__file__).resolve().parent / "reverb_data"
    raise SystemExit(
        f"\nERROR: missing {path.name}\n"
        f"  expected: {path}\n\n"
        f"Train it from dry/wet vocal packs:\n"
        f"  1. Put audio in:\n"
        f"       {data_hint / 'dry'}\n"
        f"       {data_hint / 'wet'}\n"
        f"  2. Activate genre_gender_tagger\\venv and run:\n"
        f"       python train_vocal_reverb.py\n"
        f"  3. Re-run Gender tagging.\n"
    )


def load_vocal_reverb(
    model_dir: Path,
    status=print,
    device: str | None = None,
):
    """Load the reverb router — ONNX by default, torch fallback.

    Selection order (mirrors the rest of the ONNX migration):
      1. ``STEM_ONNX=0`` env var  -> force the legacy torch router.
      2. ``vocal_reverb.onnx`` (+ config sidecar) exists AND onnxruntime imports
         -> ONNX router (DirectML/CPU). Does NOT require the .pt.
      3. otherwise -> torch router (requires vocal_reverb.pt + torch at runtime).

    Both routers share the same public API (``predict`` / ``predict_many`` /
    ``cfg`` / ``classes``), so callers are unchanged.
    """
    model_dir = Path(model_dir)
    onnx_path = model_dir / "vocal_reverb.onnx"
    # Frozen / no-GPU: default ONNX to CPU (avoids DirectML EP spam).
    onnx_device = device if device is not None else "cpu"

    if os.environ.get("STEM_ONNX", "1").strip() != "0" and onnx_path.is_file():
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            pass
        else:
            # checkpoint arg is only used as a sidecar fallback location; the
            # .pt need not exist (config comes from vocal_reverb.config.json).
            return VocalReverbRouterOnnx(
                onnx_path, onnx_path=onnx_path, device=onnx_device, status=status
            )

    if torch is None:
        raise SystemExit(
            "\nERROR: vocal_reverb.onnx missing and torch is not installed.\n"
            f"  expected ONNX at: {onnx_path}\n"
        )

    # torch fallback — requires the .pt.
    checkpoint = ensure_vocal_reverb(model_dir, status=status)
    return VocalReverbRouter(checkpoint, device=device, status=status)
