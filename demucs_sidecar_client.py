"""Parent-side HTDemucs sidecar client (GUI process).

Spawns ``demucs_sidecar`` in a child process so ORT session create / run never
holds this process's GIL. Duck-types ``DemucsOnnxModel`` for classify_backend.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from multiprocessing import shared_memory

from demucs_onnx import DEFAULT_OVERLAP, SAMPLERATE, SEGMENT, SOURCES

_CLIENT: "DemucsSidecarClient | None" = None
_CLIENT_LOCK = threading.Lock()


def _subprocess_kwargs() -> dict:
    try:
        from ffmpeg_bootstrap import subprocess_kwargs

        return dict(subprocess_kwargs())
    except Exception:
        if sys.platform == "win32":
            return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
        return {}


def _sidecar_command() -> list[str]:
    frozen = bool(getattr(sys, "frozen", False))
    exe = Path(sys.executable)
    if frozen:
        return [str(exe), "--run-demucs-sidecar"]
    here = Path(__file__).resolve().parent
    return [str(exe), "-u", str(here / "demucs_sidecar.py")]


def _sidecar_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["STEM_DEMUCS_CHILD"] = "1"
    env["STEM_ALLOW_MULTI"] = "1"
    env["STEM_TAGGER_CHILD"] = "1"
    return env


class DemucsSidecarClient:
    """One long-lived child owning the HTDemucs ORT session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._ready = False
        self._warming = False
        self._ready_event = threading.Event()
        self._device = "cpu"
        self._providers: list[str] = []
        self._req_id = 0

    @property
    def ready(self) -> bool:
        return self._ready and self._proc is not None and self._proc.poll() is None

    def ensure_started(self) -> bool:
        with self._lock:
            return self._ensure_started_unlocked()

    def _ensure_started_unlocked(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        self._ready = False
        self._ready_event.clear()
        try:
            self._proc = subprocess.Popen(
                _sidecar_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=_sidecar_env(),
                cwd=str(Path(__file__).resolve().parent),
                **_subprocess_kwargs(),
            )
        except Exception:
            self._proc = None
            return False
        hello = self._recv_line_unlocked()
        if not hello or hello.get("evt") != "hello":
            self._kill_unlocked()
            return False
        return True

    def warm(self, *, prefer_gpu: bool = True) -> bool:
        with self._lock:
            if not self._ensure_started_unlocked():
                return False
            if self._ready:
                self._ready_event.set()
                return True
            return self._warm_unlocked(prefer_gpu=prefer_gpu)

    def _warm_unlocked(self, *, prefer_gpu: bool) -> bool:
        self._warming = True
        try:
            resp = self._request_unlocked(
                {"cmd": "warm", "prefer_gpu": bool(prefer_gpu)},
            )
        finally:
            self._warming = False
        if not resp or resp.get("evt") != "ready" or not resp.get("ok"):
            return False
        self._ready = True
        self._providers = list(resp.get("providers") or [])
        self._device = str(resp.get("device") or "cpu")
        self._ready_event.set()
        return True

    def wait_ready(self, timeout_s: float | None = 300.0) -> bool:
        if self.ready:
            return True
        # Background warm in progress — wait without starting a second warm.
        if self._warming:
            ok = self._ready_event.wait(timeout=timeout_s)
            return bool(ok and self.ready)
        return self.warm(prefer_gpu=True)

    def separate_numpy(self, mix: np.ndarray, *, overlap: float = DEFAULT_OVERLAP) -> np.ndarray:
        arr = np.asarray(mix, dtype=np.float32)
        squeezed = False
        if arr.ndim == 2:
            arr = arr[np.newaxis]
            squeezed = True
        if arr.ndim != 3 or arr.shape[1] != 2:
            raise ValueError(f"expected (B,2,T) or (2,T), got {arr.shape}")
        b, _ch, t = arr.shape
        out_shape = (b, len(SOURCES), 2, t)
        in_shm = out_shm = None
        try:
            in_shm = shared_memory.SharedMemory(create=True, size=int(arr.nbytes))
            out_nbytes = int(np.prod(out_shape)) * 4
            out_shm = shared_memory.SharedMemory(create=True, size=out_nbytes)
            np.ndarray(arr.shape, dtype=np.float32, buffer=in_shm.buf)[:] = arr
            with self._lock:
                if not self._ensure_started_unlocked():
                    raise RuntimeError("Demucs sidecar not started")
                if not self._ready:
                    if not self._warm_unlocked(prefer_gpu=True):
                        raise RuntimeError("Demucs sidecar warm failed")
                resp = self._request_unlocked(
                    {
                        "cmd": "separate",
                        "in_name": in_shm.name,
                        "out_name": out_shm.name,
                        "in_shape": list(arr.shape),
                        "out_shape": list(out_shape),
                        "dtype": "float32",
                        "overlap": float(overlap),
                    },
                )
            if not resp or resp.get("evt") != "ok":
                raise RuntimeError((resp or {}).get("msg") or "Demucs separate failed")
            out = np.ndarray(out_shape, dtype=np.float32, buffer=out_shm.buf).copy()
        finally:
            for shm in (in_shm, out_shm):
                if shm is None:
                    continue
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        if squeezed:
            return out[0]
        return out

    def shutdown(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            try:
                if self._proc.poll() is None and self._proc.stdin:
                    self._request_unlocked({"cmd": "shutdown"})
            except Exception:
                pass
            self._kill_unlocked()

    def _kill_unlocked(self) -> None:
        proc = self._proc
        self._proc = None
        self._ready = False
        self._ready_event.clear()
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        except Exception:
            pass

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _request_unlocked(self, payload: dict[str, Any]) -> dict | None:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return None
        req_id = self._next_id()
        payload = dict(payload)
        payload["id"] = req_id
        try:
            self._proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._proc.stdin.flush()
        except Exception:
            return None
        while True:
            msg = self._recv_line_unlocked()
            if msg is None:
                return None
            if msg.get("id") == req_id:
                return msg

    def _recv_line_unlocked(self) -> dict | None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        while True:
            line = proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue


class DemucsSidecarModel:
    """Duck-types DemucsOnnxModel; inference runs in the sidecar child."""

    sources = SOURCES
    samplerate = SAMPLERATE
    audio_channels = 2
    segment = SEGMENT
    backend = "onnx"

    def __init__(self, client: DemucsSidecarClient, *, prefer_gpu: bool = True):
        self._client = client
        self.prefer_gpu = bool(prefer_gpu)
        self._device = "cuda" if prefer_gpu else "cpu"
        # Block the Classify/SDR worker thread until child is warm — GUI stays free.
        if not client.wait_ready(timeout_s=300.0):
            if not client.warm(prefer_gpu=prefer_gpu):
                raise RuntimeError("Demucs sidecar failed to warm")
        self._device = client._device or self._device
        self.prefer_gpu = "CUDAExecutionProvider" in (client._providers or [])

    def eval(self):
        return self

    def to(self, device: str):
        want_gpu = (device or "").strip().lower() not in ("cpu", "")
        self._device = "cuda" if want_gpu and self.prefer_gpu else "cpu"
        return self

    def cpu(self):
        return self.to("cpu")

    def separate_numpy(self, mix: np.ndarray, *, overlap: float = DEFAULT_OVERLAP) -> np.ndarray:
        return self._client.separate_numpy(mix, overlap=overlap)


def get_client() -> DemucsSidecarClient:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = DemucsSidecarClient()
            atexit.register(_CLIENT.shutdown)
        return _CLIENT


def demucs_sidecar_ready() -> bool:
    c = _CLIENT
    return bool(c is not None and c.ready)


def start_demucs_sidecar_warm(*, prefer_gpu: bool = True) -> None:
    """Spawn sidecar (if needed) and warm HTDemucs in the child process."""
    if os.environ.get("STEM_DEMUCS_IDLE_WARM", "1").strip() == "0":
        return
    if os.environ.get("STEM_DEMUCS_SIDECAR", "1").strip() == "0":
        return
    if os.environ.get("STEM_ORT_CUDA", "1").strip() == "0":
        return
    if os.environ.get("STEM_ORT_FORCE_CPU", "").strip() == "1":
        return
    if not prefer_gpu:
        return

    client = get_client()
    if not client.ensure_started():
        return

    def _run() -> None:
        try:
            client.warm(prefer_gpu=True)
        except Exception:
            pass

    threading.Thread(target=_run, name="demucs-sidecar-warm", daemon=True).start()


def load_sidecar_model(*, prefer_gpu: bool = True) -> DemucsSidecarModel:
    client = get_client()
    if not client.ensure_started():
        raise RuntimeError("Could not start Demucs sidecar process")
    return DemucsSidecarModel(client, prefer_gpu=prefer_gpu)
