"""Shared onnxruntime session helpers (CPU / CUDA).

Windows CUDA EP needs cuDNN/cublas DLLs from the ``nvidia-*-cu12`` wheels on
PATH / ``os.add_dll_directory`` before the first InferenceSession.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_NVIDIA_DLLS_READY = False
_NVIDIA_GPU_PRESENT: bool | None = None


def nvidia_gpu_present() -> bool:
    """True if a real NVIDIA GPU is visible (not just the CUDA ORT wheel).

    ``onnxruntime-gpu`` lists ``CUDAExecutionProvider`` even on VMs with no
    GPU; sessions then quietly run on CPU while still advertising CUDA.
    """
    global _NVIDIA_GPU_PRESENT
    if _NVIDIA_GPU_PRESENT is not None:
        return _NVIDIA_GPU_PRESENT
    if os.environ.get("STEM_ORT_FORCE_CPU", "").strip() == "1":
        _NVIDIA_GPU_PRESENT = False
        return False
    if os.environ.get("STEM_ORT_ASSUME_NVIDIA", "").strip() == "1":
        _NVIDIA_GPU_PRESENT = True
        return True

    # 1) nvidia-smi -L (most reliable when driver is installed)
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0 and "GPU " in (out.stdout or ""):
            _NVIDIA_GPU_PRESENT = True
            return True
    except Exception:
        pass

    # 2) CUDA runtime device count via bundled cudart (after DLL path setup)
    try:
        ensure_nvidia_cuda_dlls()
        import ctypes

        for name in ("cudart64_12.dll", "cudart64_110.dll", "cudart64_10.dll"):
            try:
                cudart = ctypes.WinDLL(name)
            except OSError:
                continue
            count = ctypes.c_int(0)
            # cudaError_t cudaGetDeviceCount(int*)
            fn = getattr(cudart, "cudaGetDeviceCount", None)
            if fn is None:
                continue
            fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
            fn.restype = ctypes.c_int
            if int(fn(ctypes.byref(count))) == 0 and int(count.value) > 0:
                _NVIDIA_GPU_PRESENT = True
                return True
            break
    except Exception:
        pass

    # 3) Windows PnP / WMI display adapters named NVIDIA
    if os.name == "nt":
        try:
            import subprocess

            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController).Name",
                ],
                capture_output=True,
                text=True,
                timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            names = (out.stdout or "").lower()
            if out.returncode == 0 and "nvidia" in names:
                _NVIDIA_GPU_PRESENT = True
                return True
        except Exception:
            pass

    _NVIDIA_GPU_PRESENT = False
    return False


def cuda_ep_usable() -> bool:
    """ORT CUDA EP is in the wheel *and* a real NVIDIA GPU is present."""
    if not nvidia_gpu_present():
        return False
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def ensure_nvidia_cuda_dlls() -> list[str]:
    """Prepend nvidia-*/bin dirs so ORT can LoadLibrary(cudnn64_9.dll)."""
    global _NVIDIA_DLLS_READY
    added: list[str] = []
    if _NVIDIA_DLLS_READY:
        return added
    try:
        import importlib.metadata as md
    except Exception:
        _NVIDIA_DLLS_READY = True
        return added

    bins: list[Path] = []
    for name in (
        "nvidia-cudnn-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cudnn-cu13",
        "nvidia-cublas-cu13",
    ):
        try:
            dist = md.distribution(name)
        except Exception:
            continue
        for f in dist.files or []:
            s = str(f).replace("\\", "/")
            if "/bin/" in s and s.lower().endswith((".dll", ".so", ".so.1")):
                p = Path(dist.locate_file(f)).resolve().parent
                if p.is_dir() and p not in bins:
                    bins.append(p)

    # Also scan site-packages/nvidia/*/bin (frozen + editable layouts).
    try:
        import site

        roots = [Path(p) for p in site.getsitepackages()]
    except Exception:
        roots = []
    try:
        import sys

        roots.append(Path(sys.prefix) / "Lib" / "site-packages")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))
            # PyInstaller often flattens as _internal/nvidia/...
            roots.append(Path(meipass) / "nvidia")
    except Exception:
        pass
    for root in roots:
        nvidia = root if root.name == "nvidia" else root / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            if bin_dir.is_dir() and bin_dir not in bins:
                bins.append(bin_dir)

    for b in bins:
        s = str(b)
        os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(s)
            except Exception:
                pass
        added.append(s)

    _NVIDIA_DLLS_READY = True
    return added


def _cuda_available() -> bool:
    return cuda_ep_usable()


def ort_providers(device: str = "") -> list:
    """Provider list for InferenceSession.

    ``device='cpu'`` → CPU only.
    Otherwise prefer CUDA when a real NVIDIA GPU + CUDA EP exist, else CPU.
    Set ``STEM_ORT_CUDA=0`` to force CPU even when CUDA is available.
    """
    d = (device or "").strip().lower()
    if d in ("cpu",) or os.environ.get("STEM_ORT_CUDA", "1").strip() == "0":
        return ["CPUExecutionProvider"]
    if cuda_ep_usable():
        ensure_nvidia_cuda_dlls()
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_ort_session(model_path: str | os.PathLike, *, device: str = "", **kwargs: Any):
    """Create InferenceSession with CUDA-or-CPU providers."""
    import onnxruntime as ort

    providers = ort_providers(device)
    if any(
        (p == "CUDAExecutionProvider" or (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider"))
        for p in providers
    ):
        ensure_nvidia_cuda_dlls()
    sess_options = kwargs.pop("sess_options", None)
    return ort.InferenceSession(
        str(model_path), sess_options=sess_options, providers=providers, **kwargs
    )
