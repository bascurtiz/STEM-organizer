"""Shared onnxruntime session helpers (CUDA / DirectML / CPU).

Default ship path prefers ``CUDAExecutionProvider`` (onnxruntime-gpu) when a
real NVIDIA GPU is present. ``DmlExecutionProvider`` remains a fallback when
the ORT build exposes it (e.g. experimental dual installs). Windows CUDA EP
needs cuDNN/cublas DLLs from the ``nvidia-*-cu12`` wheels on PATH /
``os.add_dll_directory`` before the first InferenceSession.
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


_NVIDIA_GPU_NAME: str | None | bool = False  # False = unset; None = unknown


def nvidia_gpu_name() -> str | None:
    """Best-effort display name for the first NVIDIA GPU (no torch)."""
    global _NVIDIA_GPU_NAME
    if _NVIDIA_GPU_NAME is not False:
        return _NVIDIA_GPU_NAME  # type: ignore[return-value]

    name: str | None = None
    try:
        import subprocess

        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0:
            line = (out.stdout or "").strip().splitlines()
            if line:
                cand = line[0].strip()
                if cand:
                    name = cand
    except Exception:
        pass

    if name is None and os.name == "nt":
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
            if out.returncode == 0:
                for line in (out.stdout or "").splitlines():
                    cand = line.strip()
                    if cand and "nvidia" in cand.lower():
                        name = cand
                        break
        except Exception:
            pass

    _NVIDIA_GPU_NAME = name
    return name


def classify_device_label(use_cuda: bool = True) -> str:
    """Status-bar text for the Classify compute device (ORT, not torch).

    Prefer the CUDA GPU product name when CUDA EP is usable and GPU is
    requested; else DirectML; else CPU. ``use_cuda`` is the UI "Use GPU" flag.
    """
    if not use_cuda:
        return "Device: CPU"
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        return "Device: CPU"

    if "CUDAExecutionProvider" in available and cuda_ep_usable():
        name = nvidia_gpu_name()
        if name:
            return f"Device: {name}"
        return "Device: GPU (CUDA)"

    if "DmlExecutionProvider" in available:
        return "Device: DirectML"

    return "Device: CPU"


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
        "nvidia-cufft-cu12",
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


def _provider_name(p) -> str:
    return p[0] if isinstance(p, tuple) else p


def ort_providers(device: str = "") -> list:
    """Provider list for InferenceSession.

    ``device='cpu'``, ``STEM_ORT_FORCE_CPU=1``, or ``STEM_ORT_CUDA=0`` → CPU only.
    Otherwise prefer CUDA when a real NVIDIA GPU + CUDA EP exist, else DirectML
    when available, else CPU.
    """
    d = (device or "").strip().lower()
    if (
        d in ("cpu",)
        or os.environ.get("STEM_ORT_FORCE_CPU", "").strip() == "1"
        or os.environ.get("STEM_ORT_CUDA", "1").strip() == "0"
    ):
        return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]

    out: list = []
    if "CUDAExecutionProvider" in available and cuda_ep_usable():
        ensure_nvidia_cuda_dlls()
        out.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in available:
        out.append("DmlExecutionProvider")
    out.append("CPUExecutionProvider")

    seen: set[str] = set()
    uniq: list = []
    for p in out:
        name = _provider_name(p)
        if name in seen:
            continue
        seen.add(name)
        uniq.append(p)
    return uniq


def create_ort_session(model_path: str | os.PathLike, *, device: str = "", **kwargs: Any):
    """Create InferenceSession with CUDA / DirectML / CPU providers.

    Caps ORT CPU thread pools so they do not multiply against Genre/Gender
    decode ThreadPools (otherwise batch mode freezes the desktop). GPU EPs
    (CUDA or DirectML) skip the half-core intra_op ceiling.
    """
    import onnxruntime as ort

    providers = ort_providers(device)
    if any(_provider_name(p) == "CUDAExecutionProvider" for p in providers):
        ensure_nvidia_cuda_dlls()
    sess_options = kwargs.pop("sess_options", None)
    if sess_options is None:
        sess_options = ort.SessionOptions()
    # Always tame inter-op; CPU EP also needs a hard intra_op ceiling.
    try:
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.inter_op_num_threads = 1
        using_gpu = any(
            _provider_name(p) in ("DmlExecutionProvider", "CUDAExecutionProvider")
            for p in providers
        )
        if not using_gpu:
            # Leave headroom for FE ThreadPools / the UI process.
            cpu_n = max(1, os.cpu_count() or 4)
            env_intra = os.environ.get("STEM_ORT_INTRA_OP", "").strip()
            if env_intra:
                intra = max(1, int(env_intra))
            else:
                intra = max(1, cpu_n // 2)
            sess_options.intra_op_num_threads = intra
    except Exception:
        pass
    return ort.InferenceSession(
        str(model_path), sess_options=sess_options, providers=providers, **kwargs
    )
