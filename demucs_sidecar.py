"""HTDemucs ORT sidecar process — keeps InferenceSession off the GUI GIL.

Parent speaks newline-delimited JSON on stdin; tensors travel via
``multiprocessing.shared_memory``. Launch via::

    python -u demucs_sidecar.py
    STEM-organizer.exe --run-demucs-sidecar
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from multiprocessing import shared_memory
from typing import Any


def _force_utf8_stdio() -> None:
    import io

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            buf = getattr(stream, "buffer", None)
            if buf is None:
                continue
            setattr(
                sys,
                name,
                io.TextIOWrapper(buf, encoding="utf-8", errors="replace", line_buffering=True),
            )
        except Exception:
            pass


def _reply(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _attach_array(name: str, shape: list[int], dtype: str):
    import numpy as np

    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(tuple(int(x) for x in shape), dtype=np.dtype(dtype), buffer=shm.buf)
    return shm, arr


def main() -> int:
    os.environ["STEM_DEMUCS_CHILD"] = "1"
    os.environ.setdefault("STEM_ALLOW_MULTI", "1")
    os.environ.setdefault("STEM_TAGGER_CHILD", "1")  # skip single-instance if routed via GUI entry
    _force_utf8_stdio()

    model = None
    _reply({"evt": "hello", "ok": True})

    for line in sys.stdin:
        line = (line or "").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply({"evt": "err", "msg": f"bad json: {exc}"})
            continue

        cmd = (msg.get("cmd") or "").strip().lower()
        req_id = msg.get("id")

        try:
            if cmd == "ping":
                _reply({"evt": "pong", "id": req_id, "ready": model is not None})
                continue

            if cmd == "shutdown":
                _reply({"evt": "bye", "id": req_id})
                return 0

            if cmd == "warm":
                prefer_gpu = bool(msg.get("prefer_gpu", True))
                if model is not None:
                    _reply(
                        {
                            "evt": "ready",
                            "id": req_id,
                            "ok": True,
                            "providers": list(model.session.get_providers()),
                        }
                    )
                    continue
                from demucs_onnx import DemucsOnnxModel, resolve_htdemucs_onnx

                onnx_path = resolve_htdemucs_onnx()
                if onnx_path is None:
                    _reply(
                        {
                            "evt": "ready",
                            "id": req_id,
                            "ok": False,
                            "msg": "htdemucs.onnx missing",
                        }
                    )
                    continue
                model = DemucsOnnxModel(onnx_path, prefer_gpu=prefer_gpu)
                _reply(
                    {
                        "evt": "ready",
                        "id": req_id,
                        "ok": True,
                        "providers": list(model.session.get_providers()),
                        "device": getattr(model, "_device", "cpu"),
                    }
                )
                continue

            if cmd == "separate":
                if model is None:
                    _reply({"evt": "err", "id": req_id, "msg": "not warmed"})
                    continue
                in_name = msg["in_name"]
                out_name = msg["out_name"]
                in_shape = msg["in_shape"]
                out_shape = msg["out_shape"]
                dtype = msg.get("dtype", "float32")
                overlap = float(msg.get("overlap", 0.1))
                in_shm = out_shm = None
                try:
                    in_shm, mix = _attach_array(in_name, in_shape, dtype)
                    out_shm, out = _attach_array(out_name, out_shape, dtype)
                    stems = model.separate_numpy(mix, overlap=overlap)
                    if tuple(stems.shape) != tuple(int(x) for x in out_shape):
                        raise RuntimeError(
                            f"stem shape {stems.shape} != expected {out_shape}"
                        )
                    out[:] = stems
                    _reply({"evt": "ok", "id": req_id})
                finally:
                    if in_shm is not None:
                        in_shm.close()
                    if out_shm is not None:
                        out_shm.close()
                continue

            _reply({"evt": "err", "id": req_id, "msg": f"unknown cmd {cmd!r}"})
        except Exception as exc:
            _reply(
                {
                    "evt": "err",
                    "id": req_id,
                    "msg": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc()[-2000:],
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
