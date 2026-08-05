"""Key Detect QThread — runs key_tagger/key_tagger.py and streams LOG lines."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from ffmpeg_bootstrap import subprocess_kwargs
from tagger_launch import (
    build_tagger_command,
    key_tagger_dir,
    key_tagger_script,
    missing_tagger_python_hint,
    resolve_tagger_python,
    tagger_subprocess_env,
)

from ..io_tune import ensure_tuned
from .tagger_worker import _TAGGER_DECODE_NOISE_RE, format_tagger_exit, gg_log_tag

_PROGRESS_RE = re.compile(
    r"^__progress__[\t ]+(?P<pct>[\d.]+)[\t ]+(?P<eta>\S+)[\t ]+"
    r"(?P<n>\d+)[\t ]+(?P<total>\d+)"
)
_GG_PROCESSED_RE = re.compile(
    r"^__gg_processed__[\t ]+(?P<n>\d+)[\t ]+(?P<total>\d+)"
)
_BLAS_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class KeyWorker(QThread):
    """Run MusicalKeyCNN key detection and stream LOG-friendly output."""

    log_line = Signal(str, str)
    progress = Signal(float, object, int, int, str)
    processed = Signal(int, int)
    status = Signal(str)
    finished_ok = Signal(str)

    def __init__(
        self,
        input_dir: str,
        *,
        include_subfolders: bool = True,
        write_meta: bool = True,
        overwrite_tags: bool = False,
        tag_field: str = "key",  # comment | key
        batch_mode: bool = True,
        files_from: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._input_dir = input_dir
        self._include_subfolders = include_subfolders
        self._write_meta = write_meta
        self._overwrite_tags = overwrite_tags
        self._tag_field = tag_field if tag_field in ("comment", "key") else "key"
        self._batch_mode = bool(batch_mode)
        self._files_from = files_from
        self._proc: Optional[subprocess.Popen] = None
        self._stop_requested = False
        self._final_status = "Done"
        self._progress_t0 = 0.0

    def stop(self) -> None:
        self._stop_requested = True
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass

    def run(self) -> None:  # noqa: N802
        self.log_line.emit("  Starting tagger process…", "info")
        tagger_dir = key_tagger_dir()
        script = key_tagger_script()
        python = resolve_tagger_python()
        if not script.is_file():
            self.log_line.emit(
                f"Key tagger not found:\n{script}\n\n"
                "Expected folder: key_tagger\\ beside STEM organizer.",
                "err",
            )
            self.finished_ok.emit("Failed — tagger missing")
            return
        if python is None or not python.is_file():
            self.log_line.emit(
                f"Key Detect Python not found.\n\n{missing_tagger_python_hint()}",
                "err",
            )
            self.finished_ok.emit("Failed — Python missing")
            return

        ckpt_onnx = tagger_dir / "checkpoints" / "nf50-q05-221125.onnx"
        ckpt_pt = tagger_dir / "checkpoints" / "nf50-q05-221125.pt"
        ckpt = ckpt_onnx if ckpt_onnx.is_file() else ckpt_pt
        input_dir = str(Path(self._input_dir).expanduser().resolve())
        script_args: list[str] = [
            "--mode",
            "batch" if self._batch_mode else "per_file",
            "--checkpoint",
            str(ckpt),
        ]
        if self._files_from:
            script_args.extend(["--files-from", self._files_from])
        else:
            script_args.extend(["--folder", input_dir])
            if not self._include_subfolders:
                script_args.append("--no-recursive")
        if self._write_meta:
            script_args.append("--write-meta")
            script_args.extend(["--tag-field", self._tag_field])
        if self._overwrite_tags:
            script_args.append("--overwrite")
        # Batch: no per-file LOG (Genre-style progress counter only).
        if self._batch_mode:
            script_args.extend(["--log-pace", "0"])

        on_gpu = False
        try:
            from ort_util import cuda_ep_usable, nvidia_gpu_present

            on_gpu = bool(
                os.environ.get("STEM_ORT_CUDA", "1").strip() != "0"
                and cuda_ep_usable()
                and nvidia_gpu_present()
            )
        except Exception:
            on_gpu = False
        script_args.extend(["--device", "cuda" if on_gpu else "cpu"])

        env = tagger_subprocess_env()
        env.update(_BLAS_ENV)
        root = tagger_dir.parent
        pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = _join_pp(root, pp)

        if self._batch_mode:

            def _tune_log(msg: str, tag: str = "info") -> None:
                self.log_line.emit(msg, tag)

            hint = ensure_tuned(
                input_dir,
                workload="key",
                log=_tune_log,
                inference_on_gpu=on_gpu,
            )
            env.setdefault("KEY_CQT_WORKERS", str(hint.audio_workers))

        self._progress_t0 = time.monotonic()
        try:
            self._proc = subprocess.Popen(
                build_tagger_command(script, *script_args),
                env=env,
                cwd=str(tagger_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **subprocess_kwargs(),
            )
            assert self._proc.stdout is not None
            for line in self._iter_lines(self._proc.stdout):
                if self._stop_requested:
                    break
                self._handle_line(line)
            self._proc.wait()
            if self._stop_requested:
                self.log_line.emit("[tagger stopped]", "warn")
                self._final_status = "Stopped"
            elif self._proc.returncode == 0:
                self.progress.emit(100.0, 0.0, 0, 0, "")
                self._final_status = "Done"
            else:
                detail = format_tagger_exit(self._proc.returncode)
                self.log_line.emit(f"[tagger exited: {detail}]", "warn")
                self._final_status = f"Failed (exit {self._proc.returncode})"
        except Exception as exc:
            if self._stop_requested:
                self.log_line.emit("[tagger stopped]", "warn")
                self._final_status = "Stopped"
            else:
                self.log_line.emit(str(exc), "err")
                self._final_status = "Failed"
        finally:
            self._proc = None
            self._stop_requested = False
            self.finished_ok.emit(self._final_status)
            if self._files_from:
                try:
                    Path(self._files_from).unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _iter_lines(stream):
        # readline() — not read(1024). Block reads coalesce many LOG lines.
        while True:
            line = stream.readline()
            if not line:
                break
            yield line.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")

    def _handle_line(self, line: str) -> None:
        bare = (line or "").strip()
        if self._stop_requested:
            return
        if not bare:
            self.log_line.emit("", "info")
            return
        if _TAGGER_DECODE_NOISE_RE.search(bare):
            return

        m = _GG_PROCESSED_RE.match(bare)
        if m:
            try:
                self.processed.emit(int(m.group("n")), int(m.group("total")))
            except ValueError:
                pass
            return

        m = _PROGRESS_RE.match(bare)
        if m:
            try:
                pct = float(m.group("pct"))
                n = int(m.group("n"))
                total = int(m.group("total"))
            except ValueError:
                return
            eta = None
            if n > 0 and total > n and self._progress_t0:
                elapsed = time.monotonic() - self._progress_t0
                eta = elapsed / n * (total - n)
            elif total > 0 and n >= total:
                eta = 0.0
            self.progress.emit(pct, eta, n, total, "key")
            # Batch: same Live "Processed: n/total" line as Genre/Gender.
            self.processed.emit(n, total)
            return

        if bare == "DONE":
            self.log_line.emit("DONE", "ok")
            return

        # Badge lines: "  Db / C# 87%" → gg_result (Per-file live results).
        # Single key chip — no blank after (blank only for 2+ badges elsewhere).
        if re.match(r"^\s*.+\s+\d+%\s*$", bare) and re.search(r"\d+%", bare):
            if not bare.lower().startswith(("error", "tagged", "total", "files")):
                self.log_line.emit(
                    bare if bare.startswith(" ") else f"  {bare}",
                    "gg_result",
                )
                return

        if "[skip existing]" in bare.lower() or bare.lower().startswith("[skip"):
            self.log_line.emit(
                bare if bare.startswith("  ") else f"  {bare}",
                "warn",
            )
            return

        if bare.startswith("Skipped") or bare.startswith("  Skipped"):
            self.log_line.emit(
                bare if bare.startswith("  ") else f"  {bare}",
                "warn",
            )
            return

        # Keep leading indent from tagger (Loading libraries / Scanning / …).
        self.log_line.emit(line, gg_log_tag(bare))


def _join_pp(root: Path, existing: str) -> str:
    import os

    parts = [str(root)]
    if existing.strip():
        parts.append(existing)
    return os.pathsep.join(parts)
