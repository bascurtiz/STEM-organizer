"""PANNs vocal-type tagger QThread wrapper.

Spawns ``panns_tagger/panns_tagger.py``, streams stdout, parses JSON results
and ``__progress__`` markers, emits Qt signals for the Genre & Gender tab.
"""

from __future__ import annotations

import json
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
    missing_tagger_python_hint,
    panns_tagger_dir,
    panns_tagger_script,
    resolve_tagger_python,
    tagger_subprocess_env,
)

from ..io_tune import ensure_tuned
from .tagger_worker import format_tagger_exit, gg_log_tag

_FOCUS = ("Singing", "Speech", "Rapping", "Humming", "Choir")
_PROGRESS_RE = re.compile(
    r"^__progress__[\t ]+(?P<pct>[\d.]+)[\t ]+(?P<eta>\S+)[\t ]+"
    r"(?P<n>\d+)[\t ]+(?P<total>\d+)"
)
_GG_PROCESSED_RE = re.compile(
    r"^__gg_processed__[\t ]+(?P<n>\d+)[\t ]+(?P<total>\d+)"
)
_FILE_STATUS_RE = re.compile(r"^\[\d+/\d+\]\s+")
_LIBSNDFILE_NOISE_RE = re.compile(
    r"^Warning:\s*Xing stream size"
    r"|dequantization failed"
    r"|libmpg123[/\\](?:layer3|id3)\.c"
    r"|bad RVA2 tag"
    r"|^\[.*libmpg123.*\]\s*error:",
    re.IGNORECASE,
)
# PANNs bootstrap chatter (paths / load) — keep real download / error lines.
_PANNS_MODEL_NOISE_RE = re.compile(
    r"^checkpoint:"
    r"|^loading PANNs Cnn14"
    r"|^Checkpoint path:",
    re.IGNORECASE,
)


class PannsWorker(QThread):
    """Run PANNs Cnn14 vocal-type tagging and stream LOG-friendly output."""

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
        tag_field: str = "comment",  # comment | vocal
        segment_sec: float = 0.0,
        batch_mode: bool = True,
        files_from: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._input_dir = input_dir
        self._batch_mode = batch_mode
        self._include_subfolders = include_subfolders
        self._write_meta = write_meta
        self._overwrite_tags = overwrite_tags
        self._tag_field = tag_field if tag_field in ("comment", "vocal") else "comment"
        self._segment_sec = float(segment_sec) if segment_sec and segment_sec > 0 else 0.0
        self._files_from = files_from
        self._proc: Optional[subprocess.Popen] = None
        self._stop_requested = False
        self._final_status = "Done"
        self._progress_t0 = 0.0
        self._blank_before_processed = False

    def stop(self) -> None:
        self._stop_requested = True
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass

    def _emit_processed(self, n: int, total: int) -> None:
        """Blank line once before the first Processed: n/total live line."""
        if not self._blank_before_processed:
            self.log_line.emit("", "info")
            self._blank_before_processed = True
        self.processed.emit(n, total)

    def run(self) -> None:  # noqa: N802
        tagger_dir = panns_tagger_dir()
        script = panns_tagger_script()
        python = resolve_tagger_python()
        if not script.is_file():
            self.log_line.emit(
                f"PANNs tagger not found:\n{script}\n\n"
                "Expected folder: panns_tagger\\ beside STEM organizer.",
                "err",
            )
            self.finished_ok.emit("Failed — tagger missing")
            return
        if python is None or not python.is_file():
            self.log_line.emit(
                f"PANNs Python not found.\n\n{missing_tagger_python_hint()}",
                "err",
            )
            self.finished_ok.emit("Failed — Python missing")
            return

        input_dir = str(Path(self._input_dir).expanduser().resolve())
        script_args: list[str] = [
            "--top",
            "8",
            "--focus-only",
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
        if self._segment_sec > 0:
            script_args.extend(["--segment-sec", str(self._segment_sec)])
        if self._batch_mode:
            script_args.append("--batch")

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
        if self._batch_mode and not self._segment_sec:

            def _tune_log(msg: str, tag: str = "info") -> None:
                self.log_line.emit(msg, tag)

            hint = ensure_tuned(
                input_dir,
                workload="panns",
                log=_tune_log,
                inference_on_gpu=on_gpu,
            )
            env.setdefault("PANNS_AUDIO_WORKERS", str(hint.audio_workers))
            env.setdefault("PANNS_BATCH_SIZE", str(hint.gpu_batch_size))
            if not on_gpu:
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("MKL_NUM_THREADS", "1")
                env.setdefault("OPENBLAS_NUM_THREADS", "1")
                env.setdefault("NUMEXPR_NUM_THREADS", "1")

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
        if _LIBSNDFILE_NOISE_RE.search(bare):
            return
        if _PANNS_MODEL_NOISE_RE.search(bare):
            return
        if not bare:
            if not self._batch_mode:
                self.log_line.emit("", "info")
            return

        m_proc = _GG_PROCESSED_RE.match(bare)
        if m_proc:
            try:
                n = int(m_proc.group("n"))
                total = int(m_proc.group("total"))
            except ValueError:
                return
            self._emit_processed(n, total)
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
            self.progress.emit(pct, eta, n, total, "panns")
            # Live LOG line comes from __gg_processed__ in batch mode.
            if not self._batch_mode:
                self._emit_processed(n, total)
            return

        if _FILE_STATUS_RE.match(bare):
            # Legacy "[i/n] name" status — counter is in === [i/n] name === now.
            return

        if bare.startswith("{"):
            try:
                payload = json.loads(bare)
            except json.JSONDecodeError:
                if not self._batch_mode:
                    self.log_line.emit(bare, "info")
                return
            self._emit_result(payload)
            return

        # Status from stderr-merged stream (keep leading indent from tagger).
        if bare == "DONE":
            self.log_line.emit("DONE", "ok")
            return
        self.log_line.emit(line, gg_log_tag(bare))

    def _file_header(self, name: str, payload: dict) -> str:
        try:
            i = int(payload.get("index") or 0)
            n = int(payload.get("total") or 0)
        except (TypeError, ValueError):
            i = n = 0
        if i > 0 and n > 0:
            return f"=== [{i}/{n}] {name} ==="
        return f"=== {name} ==="

    def _emit_result(self, payload: dict) -> None:
        path = str(payload.get("path", ""))
        name = Path(path).name if path else "(unknown)"
        header = self._file_header(name, payload)

        if payload.get("error") or payload.get("tag_error"):
            # Always show failures (batch + per-file).
            if payload.get("error"):
                self.log_line.emit(f"ERROR: {path or name}", "log_note_err")
                detail = str(payload.get("error") or "").strip()
                if detail:
                    self.log_line.emit(f"  {detail}", "log_note_err")
            if payload.get("tag_error"):
                if not payload.get("error"):
                    self.log_line.emit(header, "info")
                self.log_line.emit(
                    f"  Tag write failed: {payload['tag_error']}", "log_note_err"
                )
            return

        # Batch mode: one-liner progress only — no per-file blocks.
        if self._batch_mode:
            return

        if payload.get("skipped"):
            self.log_line.emit(header, "info")
            self.log_line.emit("  [skip] already tagged", "warn")
            return

        label = str(payload.get("label", "?"))
        score = float(payload.get("score", 0.0) or 0.0)
        vocal = payload.get("vocal") or {}

        self.log_line.emit(header, "info")
        # Winning badge only (no runner-ups).
        n_badges = 0
        pct = float(vocal.get(label, score) if label in vocal else score or 0.0) * 100.0
        if label in _FOCUS:
            self.log_line.emit(f"  {label} {pct:.0f}%", "gg_result")
            n_badges += 1
        else:
            self.log_line.emit(f"  {label} {pct:.0f}%", "info")

        segs = payload.get("segments") or []
        if segs:
            self.log_line.emit(f"  Segments ({len(segs)}):", "info")
            for seg in segs:
                t0 = float(seg.get("start", 0))
                t1 = float(seg.get("end", 0))
                sl = str(seg.get("label", "?"))
                ss = float(seg.get("score", 0) or 0)
                if sl in _FOCUS:
                    self.log_line.emit(
                        f"  {sl} {ss * 100:.0f}%  {t0:.1f}–{t1:.1f}s",
                        "gg_result",
                    )
                    n_badges += 1
                else:
                    self.log_line.emit(
                        f"    {t0:5.1f}–{t1:5.1f}s  {sl} {ss * 100:.0f}%",
                        "info",
                    )
        # Blank only when 2+ badge chips for this file
        if n_badges >= 2:
            self.log_line.emit("", "info")
