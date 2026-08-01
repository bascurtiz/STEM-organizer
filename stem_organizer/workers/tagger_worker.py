"""Genre & Gender tagger QThread wrapper.

Spawns the bundled genre_gender_tagger subprocess with the GG_* env-var
protocol, streams stdout line by line, parses the progress markers and tqdm
bars, and emits Qt signals.

Frozen: host Python + site-packages\\ beside the exe (see tagger_launch).
Source: genre_gender_tagger\\venv when present.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from ffmpeg_bootstrap import subprocess_kwargs
from tagger_launch import (
    build_tagger_command,
    genre_gender_dir,
    genre_gender_script,
    missing_tagger_python_hint,
    resolve_tagger_python,
    tagger_subprocess_env,
)

from ..io_tune import ensure_tuned
from ..settings_store import SettingsStore


_TQDM_PCT_RE = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)%\|.*?\|?\s*(?P<cur>\d+)/(?P<total>\d+)"
)
_PROGRESS_LOG_RE = re.compile(
    r"^.+?:\s*\d[\d,]*/\d[\d,]*\s*\(\d+(?:\.\d+)?%\)$"
    r"|^\s*\d+(?:\.\d+)?%\|"
    r"|it/s\]"
)
_GG_RESULT_KEY_RE = re.compile(r"^(GENRE|STYLE|CONF|GENDER|REVERB):\s*(.*)$", re.IGNORECASE)
_GG_AUDIO_NAME_RE = re.compile(r"\.(flac|mp3|wav|m4a|aiff?|ogg|opus)\s*$", re.IGNORECASE)
_GG_BADGE_LINE_RE = re.compile(
    r"^\s*(female|male|dry|wet)(?:\s+\(confidence\s+[^)]+\)|\s+\d+%)?\s*$",
    re.IGNORECASE,
)
_TAGGER_DECODE_NOISE_RE = re.compile(
    r"^Warning:\s*Xing stream size"
    r"|dequantization failed"
    r"|libmpg123[/\\](?:layer3|id3)\.c"
    r"|bad RVA2 tag"
    r"|^\[.*libmpg123.*\]\s*error:",
    re.IGNORECASE,
)
# Gender model bootstrap chatter (paths / ORT providers) — keep downloads.
_GG_GENDER_MODEL_NOISE_RE = re.compile(
    r"^Loading voice-gender models"
    r"|^Models loaded \("
    r"|^\s*using bundled\b"
    r"|^\s*ONNX Runtime providers:"
    r"|^\s*using \S+ for discogs-effnet"
    r"|^\s*DirectML unavailable",
    re.IGNORECASE,
)


def gg_log_tag(line: str) -> str:
    s = (line or "").strip()
    if not s:
        return "info"
    low = s.lower()
    # Decode/file errors: small red (log_note_err), not bold body err.
    # Covers "ERROR: path", "Error : decoder…", "error opening…",
    # "SKIP (tag error): …".
    if low.startswith("error:") or low.startswith("error :") or low.startswith(
        "error opening"
    ):
        return "log_note_err"
    if low.startswith("error"):
        return "log_note_err"
    if low.startswith("skip (tag error)"):
        return "log_note_err"
    if low.startswith("[tagger exited"):
        return "err"
    if s == "DONE":
        return "ok"
    if low.startswith("[tagger") or low.startswith("stop requested"):
        return "warn"
    if re.match(r"^=== .+ Summary ===$", s):
        return "info"
    if s.startswith("===") and s.endswith("==="):
        return "info"
    if _GG_BADGE_LINE_RE.match(s) or low.startswith("(confidence"):
        return "gg_result"
    if _GG_RESULT_KEY_RE.match(s):
        return "gg_result"
    if _GG_AUDIO_NAME_RE.search(s) and ":" not in s.split()[0]:
        return "gg_file"
    if s.startswith("Tagged:") or s.startswith("  Tagged:"):
        return "ok"
    if s.startswith("  Passed:"):
        return "ok"
    if (
        "[skip existing]" in low
        or "[skip]" in low
        or s.startswith(("  Skipped", "Skipped:", "Skipped ("))
    ):
        return "warn"
    if s.startswith((
        "  Total time:", "  Files:", "  Sec/file:", "  Files/min:",
        "  Tagged:", "  Peak VRAM:", "  Results:", "  Phase timing:",
        "  Errors:", "  Lossless:", "  OK:",
    )):
        return "info"
    if s.startswith("    ") and ":" in s:
        return "warn"
    if s.startswith("Processing"):
        return "info"
    return "info"


def format_tagger_exit(code: Optional[int]) -> str:
    if code is None:
        return "unknown"
    code_u = code & 0xFFFFFFFF if code < 0 else int(code)
    if code_u == 0xC0000005:
        return (
            f"{code} ACCESS_VIOLATION (usually RAM exhausted). "
            "Batch now tags in 256-file waves — retry; already-tagged "
            "files from a partial run are kept."
        )
    if code_u == 0xC000012D:
        return f"{code} out of system resources / memory"
    return str(code)


class TaggerWorker(QThread):
    """Run the bundled genre/gender tagger subprocess and stream its output."""

    log_line = Signal(str, str)         # (message, tag)
    progress = Signal(float, object, int, int, str)  # pct, eta, n, total, phase
    processed = Signal(int, int)        # n, total (single live line)
    status = Signal(str)
    finished_ok = Signal(str)           # final status string

    def __init__(
        self,
        mode: str,            # "genre" | "gender"
        input_dir: str,
        *,
        batch_mode: bool,
        tag_style: str,       # combined|split
        gender_field: str,    # comment|gender
        write_meta: bool,
        csv_path: str,
        include_subfolders: bool,
        overwrite_tags: bool,
        settings: Optional[SettingsStore] = None,
        files_from: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._input_dir = input_dir
        self._batch_mode = batch_mode
        self._tag_style = tag_style
        self._gender_field = gender_field
        self._write_meta = write_meta
        self._csv_path = csv_path
        self._include_subfolders = include_subfolders
        self._overwrite_tags = overwrite_tags
        self._settings = settings
        self._files_from = files_from
        self._proc: Optional[subprocess.Popen] = None
        self._stop_requested = False
        self._final_status = "Done"
        # HF model load often emits blank stdout lines; keep other blanks intact.
        self._suppress_model_load_blanks = False

    def stop(self) -> None:
        self._stop_requested = True
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass

    def run(self) -> None:  # noqa: N802 Qt name
        tagger_dir = genre_gender_dir()
        script = genre_gender_script()
        python = resolve_tagger_python()
        if not script.is_file():
            self.log_line.emit(
                f"Bundled tagger not found:\n{script}\n\n"
                "Expected folder: genre_gender_tagger\\ beside STEM organizer.",
                "err",
            )
            self.finished_ok.emit("Failed — tagger missing")
            return
        if python is None or not python.is_file():
            self.log_line.emit(
                f"Genre & Gender Python not found.\n\n{missing_tagger_python_hint()}",
                "err",
            )
            self.finished_ok.emit("Failed — Python missing")
            return

        # Resolve input while STEM cwd may differ; tagger subprocess cwd is tagger_dir.
        input_dir = str(Path(self._input_dir).expanduser().resolve())

        env = tagger_subprocess_env()
        env["GG_MODE"] = self._mode
        env["GG_INPUT"] = input_dir
        env["GG_BATCH"] = "1" if self._batch_mode else "0"
        env["GG_GENDER_FIELD"] = self._gender_field
        env["GG_WRITE_META"] = "1" if self._write_meta else "0"
        env["GG_OVERWRITE"] = "1" if self._overwrite_tags else "0"
        env["GG_RECURSIVE"] = "1" if self._include_subfolders else "0"
        if self._mode == "gender":
            env["GG_REVERB_MODE"] = (
                self._tag_style if self._tag_style in ("combined", "split") else "combined"
            )
            env["GG_TAG_STYLE"] = "combined"
        else:
            env["GG_TAG_STYLE"] = self._tag_style
            env["GG_REVERB_MODE"] = "combined"
        if self._csv_path:
            env["GG_CSV"] = self._csv_path
        if self._files_from:
            env["GG_FILES_FROM"] = self._files_from

        if self._batch_mode:
            workload = "gender" if self._mode == "gender" else "genre"

            def _tune_log(msg: str, tag: str = "info") -> None:
                self.log_line.emit(msg, tag)

            hint = ensure_tuned(
                input_dir,
                self._settings,
                workload=workload,
                log=_tune_log,
            )
            # Respect explicit parent-process env overrides.
            env.setdefault("GG_AUDIO_WORKERS", str(hint.audio_workers))
            env.setdefault("GG_BATCH_SIZE", str(hint.gpu_batch_size))
            env.setdefault("GG_FILE_CHUNK", str(hint.file_chunk))

        try:
            self._proc = subprocess.Popen(
                build_tagger_command(script),
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
            for line in self._iter_lines(self._proc.stdout):
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
                msg = f"[tagger exited: {detail}]"
                if str(detail).strip() in ("1", "-1"):
                    msg += " Scroll up for ERROR lines."
                self.log_line.emit(msg, "warn")
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
        if _TAGGER_DECODE_NOISE_RE.search(bare):
            return
        if _GG_GENDER_MODEL_NOISE_RE.search(bare):
            return
        if bare.startswith("__gg_processed__\t") or bare.startswith("__gg_processed__ "):
            parts = bare.split("\t") if "\t" in bare else bare.split()
            try:
                n = int(float(parts[1]))
                total = int(float(parts[2]))
            except (IndexError, ValueError):
                return
            self.processed.emit(n, total)
            return
        if bare.startswith("__progress__\t") or bare.startswith("__progress__ "):
            parts = bare.split("\t") if "\t" in bare else bare.split()
            try:
                pct = float(parts[1])
            except (IndexError, ValueError):
                return
            eta = None
            if len(parts) >= 3 and parts[2] not in ("", "?"):
                try:
                    eta = float(parts[2])
                except ValueError:
                    eta = None
            n = total = 0
            phase = ""
            if len(parts) >= 5:
                try:
                    n = int(float(parts[3]))
                    total = int(float(parts[4]))
                except ValueError:
                    n = total = 0
            if len(parts) >= 6:
                phase = str(parts[5] or "").strip()
            self.progress.emit(pct, eta, n, total, phase)
            return
        if not bare:
            if self._suppress_model_load_blanks:
                return
            self.log_line.emit("", "info")
            return
        if bare.endswith("Loading model..."):
            self._suppress_model_load_blanks = True
        elif bare.endswith("Model loaded"):
            self._suppress_model_load_blanks = False
        parsed = _TQDM_PCT_RE.search(line)
        if parsed:
            try:
                pct = float(parsed.group("pct"))
            except ValueError:
                pct = None
            n = total = 0
            try:
                n = int(parsed.group("cur"))
                total = int(parsed.group("total"))
            except (ValueError, IndexError):
                pass
            if pct is not None:
                self.progress.emit(pct, None, n, total, "")
            return
        if _PROGRESS_LOG_RE.search(line):
            return
        self.log_line.emit(line, gg_log_tag(line))
