"""FFmpeg waveform/duration helpers + sounddevice audition (same stack as STEM Player)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Any, Literal

import psutil

WaveformPeaks = tuple[tuple[float, float], ...]
AudioEvent = tuple[int, Literal["waveform", "duration", "error"], object]

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_STARTUPINFO = None
if sys.platform == "win32":
    _STARTUPINFO = subprocess.STARTUPINFO()
    _STARTUPINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW


@dataclass(frozen=True, slots=True)
class AudioTools:
    ffmpeg: Path
    ffprobe: Path


def resolve_audio_tools(project_root: Path | None = None) -> AudioTools | None:
    """Resolve ffmpeg + ffprobe for waveform / duration (ffplay not required)."""
    if project_root is None:
        try:
            from ffmpeg_bootstrap import ffmpeg_path, ffprobe_path

            ffmpeg = ffmpeg_path()
            ffprobe = ffprobe_path()
            if ffmpeg and ffprobe:
                return AudioTools(ffmpeg=Path(ffmpeg), ffprobe=Path(ffprobe))
        except ImportError:
            pass

    root = project_root or Path(__file__).resolve().parents[1]
    bundled = root / "ffmpeg"
    suffix = ".exe" if sys.platform == "win32" else ""
    ffmpeg_p = bundled / f"ffmpeg{suffix}"
    ffprobe_p = bundled / f"ffprobe{suffix}"
    if ffmpeg_p.is_file() and ffprobe_p.is_file():
        return AudioTools(ffmpeg=ffmpeg_p, ffprobe=ffprobe_p)

    found_ffmpeg = shutil.which("ffmpeg")
    found_ffprobe = shutil.which("ffprobe")
    if found_ffmpeg and found_ffprobe:
        return AudioTools(
            ffmpeg=Path(found_ffmpeg),
            ffprobe=Path(found_ffprobe),
        )
    return None


def _sounddevice_stack_ok() -> bool:
    try:
        import sounddevice  # noqa: F401
        import soundfile  # noqa: F401

        return True
    except ImportError:
        return False


def reduce_pcm_peaks(samples: array, target_bins: int = 900) -> WaveformPeaks:
    """Reduce mono float PCM to normalized min/max waveform bins."""
    if not samples or target_bins <= 0:
        return ()
    bucket_size = max(1, (len(samples) + target_bins - 1) // target_bins)
    raw: list[tuple[float, float]] = []
    maximum = 0.0
    for start in range(0, len(samples), bucket_size):
        chunk = samples[start : start + bucket_size]
        low = min(chunk)
        high = max(chunk)
        raw.append((low, high))
        maximum = max(maximum, abs(low), abs(high))
    if maximum <= 1e-12:
        return tuple((0.0, 0.0) for _ in raw)
    scale = 1.0 / maximum
    return tuple((low * scale, high * scale) for low, high in raw)


class WaveformCache:
    def __init__(self, max_entries: int = 32) -> None:
        self.max_entries = max_entries
        self._items: OrderedDict[tuple[str, int, int], WaveformPeaks] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)

    def get(self, path: Path) -> WaveformPeaks | None:
        key = self.key(path)
        with self._lock:
            peaks = self._items.get(key)
            if peaks is not None:
                self._items.move_to_end(key)
            return peaks

    def put(self, path: Path, peaks: WaveformPeaks) -> None:
        key = self.key(path)
        with self._lock:
            self._items[key] = peaks
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)


class AudioPreviewService:
    """Waveform via ffmpeg; audition via sounddevice (same as STEM Player)."""

    def __init__(
        self,
        project_root: Path | None = None,
        *,
        tools: AudioTools | None = None,
        cache: WaveformCache | None = None,
    ) -> None:
        self.tools = tools if tools is not None else resolve_audio_tools(project_root)
        self.cache = cache or WaveformCache()
        self.events: SimpleQueue[AudioEvent] = SimpleQueue()
        self.generation = 0
        self.active_path: Path | None = None
        self.duration = 0.0
        self._waveform_process: subprocess.Popen[bytes] | None = None
        self._waveform_lock = threading.Lock()
        self._probe_process: subprocess.Popen[bytes] | None = None
        self._probe_lock = threading.Lock()
        self._engine: Any = None
        self._engine_lock = threading.Lock()
        self._pcm_ready = False
        self._pcm_error: str | None = None
        self._play_when_ready = False
        self._resume_position = 0.0

    @property
    def available(self) -> bool:
        """True when waveform/duration tools (ffmpeg + ffprobe) are present."""
        return self.tools is not None

    @property
    def playback_available(self) -> bool:
        """True when sounddevice + soundfile are importable (STEM Player stack)."""
        return _sounddevice_stack_ok()

    @property
    def unavailable_message(self) -> str:
        return "Add ffmpeg and ffprobe to the ffmpeg folder (or re-run install-deps.bat)."

    @property
    def playback_unavailable_message(self) -> str:
        return "Audition needs sounddevice + soundfile (same as STEM Player)."

    def load(self, path: Path) -> int:
        """Stop current audio and asynchronously load waveform + decode for play."""
        self.generation += 1
        generation = self.generation
        self.stop()
        self._cancel_waveform()
        self._cancel_probe()
        self.active_path = path
        self.duration = 0.0
        self._pcm_ready = False
        self._pcm_error = None
        self._play_when_ready = False
        self._resume_position = 0.0

        if not self.available:
            self.events.put((generation, "error", self.unavailable_message))
            return generation
        if not path.is_file():
            self.events.put((generation, "error", "Audio file is missing."))
            return generation
        threading.Thread(
            target=self._probe_duration,
            args=(generation, path),
            daemon=True,
        ).start()
        if self.playback_available:
            threading.Thread(
                target=self._decode_for_playback,
                args=(generation, path),
                daemon=True,
            ).start()
        try:
            cached = self.cache.get(path)
        except OSError as exc:
            self.events.put((generation, "error", str(exc)))
            return generation
        if cached is not None:
            self.events.put((generation, "waveform", cached))
            return generation

        threading.Thread(
            target=self._extract_waveform,
            args=(generation, path),
            daemon=True,
        ).start()
        return generation

    def _decode_for_playback(self, generation: int, path: Path) -> None:
        try:
            from stem_organizer.player.audio_engine import AudioEngine, PLAYER_SR
            from stem_organizer.player.audio_io import load_player_audio
            from stem_organizer.player.track_state import TrackState
        except ImportError as exc:
            self._pcm_error = str(exc)
            return
        try:
            audio = load_player_audio(str(path), sr=PLAYER_SR, ch=2)
        except Exception as exc:
            if generation == self.generation:
                self._pcm_error = str(exc).splitlines()[0][:160]
            return
        if generation != self.generation:
            return
        track = TrackState(name=path.name, path=path, audio=audio, color="#a855f7")
        engine = AudioEngine([track], sr=PLAYER_SR)
        try:
            engine.start_stream()
        except Exception as exc:
            if generation == self.generation:
                self._pcm_error = f"Audio output failed: {exc}"
            return
        with self._engine_lock:
            if generation != self.generation:
                engine.stop_stream()
                return
            old = self._engine
            self._engine = engine
            self._pcm_ready = True
            self._pcm_error = None
            if self.duration <= 0:
                self.duration = float(engine.duration)
                self.events.put((generation, "duration", self.duration))
            play_now = self._play_when_ready
            resume = self._resume_position
            self._play_when_ready = False
        if old is not None:
            try:
                old.set_playing(False)
                old.stop_stream()
            except Exception:
                pass
        if play_now:
            engine.position = resume
            engine.set_playing(True)

    def _probe_duration(self, generation: int, path: Path) -> None:
        assert self.tools is not None
        command = [
            str(self.tools.ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=_CREATE_NO_WINDOW,
                startupinfo=_STARTUPINFO,
            )
            with self._probe_lock:
                if generation != self.generation:
                    self._terminate_process(process)
                    return
                self._probe_process = process
            stdout, _stderr = process.communicate(timeout=10)
            if generation != self.generation or process.returncode:
                return
            duration = float(stdout.decode("ascii", errors="ignore").strip())
            if duration > 0:
                self.duration = duration
                self.events.put((generation, "duration", duration))
        except subprocess.TimeoutExpired:
            if process is not None:
                self._terminate_process(process)
        except (OSError, ValueError):
            pass
        finally:
            with self._probe_lock:
                if self._probe_process is process:
                    self._probe_process = None

    def _extract_waveform(self, generation: int, path: Path) -> None:
        assert self.tools is not None
        command = [
            str(self.tools.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "a:0",
            "-ac",
            "1",
            "-ar",
            "4000",
            "-f",
            "f32le",
            "pipe:1",
        ]
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                creationflags=_CREATE_NO_WINDOW,
                startupinfo=_STARTUPINFO,
            )
            with self._waveform_lock:
                if generation != self.generation:
                    process.terminate()
                    return
                self._waveform_process = process
            stdout, stderr = process.communicate()
            if generation != self.generation:
                return
            if process.returncode:
                detail = stderr.decode("utf-8", errors="replace").strip()
                self.events.put(
                    (generation, "error", detail or "Unable to decode this audio file.")
                )
                return
            samples = array("f")
            samples.frombytes(stdout)
            if sys.byteorder != "little":
                samples.byteswap()
            peaks = reduce_pcm_peaks(samples)
            self.cache.put(path, peaks)
            self.events.put((generation, "waveform", peaks))
        except (OSError, ValueError) as exc:
            if generation == self.generation:
                self.events.put((generation, "error", str(exc)))
        finally:
            with self._waveform_lock:
                if self._waveform_process is process:
                    self._waveform_process = None

    def _cancel_waveform(self) -> None:
        with self._waveform_lock:
            process = self._waveform_process
            self._waveform_process = None
        if process is not None and process.poll() is None:
            self._terminate_process(process)

    def _cancel_probe(self) -> None:
        with self._probe_lock:
            process = self._probe_process
            self._probe_process = None
        if process is not None and process.poll() is None:
            self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        """Kill *process* and any children so Windows releases audio file handles."""
        try:
            parent = psutil.Process(process.pid)
        except (psutil.Error, OSError):
            parent = None

        targets: list[psutil.Process] = []
        if parent is not None:
            try:
                targets.extend(parent.children(recursive=True))
            except (psutil.Error, OSError):
                pass
            targets.append(parent)

        if targets:
            for proc in targets:
                try:
                    proc.terminate()
                except (psutil.Error, OSError):
                    pass
            _gone, alive = psutil.wait_procs(targets, timeout=1.0)
            for proc in alive:
                try:
                    proc.kill()
                except (psutil.Error, OSError):
                    pass
            if alive:
                psutil.wait_procs(alive, timeout=1.0)
            return

        try:
            process.terminate()
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except OSError:
            pass

    def _teardown_engine(self) -> None:
        with self._engine_lock:
            engine = self._engine
            self._engine = None
            self._pcm_ready = False
        if engine is not None:
            try:
                engine.set_playing(False)
                engine.stop_stream()
            except Exception:
                pass

    def play_pause(self) -> Literal["playing", "paused", "stopped"]:
        if (
            not self.playback_available
            or self.active_path is None
            or not self.active_path.is_file()
        ):
            return "stopped"
        with self._engine_lock:
            engine = self._engine
            ready = self._pcm_ready
            err = self._pcm_error
        if err and not ready:
            return "stopped"
        if not ready or engine is None:
            # Decode still in flight — start when ready.
            self._play_when_ready = True
            self._resume_position = self.playback_position()
            return "playing"
        if engine.playing:
            self._resume_position = float(engine.position)
            engine.set_playing(False)
            return "paused"
        if engine.position >= engine.duration - 0.01:
            engine.position = 0.0
        engine.set_playing(True)
        return "playing"

    def playback_state(self) -> Literal["playing", "paused", "stopped"]:
        with self._engine_lock:
            engine = self._engine
            ready = self._pcm_ready
            pending = self._play_when_ready
        if pending and not ready:
            return "playing"
        if engine is None or not ready:
            return "stopped"
        if engine.playing:
            return "playing"
        if engine.position > 0.01 and engine.position < engine.duration - 0.01:
            return "paused"
        return "stopped"

    def playback_position(self) -> float:
        with self._engine_lock:
            engine = self._engine
            ready = self._pcm_ready
            resume = self._resume_position
        if engine is not None and ready:
            return float(engine.position)
        return float(resume)

    def seek(self, seconds: float) -> float:
        """Seek to an absolute position in seconds."""
        if self.active_path is None:
            return 0.0
        limit = self.duration if self.duration > 0 else float("inf")
        target = max(0.0, min(limit, float(seconds)))
        with self._engine_lock:
            engine = self._engine
            ready = self._pcm_ready
            self._resume_position = target
        if engine is not None and ready:
            was_playing = engine.playing
            engine.position = target
            if was_playing:
                engine.set_playing(True)
        return target

    def seek_relative(self, delta: float) -> float:
        """Seek by a relative offset in seconds (keyboard Left/Right)."""
        return self.seek(self.playback_position() + float(delta))

    def stop(self) -> None:
        self._play_when_ready = False
        self._resume_position = 0.0
        with self._engine_lock:
            engine = self._engine
        if engine is not None:
            try:
                engine.set_playing(False)
                engine.position = 0.0
            except Exception:
                pass

    def reset(self) -> None:
        self.generation += 1
        self._play_when_ready = False
        self._teardown_engine()
        self._cancel_waveform()
        self._cancel_probe()
        self.active_path = None
        self.duration = 0.0
        self._pcm_error = None
        self._resume_position = 0.0

    def _kill_orphan_audio_tools(self) -> None:
        """Kill leftover ffmpeg/ffprobe children of this process."""
        tool_names = {"ffmpeg", "ffmpeg.exe", "ffprobe", "ffprobe.exe"}
        if self.tools is not None:
            tool_names.add(self.tools.ffmpeg.name.lower())
            tool_names.add(self.tools.ffprobe.name.lower())
        try:
            children = psutil.Process().children(recursive=True)
        except (psutil.Error, OSError):
            return
        targets: list[psutil.Process] = []
        for proc in children:
            try:
                name = (proc.name() or "").lower()
            except (psutil.Error, OSError):
                continue
            if name in tool_names:
                targets.append(proc)
        if not targets:
            return
        for proc in targets:
            try:
                proc.kill()
            except (psutil.Error, OSError):
                pass
        psutil.wait_procs(targets, timeout=1.5)

    def release_for_file_ops(self, *, settle_s: float | None = None) -> None:
        """Stop preview/decode jobs and clear buffers before rename/move.

        Playback uses an in-RAM copy (soundfile/ffmpeg → sounddevice), but
        in-flight ffmpeg/ffprobe waveform jobs can still hold share locks.
        """
        self.reset()
        self._kill_orphan_audio_tools()
        if settle_s is None:
            settle_s = 0.15 if sys.platform == "win32" else 0.0
        if settle_s > 0:
            time.sleep(settle_s)

    def shutdown(self) -> None:
        self.reset()
        self._kill_orphan_audio_tools()
