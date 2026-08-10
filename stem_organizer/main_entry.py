"""Entry point: QApplication + splash + single-instance + main window.

Stage 9 wires together:
  - single_instance.acquire_single_instance() gate
  - QSplashScreen + StartupWorker (deps_bootstrap + classify_backend._init_ml)
  - MainWindow construction while splash still visible, then splash.finish(main)
  - update_checker background thread
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

# qfluentwidgets prints a promotional banner ("QFluentWidgets Pro is now
# released …") to stdout on first import. That import happens here, at module
# load time, in EVERY process — including the --run-tagger tagger children
# whose stdout the GUI captures into the log panel. Silence it so the banner
# never clutters the log.
with contextlib.redirect_stdout(io.StringIO()):
    from . import theme
    from .app import MainWindow
    from .settings_store import SettingsStore, app_dir
    from .splash import Splash, dismiss_splash, show_splash_and_startup
    from .widgets.dialogs import show_info


def _startup_error_report(exc: BaseException) -> str:
    """Full diagnostics for startup_error.log (exc may no longer be 'active')."""
    parts: list[str] = [
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "",
        f"executable: {sys.executable}",
        f"frozen: {getattr(sys, 'frozen', False)}",
        f"_MEIPASS: {getattr(sys, '_MEIPASS', None)}",
        f"cwd: {os.getcwd()}",
        f"python: {sys.version}",
        "",
        "sys.path:",
    ]
    parts.extend(f"  {p}" for p in sys.path)
    parts.append("")
    parts.append("site-packages candidates:")
    try:
        from deps_bootstrap import app_dir as deps_app_dir
        from deps_bootstrap import external_site_dirs

        parts.append(f"  deps app_dir: {deps_app_dir()}")
        for path in external_site_dirs():
            parts.append(f"  {path}  exists={path.is_dir()}")
    except Exception as list_exc:
        parts.append(f"  (could not list: {list_exc})")
    return "\n".join(parts)


def _is_multiprocessing_child() -> bool:
    """True when this process is a ProcessPool / mp worker (frozen .exe re-exec)."""
    try:
        import multiprocessing as mp

        if mp.current_process().name != "MainProcess":
            return True
    except Exception:
        pass
    joined = " ".join(sys.argv).lower()
    if "--multiprocessing" in joined:
        return True
    if "multiprocessing.spawn" in joined or "multiprocessing.forking" in joined:
        return True
    return False


def run(argv: list[str] | None = None) -> int:
    # Pool workers (frozen .exe) re-exec this entry — handle them before Qt / mutex.
    import multiprocessing as mp

    mp.freeze_support()
    if _is_multiprocessing_child():
        return 0

    argv = list(sys.argv if argv is None else argv)

    # Phase 3: tagger self-dispatch (no host Python). Must run before Qt /
    # single-instance so child processes don't steal the GUI mutex.
    try:
        from tagger_launch import maybe_run_tagger_dispatch

        tagger_code = maybe_run_tagger_dispatch(argv)
        if tagger_code is not None:
            return int(tagger_code)
    except Exception:
        # If dispatch import fails, fall through to GUI (dev / incomplete build).
        if any(a == "--run-tagger" for a in argv):
            raise

    # Quiet Qt font fallback spam (OpenType support missing for Consolas/Arial/…).
    # Must be set before QApplication is constructed.
    _font_rule = "qt.text.font.db=false"
    existing = os.environ.get("QT_LOGGING_RULES", "").strip()
    if "qt.text.font.db" not in existing:
        os.environ["QT_LOGGING_RULES"] = (
            f"{existing};{_font_rule}" if existing else _font_rule
        )

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("STEM organizer")
    app.setApplicationDisplayName("STEM organizer")
    app.setApplicationVersion(theme.APP_VERSION)

    icon_path = Path(__file__).resolve().parent.parent / "logo.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    theme.apply_theme(app)

    # Single instance gate (set STEM_ALLOW_MULTI=1 to open a second window for testing)
    try:
        from single_instance import acquire_single_instance

        if os.environ.get("STEM_ALLOW_MULTI", "").strip() not in ("1", "true", "True", "yes", "YES"):
            if os.environ.get("STEM_TAGGER_CHILD", "").strip() in ("1", "true", "True"):
                pass  # tagger self-dispatch child — no mutex
            elif not acquire_single_instance():
                # Pool workers re-exec the frozen .exe — exit quiet, no dialog.
                if _is_multiprocessing_child():
                    return 0
                show_info(None, "STEM organizer", "Another instance is already running.")
                return 0
    except Exception:
        # single_instance optional — continue if missing
        pass

    settings = SettingsStore()

    # Splash + startup
    splash_shown = False
    try:
        from .splash import Splash  # noqa: F401 — verify import works

        splash_shown = True
    except Exception:
        splash_shown = False

    if splash_shown:
        splash, worker = show_splash_and_startup(
            on_ready=lambda exc: _on_ready(app, settings, exc, splash)
        )
        return app.exec()
    else:
        _construct_and_show(app, settings, None, splash=None)
        return app.exec()


def _on_ready(
    app: QApplication,
    settings: SettingsStore,
    startup_error,
    splash: Splash | None,
) -> None:
    if startup_error is not None:
        # Always dismiss splash on failure so it never sticks forever.
        if splash is not None:
            splash.close()
        report = _startup_error_report(startup_error)
        try:
            sys.stderr.write(report + "\n")
        except Exception:
            pass
        try:
            log_path = app_dir() / "startup_error.log"
            log_path.write_text(report, encoding="utf-8")
        except OSError:
            pass
        # Keep dialog readable; full traceback + sys.path are in the log.
        msg = str(startup_error).strip() or repr(startup_error)
        if len(msg) > 1800:
            msg = msg[:1800] + "\n…"
        show_info(
            None,
            "STEM organizer — startup failed",
            f"Startup failed:\n{msg}\n\nDetails were written to startup_error.log.",
        )
        sys.exit(1)
    _construct_and_show(app, settings, startup_error, splash=splash)


def _construct_and_show(
    app: QApplication,
    settings: SettingsStore,
    startup_error,
    splash: Splash | None = None,
) -> None:
    # Build UI while splash still covers the screen (avoids blank gap).
    try:
        window = MainWindow(settings)

        from .tabs import register_all_tabs

        tab_errors = register_all_tabs(window, settings)
        if tab_errors:
            # Frozen builds have no console — surface failures in the LOG panel.
            window.append_log(
                "Some tabs failed to load (see tab_errors.log beside the exe):",
                "warn",
            )
            for line in tab_errors:
                first = (line.splitlines() or [line])[0]
                window.append_log(f"  {first}", "err")

        theme.polish_fluent_controls(window)
        # Rename Apply clones Clear width after Clear gets polished fonts/padding
        from PySide6.QtCore import QTimer

        from .renamer.rules_panel import RulesPanel

        def _sync_rename_apply() -> None:
            for panel in window.findChildren(RulesPanel):
                panel.match_apply_to_clear()

        _sync_rename_apply()
        window.show()
        QTimer.singleShot(0, _sync_rename_apply)

        from .widgets.titlebar import apply_window_corner_preference

        # Region clip needs a mapped HWND + final size (same as CTk after show)
        apply_window_corner_preference(window, theme.WINDOW_CORNER_RADIUS)

        # Classic Qt handoff: main is shown, then splash fades/finishes onto it.
        if splash is not None:
            dismiss_splash(splash, window)
            splash = None  # owned by fade; don't double-close in except

        # Background update check
        try:
            from update_checker import run_check_in_thread

            run_check_in_thread(theme.APP_VERSION, window)
        except Exception:
            pass

        # HTDemucs loads lazily on the first SI-SDR / separation run (the
        # worker logs "Loading model …" and the session is cached for the rest
        # of the session) — no background warm, no post-launch GIL stall.
    except Exception:
        if splash is not None:
            splash.close()
        raise


if __name__ == "__main__":
    raise SystemExit(run())
