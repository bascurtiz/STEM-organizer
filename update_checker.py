# update_checker.py
import os
import threading
import webbrowser

import requests
import tkinter as tk
from packaging.version import parse as parse_version

GITHUB_REPO_OWNER = 'bascurtiz'
GITHUB_REPO_NAME = 'STEM-organizer'
GITHUB_API_URL = (
    f'https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/latest'
)
RELEASES_PAGE_URL = f'https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases'

# Set STEM_FORCE_UPDATE_DIALOG=1 to show the update dialog with fake data (no remote needed).
# Optional: STEM_FORCE_UPDATE_TAG=v9.9.9 to control the version string shown.
_FORCE_TRUTHY = ("1", "true", "True", "yes", "YES")


def _force_update_dialog() -> bool:
    return os.environ.get("STEM_FORCE_UPDATE_DIALOG", "").strip() in _FORCE_TRUTHY


def _force_update_tag() -> str:
    tag = os.environ.get("STEM_FORCE_UPDATE_TAG", "").strip()
    return tag or "v99.0.0"


def get_latest_release_info():
    """Fetch the latest release information from the GitHub API."""
    try:
        headers = {'Accept': 'application/vnd.github.v3+json'}
        response = requests.get(GITHUB_API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        print(f'[Update Check] Network or API error: {exc}')
        return None
    except Exception as exc:
        print(f'[Update Check] Unexpected error fetching release info: {exc}')
        return None


def compare_versions(current_version_str, latest_version_str):
    """Return True when the latest release tag is newer than the current version."""
    try:
        current_version_str = current_version_str.lstrip('v').lstrip('.')
        latest_version_str = latest_version_str.lstrip('v').lstrip('.')

        current_version = parse_version(current_version_str)
        latest_version = parse_version(latest_version_str)
        return latest_version > current_version
    except Exception as exc:
        print(
            f"[Update Check] Error comparing versions "
            f"('{current_version_str}' vs '{latest_version_str}'): {exc}"
        )
        return False


def _show_update_dialog_qt(new_version_tag, parent_window):
    """Fluent opaque-card prompt (matches app dialogs; works with Qt MainWindow)."""
    from stem_organizer.widgets.dialogs import ask_yes_no

    message = (
        f'A new version ({new_version_tag}) is available.\n'
        f'Visit the Releases page on GitHub to download.'
    )
    if ask_yes_no(
        parent_window,
        'Update Available',
        message,
        yes_text='Download Update',
        no_text='Later',
    ):
        try:
            webbrowser.open(RELEASES_PAGE_URL, new=2)
        except Exception as exc:
            print(f'[Update Check] Failed to open browser: {exc}')


def _show_update_dialog_tk(new_version_tag, parent_window):
    """Legacy Tk dialog (only when parent is a real Tk window)."""
    if parent_window is None or not isinstance(parent_window, (tk.Tk, tk.Toplevel)):
        print('ERROR: Cannot show update dialog, invalid parent window.')
        return
    try:
        if not parent_window.winfo_exists():
            return
    except tk.TclError:
        return

    dlg = tk.Toplevel(parent_window)
    dlg.title('Update Available')
    dlg.resizable(False, False)
    dlg.attributes('-topmost', True)
    dlg.transient(parent_window)

    message = (
        f'A new version ({new_version_tag}) is available.\n'
        f'Visit the Releases page on GitHub to download.'
    )
    tk.Label(dlg, text=message, justify='left', wraplength=380).pack(padx=20, pady=(20, 12))

    btn_row = tk.Frame(dlg)
    btn_row.pack(pady=(0, 20))

    def open_release_page():
        try:
            webbrowser.open(RELEASES_PAGE_URL, new=2)
        except Exception as exc:
            print(f'[Update Check] Failed to open browser: {exc}')
        dlg.destroy()

    tk.Button(btn_row, text='Download Update', width=16, command=open_release_page).pack(
        side='left', padx=6
    )
    tk.Button(btn_row, text='Later', width=10, command=dlg.destroy).pack(side='left', padx=6)

    dlg.update_idletasks()
    pw, ph = parent_window.winfo_width(), parent_window.winfo_height()
    px, py = parent_window.winfo_x(), parent_window.winfo_y()
    dw, dh = dlg.winfo_width(), dlg.winfo_height()
    dlg.geometry(f'+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}')

    dlg.grab_set()
    dlg.focus_force()
    dlg.bind('<Return>', lambda _e: open_release_page())
    dlg.bind('<Escape>', lambda _e: dlg.destroy())
    dlg.protocol('WM_DELETE_WINDOW', dlg.destroy)
    parent_window.wait_window(dlg)


def show_update_dialog(new_version_tag, parent_window):
    """Show a modal dialog when a newer release is available."""
    try:
        from PySide6.QtWidgets import QWidget

        if isinstance(parent_window, QWidget):
            _show_update_dialog_qt(new_version_tag, parent_window)
            return
    except Exception as exc:
        print(f'[Update Check] Qt dialog failed, falling back to Tk: {exc}')

    _show_update_dialog_tk(new_version_tag, parent_window)


def _make_ui_scheduler(root_window):
    """Build a callable that posts work onto the UI thread.

    Must be constructed on the UI thread (``run_check_in_thread`` entry).
    """
    try:
        from PySide6.QtCore import QObject, Signal
        from PySide6.QtWidgets import QWidget

        if isinstance(root_window, QWidget):
            class _UiBridge(QObject):
                request = Signal(object)

            bridge = _UiBridge(root_window)
            bridge.request.connect(lambda fn: fn())
            return lambda fn: bridge.request.emit(fn)
    except Exception as exc:
        print(f'[Update Check] Qt UI scheduler unavailable: {exc}')

    if hasattr(root_window, 'after') and callable(getattr(root_window, 'after')):
        return lambda fn: root_window.after(0, fn)

    return lambda fn: fn()


def run_check_in_thread(current_version, root_window):
    """Check GitHub releases in a background thread; show dialog on the UI thread."""
    schedule = _make_ui_scheduler(root_window)

    # Force path: no network; defer so the main window finishes mapping first.
    if _force_update_dialog():
        latest_tag = _force_update_tag()
        print(
            f'[Update Check] STEM_FORCE_UPDATE_DIALOG set — '
            f'showing dialog for {latest_tag} (current {current_version}).'
        )

        def _show(tag=latest_tag):
            show_update_dialog(tag, parent_window=root_window)

        try:
            from PySide6.QtCore import QTimer
            from PySide6.QtWidgets import QWidget

            if isinstance(root_window, QWidget):
                QTimer.singleShot(400, _show)
                return
        except Exception:
            pass
        schedule(_show)
        return

    def threaded_task():
        release_info = get_latest_release_info()
        if not release_info or 'tag_name' not in release_info:
            print('[Update Check] Could not retrieve valid release information from thread.')
            return

        latest_tag = release_info['tag_name']
        print(f'[Update Check] Latest: {latest_tag}, Current: {current_version}')

        if compare_versions(current_version, latest_tag):
            print(f'[Update Check] New version found: {latest_tag}. Scheduling dialog.')
            tag = latest_tag
            schedule(lambda: show_update_dialog(tag, parent_window=root_window))
        else:
            print('[Update Check] Application is up-to-date.')

    threading.Thread(target=threaded_task, daemon=True).start()
