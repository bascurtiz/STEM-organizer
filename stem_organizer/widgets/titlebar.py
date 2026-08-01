"""Custom frameless-window title bar + Win32 rounded corners.

Port of stem_organizer_ui._build_custom_title_bar, _bind_title_drag,
_enable_edge_resize, _refresh_window_corners / _win_apply_rounded_corners.

Frameless windows need SetWindowRgn (CreateRoundRectRgn) — DWM corner
preference alone does not round overrideredirect / FramelessWindowHint UIs.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, QPoint, QTimer, Qt
from PySide6.QtGui import QFont, QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from .. import theme


# ---------------------------------------------------------------------------
# Win32 rounded corners — SetWindowRgn (frameless) + DWM chrome (optional)
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    try:
        _user32 = ctypes.WinDLL("user32")  # type: ignore[attr-defined]
        _gdi32 = ctypes.WinDLL("gdi32")  # type: ignore[attr-defined]
    except OSError:
        _user32 = None
        _gdi32 = None
    try:
        _dwmapi = ctypes.WinDLL("dwmapi")  # type: ignore[attr-defined]
    except OSError:
        _dwmapi = None
else:
    _user32 = None
    _gdi32 = None
    _dwmapi = None

DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWCP_DONOTROUND = 1
DWMWCP_ROUND = 2

# WM_NCHITTEST results — native edge resize for frameless windows
WM_NCHITTEST = 0x0084
WM_NCCALCSIZE = 0x0083
WM_NCPAINT = 0x0085
WM_NCACTIVATE = 0x0086
WM_NCLBUTTONDBLCLK = 0x00A3  # caption double-click → OS maximize
WVR_REDRAW = 0x0100
HTCLIENT = 1
HTCAPTION = 2
HTLEFT = 10
HTRIGHT = 11
HTTOP = 12
HTTOPLEFT = 13
HTTOPRIGHT = 14
HTBOTTOM = 15
HTBOTTOMLEFT = 16
HTBOTTOMRIGHT = 17

# Win32 style bits — without WS_THICKFRAME, Windows never sends usable
# WM_NCHITTEST edge hits for FramelessWindowHint windows (resize stays dead).
GWL_STYLE = -16
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WS_SYSMENU = 0x00080000
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020
SWP_NOACTIVATE = 0x0010

RESIZE_BORDER = 10  # wide enough to show / grab the OS resize cursor easily


def enable_win32_thick_frame(window: QWidget) -> None:
    """Add WS_THICKFRAME so the OS delivers edge hit-tests + native resize.

    Must run only after the platform window has been shown successfully.
    Calling ``winId()`` / ``SetWindowLong`` during ``__init__`` (before the
    first ``show``) leaves Qt's window-creation state inconsistent and can
    make the next ``CreateWindowEx`` fail.

    Arm ``_win32_thick_frame`` *before* ``SWP_FRAMECHANGED`` so the synchronous
    ``WM_NCCALCSIZE`` is handled and Windows does not leave light thick-frame
    chrome around the client (intermittent gray/white borders).
    """
    if _user32 is None or sys.platform != "win32":
        return
    if not window.isVisible():
        return
    try:
        hwnd = int(window.winId())
        if hwnd == 0:
            return
        # Already applied — avoid SetWindowPos(FRAMECHANGED) which fights
        # Qt during maximize and spam "Unable to set geometry" warnings.
        if getattr(window, "_win32_thick_frame", False):
            style = int(_user32.GetWindowLongW(hwnd, GWL_STYLE))
            if style & WS_THICKFRAME:
                return
        style = int(_user32.GetWindowLongW(hwnd, GWL_STYLE))
        # WS_THICKFRAME alone unlocks edge hit-testing. Avoid WS_CAPTION /
        # WS_SYSMENU here — they get mirrored into Qt title hints and can
        # break a later CreateWindowEx when combined with FramelessWindowHint.
        new_style = style | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX
        # Must be True before FRAMECHANGED — that call sends WM_NCCALCSIZE now.
        window._win32_thick_frame = True  # type: ignore[attr-defined]
        if new_style != style:
            _user32.SetWindowLongW(hwnd, GWL_STYLE, new_style)
            _user32.SetWindowPos(
                hwnd,
                None,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
            )
        # FRAMECHANGED can invalidate the region clip; re-apply dark chrome.
        apply_window_corner_preference(window, theme.WINDOW_CORNER_RADIUS)
        window.update()
    except Exception:
        pass


def prepare_dark_frameless_chrome(window: QWidget) -> None:
    """Paint theme bg on the HWND/client so default gray never shows in seams.

    QSS alone is not always enough for QMainWindow / top-level QWidget while
    Win32 thick-frame geometry settles; palette + auto-fill closes the gap.
    """
    from PySide6.QtGui import QColor, QPalette

    targets: list[QWidget] = [window]
    cw_getter = getattr(window, "centralWidget", None)
    if callable(cw_getter):
        cw = cw_getter()
        if cw is not None and cw is not window:
            targets.append(cw)

    bg = QColor(theme.COLORS["bg"])
    for w in targets:
        pal = w.palette()
        for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
            pal.setColor(group, QPalette.Window, bg)
            pal.setColor(group, QPalette.Base, bg)
        w.setPalette(pal)
        w.setAutoFillBackground(True)
        w.setAttribute(Qt.WA_StyledBackground, True)
        # Do not leave translucent attrs that reveal desktop/system gray.
        w.setAttribute(Qt.WA_TranslucentBackground, False)


def _win_is_zoomed(hwnd: int) -> bool:
    if _user32 is None or hwnd == 0:
        return False
    try:
        return bool(_user32.IsZoomed(hwnd))
    except Exception:
        return False


def _win_get_resize_border_thickness(hwnd: int, horizontal: bool) -> int:
    """DPI-aware frame border thickness (maximized NCCALCSIZE inset)."""
    if _user32 is None:
        return 8
    try:
        SM_CXSIZEFRAME = 32
        SM_CYSIZEFRAME = 33
        SM_CXPADDEDBORDER = 92
        index = SM_CXSIZEFRAME if horizontal else SM_CYSIZEFRAME
        # Prefer per-window DPI metrics (Win10+); fall back to primary.
        try:
            dpi = int(_user32.GetDpiForWindow(hwnd))
            if dpi > 0 and hasattr(_user32, "GetSystemMetricsForDpi"):
                frame = int(_user32.GetSystemMetricsForDpi(index, dpi))
                pad = int(_user32.GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi))
                if frame + pad > 0:
                    return frame + pad
        except Exception:
            pass
        frame = int(_user32.GetSystemMetrics(index))
        pad = int(_user32.GetSystemMetrics(SM_CXPADDEDBORDER))
        if frame + pad > 0:
            return frame + pad
    except Exception:
        pass
    return 8


def is_window_filled(window: QWidget) -> bool:
    """True when the window fills the monitor (OS maximize or CTk-style fill)."""
    if getattr(window, "_custom_maximized", False):
        return True
    try:
        if window.isMaximized():
            return True
    except Exception:
        pass
    try:
        hwnd = int(window.winId())
        if hwnd and _win_is_zoomed(hwnd):
            return True
    except Exception:
        pass
    return False


def _win_monitor_work_area(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    """Monitor work area in physical pixels: (x, y, width, height).

    Same source as CTk ``_win_work_area`` — ``GetMonitorInfoW.rcWork``.
    Physical coords match ``SetWindowPos`` (unlike Qt logical ``availableGeometry``).
    """
    if _user32 is None or hwnd == 0:
        return None
    try:
        class _MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        hmon = _user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
        if not hmon:
            return None
        mi = _MONITORINFO()
        mi.cbSize = ctypes.sizeof(_MONITORINFO)
        if not _user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            return None
        work = mi.rcWork
        return (
            int(work.left),
            int(work.top),
            int(work.right - work.left),
            int(work.bottom - work.top),
        )
    except Exception:
        return None


def apply_work_area_fill(window: QWidget) -> None:
    """Fill the monitor work area without Qt ``setGeometry`` frame-margin fights.

    ``setGeometry(availableGeometry())`` asks for a *client* rect whose implied
    ``WS_THICKFRAME`` frame sits outside the monitor (e.g. 3440x1400+0+0 → frame
    at -7,-9). Win32 then adjusts and Qt logs ``Unable to set geometry``.

    CTk avoids that by ``SetWindowPos`` on the physical ``rcWork`` rect. Paired
    with our ``WM_NCCALCSIZE`` (full client when not OS-zoomed), the HWND fills
    the work area with no white thick-frame chrome.
    """
    if sys.platform == "win32" and _user32 is not None:
        try:
            hwnd = int(window.winId())
            work = _win_monitor_work_area(hwnd) if hwnd else None
            if work is not None and hwnd:
                x, y, w, h = work
                cur = wintypes.RECT()
                if _user32.GetWindowRect(hwnd, ctypes.byref(cur)):
                    if (
                        int(cur.left) == x
                        and int(cur.top) == y
                        and int(cur.right - cur.left) == w
                        and int(cur.bottom - cur.top) == h
                    ):
                        return
                if _user32.SetWindowPos(
                    hwnd, None, x, y, w, h, SWP_NOZORDER | SWP_NOACTIVATE
                ):
                    return
        except Exception:
            pass
    # Non-Windows / fallback — still prefer work-area fill over showMaximized.
    screen = window.screen()
    if screen is None:
        from PySide6.QtWidgets import QApplication

        screen = QApplication.primaryScreen()
    if screen is not None:
        avail = screen.availableGeometry()
        g = window.geometry()
        if g == avail:
            return
        window.setGeometry(avail)


def apply_window_rect(window: QWidget, rect) -> None:
    """Restore a Qt logical geometry; skip no-op to avoid setGeometry spam."""
    if rect is None or not rect.isValid():
        return
    g = window.geometry()
    if g.x() == rect.x() and g.y() == rect.y() and g.width() == rect.width() and g.height() == rect.height():
        return
    window.setGeometry(rect)


def _sync_titlebar_max_glyph(window: QWidget) -> None:
    tb = getattr(window, "title_bar", None)
    if tb is not None and hasattr(tb, "_sync_max_glyph"):
        tb._sync_max_glyph()


def _refresh_frame_grips(window: QWidget) -> None:
    handler = getattr(window, "_frame_resize_handler", None)
    if handler is not None:
        handler._layout_grips()
        handler._raise_grips()


def center_window_default_size(window: QWidget, width: int, height: int) -> None:
    """Resize to ``width``×``height`` (clamped to work area) and center."""
    from PySide6.QtWidgets import QApplication

    screen = window.screen() or QApplication.primaryScreen()
    if screen is None:
        window.resize(width, height)
        return
    avail = screen.availableGeometry()
    w = min(width, avail.width())
    h = min(height, avail.height())
    window.resize(w, h)
    x = avail.x() + (avail.width() - w) // 2
    y = avail.y() + (avail.height() - h) // 2
    window.move(x, y)


def toggle_work_area_maximize(window: QWidget) -> None:
    """Fill the monitor work area (CTk-style) — avoid OS ``showMaximized``.

    WS_THICKFRAME + Qt showMaximized leaves bright/white edges around the
    client. Matching CTk, fill via SetWindowPos(rcWork). Window attrs used:
    ``_custom_maximized``, ``_restore_geometry``.
    """
    if getattr(window, "_custom_maximized", False):
        window._custom_maximized = False  # type: ignore[attr-defined]
        geo = getattr(window, "_restore_geometry", None)
        if geo is not None and geo.isValid():
            apply_window_rect(window, geo)
        else:
            window.showNormal()
        apply_window_corner_preference(window, theme.WINDOW_CORNER_RADIUS)
        _sync_titlebar_max_glyph(window)
        _refresh_frame_grips(window)
        return

    # If somehow OS-maximized, normalize first then treat as restore path.
    if window.isMaximized():
        window.showNormal()
        apply_window_corner_preference(window, theme.WINDOW_CORNER_RADIUS)
        _sync_titlebar_max_glyph(window)
        return

    window._restore_geometry = window.geometry()  # type: ignore[attr-defined]
    # Flag before move so NCCALCSIZE (if any) does not treat this as OS max.
    window._custom_maximized = True  # type: ignore[attr-defined]
    apply_work_area_fill(window)
    apply_window_corner_preference(window, theme.WINDOW_CORNER_RADIUS)
    _sync_titlebar_max_glyph(window)
    handler = getattr(window, "_frame_resize_handler", None)
    if handler is not None:
        handler._layout_grips()


def apply_default_size_after_unminimize(
    window: QWidget, width: int, height: int
) -> None:
    """After taskbar restore: clear fill state and reset to default size."""
    if window.isMinimized():
        return
    window._custom_maximized = False  # type: ignore[attr-defined]
    window._restore_geometry = None  # type: ignore[attr-defined]
    if window.isMaximized():
        window.showNormal()
    center_window_default_size(window, width, height)
    apply_window_corner_preference(window, theme.WINDOW_CORNER_RADIUS)
    _sync_titlebar_max_glyph(window)
    _refresh_frame_grips(window)


def note_minimize_restore_to_default(
    window: QWidget, event, *, width: int, height: int
) -> None:
    """Hook for ``changeEvent``: minimize → taskbar restore → default size."""
    if event.type() != QEvent.Type.WindowStateChange:
        return
    if window.isMinimized():
        window._was_minimized = True  # type: ignore[attr-defined]
        return
    if getattr(window, "_was_minimized", False):
        window._was_minimized = False  # type: ignore[attr-defined]
        QTimer.singleShot(
            0, lambda: apply_default_size_after_unminimize(window, width, height)
        )


def refresh_frameless_chrome(window: QWidget) -> None:
    """Force re-apply rounded region + dark DWM after focus/activation glitches.

    Alt-Tab / Explorer focus return can make WS_THICKFRAME paint light NC
    borders; clearing the region cache and re-clipping closes those seams.
    """
    if sys.platform != "win32":
        return
    try:
        # Never touch Win32 chrome while the HWND is going away — that path
        # raises STATUS_FATAL_USER_CALLBACK_EXCEPTION inside nativeEvent.
        if not window.isVisible() or window.isMinimized():
            return
        if not getattr(window, "_win32_thick_frame", False):
            return
        wh = window.windowHandle()
        if wh is None or not wh.isValid():
            return
    except RuntimeError:
        return
    except Exception:
        return
    try:
        window._round_corner_key = None  # type: ignore[attr-defined]
    except Exception:
        pass
    apply_window_corner_preference(window, theme.WINDOW_CORNER_RADIUS)
    try:
        window.update()
    except Exception:
        pass


def note_activation_chrome_refresh(window: QWidget, event) -> None:
    """Hook for ``changeEvent``: re-paint chrome when the window gains focus."""
    try:
        et = event.type()
    except Exception:
        return
    # WindowActivate = this window; ActivationChange fires on app-level switches.
    if et not in (
        QEvent.Type.WindowActivate,
        QEvent.Type.ApplicationActivate,
        QEvent.Type.ActivationChange,
    ):
        return
    if not getattr(window, "_win32_thick_frame", False):
        return
    if not window.isVisible() or window.isMinimized():
        return
    # ActivationChange also fires on deactivate — only refresh when active.
    if et == QEvent.Type.ActivationChange and not window.isActiveWindow():
        return
    # Defer so Windows finishes its own NC paint first, then we overwrite it.
    QTimer.singleShot(0, lambda w=window: refresh_frameless_chrome(w))


def handle_native_frame_message(window: QWidget, msg) -> Optional[tuple]:
    """Handle WM_NCHITTEST / WM_NCCALCSIZE for frameless + thick-frame resize.

    Returns ``(True, result)`` when handled, else ``None``.

    Important: do nothing until ``enable_win32_thick_frame`` has run. Handling
    ``WM_NCCALCSIZE`` during the initial ``CreateWindowEx`` path breaks window
    creation (CreateWindowEx failed / fatal callback exception).
    """
    if sys.platform != "win32":
        return None
    if not getattr(window, "_win32_thick_frame", False):
        return None
    try:
        # Destroy / recreate in progress — never intercept NC messages.
        if window.testAttribute(Qt.WA_DeleteOnClose) and not window.isVisible():
            return None
        wh = window.windowHandle()
        if wh is None or not wh.isValid():
            return None
    except RuntimeError:
        return None
    except Exception:
        pass
    try:
        message = int(msg.message)
    except Exception:
        return None

    if message == WM_NCCALCSIZE:
        # Claim the entire window as client area so WS_THICKFRAME does not
        # paint a native title bar / border.
        # Custom (work-area) maximize is NOT IsZoomed — do not inset then, or
        # you get white gaps / bright edges around the filled window.
        try:
            if not msg.wParam:
                return True, 0
            hwnd = int(window.winId())
            # Only inset for true OS-maximized (IsZoomed) windows.
            if _win_is_zoomed(hwnd) and not getattr(window, "_custom_maximized", False):
                import ctypes
                from ctypes import wintypes

                class _NCCALCSIZE_PARAMS(ctypes.Structure):
                    _fields_ = [("rgrc", wintypes.RECT * 3), ("lppos", ctypes.c_void_p)]

                params = ctypes.cast(msg.lParam, ctypes.POINTER(_NCCALCSIZE_PARAMS)).contents
                rect = params.rgrc[0]
                tx = _win_get_resize_border_thickness(hwnd, True)
                ty = _win_get_resize_border_thickness(hwnd, False)
                rect.left += tx
                rect.top += ty
                rect.right -= tx
                rect.bottom -= ty
            return True, WVR_REDRAW
        except Exception:
            return True, 0

    # Suppress default thick-frame NC painting on activate/deactivate — that is
    # what flashes light/unpainted edges when focus returns from Explorer.
    # Do NOT schedule chrome refresh from inside nativeEvent (can re-enter Win32
    # during destroy and fatal-crash the callback). Qt changeEvent covers focus.
    if message == WM_NCACTIVATE:
        return True, 1  # TRUE → we handled activation chrome

    # Do not swallow WM_NCPAINT — returning handled here has caused
    # CreateWindowEx / fatal callback failures on HWND recreate.

    if message == WM_NCHITTEST:
        try:
            import ctypes

            if is_window_filled(window):
                return None  # no edge resize when filled
            lp = int(msg.lParam)
            sx = ctypes.c_short(lp & 0xFFFF).value
            sy = ctypes.c_short((lp >> 16) & 0xFFFF).value
            hit = nchittest_resize(window, screen_pos=(sx, sy))
            if hit is not None:
                return True, int(hit)
        except Exception:
            return None
        return None

    # HTCAPTION hit-tests enable native drag, but OS also maximizes on caption
    # double-click — that fights SetWindowRgn rounded corners. Maximize button
    # (and our Qt toggle) remain the only maximize paths.
    if message == WM_NCLBUTTONDBLCLK:
        try:
            if int(msg.wParam) == HTCAPTION:
                return True, 0
        except Exception:
            return True, 0

    return None


def _clamp_resize_bounds(
    window: QWidget, x: int, y: int, w: int, h: int
) -> tuple[int, int, int, int]:
    min_w = max(1, window.minimumWidth())
    min_h = max(1, window.minimumHeight())
    w = max(min_w, int(w))
    h = max(min_h, int(h))
    screen = window.screen()
    if screen is not None:
        avail = screen.availableGeometry()
        # Keep a sliver of the title bar on-screen so the window can't vanish.
        margin = 48
        x = min(max(int(x), avail.x() - w + margin), avail.x() + avail.width() - margin)
        y = min(max(int(y), avail.y()), avail.y() + avail.height() - margin)
        w = min(w, max(min_w, avail.width()))
        h = min(h, max(min_h, avail.height()))
    return x, y, w, h


def apply_window_geometry(window: QWidget, x: int, y: int, w: int, h: int) -> None:
    """Apply a new window rect in Qt logical coordinates.

    Prefer ``setGeometry`` over raw ``SetWindowPos`` — Qt geometry and the
    mouse cursor share the same (device-independent) space; SetWindowPos wants
    physical pixels and jumps badly on HiDPI.
    """
    if is_window_filled(window):
        return
    x, y, w, h = _clamp_resize_bounds(window, x, y, w, h)
    # Avoid no-op setGeometry spam (Qt logs Unable to set geometry on clash).
    g = window.geometry()
    if g.x() == x and g.y() == y and g.width() == w and g.height() == h:
        return
    window.setGeometry(x, y, w, h)


def _win_colorref_from_hex(hex_color: str) -> int:
    """COLORREF = 0x00BBGGRR."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b << 16) | (g << 8) | r


def apply_rounded_window_region(
    window: QWidget,
    *,
    maximized: Optional[bool] = None,
    radius: int = theme.WINDOW_CORNER_RADIUS,
) -> None:
    """Clip frameless window via SetWindowRgn — same approach as CTk.

    When maximized (or radius <= 0), clears the region so the window is square.
    Skips no-op updates to avoid flicker / busy cursor loops.

    Region size uses the HWND physical rect (``GetWindowRect``), not Qt logical
    ``width()``/``height()`` — on HiDPI a logical-sized region leaves unpainted
    strips around the clip.
    """
    if _user32 is None or _gdi32 is None or sys.platform != "win32":
        return
    if getattr(window, "_rounding_corners", False):
        return
    try:
        hwnd = int(window.winId())
        if hwnd == 0:
            return
        # Prefer physical HWND size so CreateRoundRectRgn matches the frame.
        cur = wintypes.RECT()
        if _user32.GetWindowRect(hwnd, ctypes.byref(cur)):
            w = max(int(cur.right - cur.left), 1)
            h = max(int(cur.bottom - cur.top), 1)
        else:
            try:
                dpr = float(window.devicePixelRatioF())
            except Exception:
                dpr = 1.0
            w = max(int(round(window.width() * dpr)), 1)
            h = max(int(round(window.height() * dpr)), 1)
        if w < 2 or h < 2:
            return
        if maximized is None:
            is_max = is_window_filled(window)
        else:
            is_max = bool(maximized)
        try:
            dpr = float(window.devicePixelRatioF())
        except Exception:
            dpr = 1.0
        pr = max(1, int(round(radius * dpr))) if radius > 0 else 0
        key = (w, h, is_max, int(pr), "rgn")
        if getattr(window, "_round_corner_key", None) == key:
            return

        window._rounding_corners = True  # type: ignore[attr-defined]
        try:
            if is_max or radius <= 0:
                _user32.SetWindowRgn(hwnd, 0, True)
            else:
                # CreateRoundRectRgn: right/bottom exclusive; ellipse diameter = 2*r
                hrgn = _gdi32.CreateRoundRectRgn(0, 0, w + 1, h + 1, pr * 2, pr * 2)
                if not hrgn:
                    return
                # On success the system owns hrgn — do not DeleteObject.
                if not _user32.SetWindowRgn(hwnd, hrgn, True):
                    _gdi32.DeleteObject(hrgn)
                    return
            window._round_corner_key = key  # type: ignore[attr-defined]
        finally:
            window._rounding_corners = False  # type: ignore[attr-defined]
    except Exception:
        try:
            window._rounding_corners = False  # type: ignore[attr-defined]
        except Exception:
            pass


def apply_window_corner_preference(window: QWidget, radius: int = 12) -> None:
    """Apply CTk-style rounded clip + dark DWM border on Win11."""
    if sys.platform != "win32":
        return
    # winId() before the first successful show() can poison CreateWindowEx.
    try:
        if not window.isVisible():
            return
    except Exception:
        return
    is_max = is_window_filled(window)
    apply_rounded_window_region(window, maximized=is_max, radius=radius)
    if _dwmapi is None:
        return
    try:
        hwnd = int(window.winId())
        if hwnd == 0:
            return
        # Clear any residual region when filled — white halo on framed maximize.
        if is_max and _user32 is not None:
            _user32.SetWindowRgn(hwnd, 0, True)
        pref = ctypes.c_int(DWMWCP_DONOTROUND if is_max else DWMWCP_ROUND)
        _dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_WINDOW_CORNER_PREFERENCE,
            ctypes.byref(pref),
            ctypes.sizeof(pref),
        )
        border = ctypes.c_uint(_win_colorref_from_hex(theme.COLORS["border"]))
        _dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_BORDER_COLOR,
            ctypes.byref(border),
            ctypes.sizeof(border),
        )
        # Caption color (Win11) — match client bg so activate NC flash stays dark.
        caption = ctypes.c_uint(_win_colorref_from_hex(theme.COLORS["bg"]))
        _dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_CAPTION_COLOR,
            ctypes.byref(caption),
            ctypes.sizeof(caption),
        )
        dark = ctypes.c_int(1)
        _dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(dark),
            ctypes.sizeof(dark),
        )
    except Exception:
        pass


def disarm_win32_thick_frame(window: QWidget) -> None:
    """Stop NC message handling before HWND teardown / recreate."""
    try:
        window._win32_thick_frame = False  # type: ignore[attr-defined]
    except Exception:
        pass


def install_rounded_corner_watcher(
    window: QWidget,
    *,
    radius: int = theme.WINDOW_CORNER_RADIUS,
) -> None:
    """Re-apply rounded region on resize / maximize (debounced like CTk)."""
    from PySide6.QtCore import QObject

    timer = QTimer(window)
    timer.setSingleShot(True)
    timer.setInterval(80)

    def _apply() -> None:
        if getattr(window, "_rounding_corners", False):
            return
        try:
            if not window.isVisible():
                return
        except RuntimeError:
            return
        apply_window_corner_preference(window, radius)

    timer.timeout.connect(_apply)

    class CornerFilter(QObject):
        def eventFilter(self, obj, event):  # noqa: N802
            et = event.type()
            if et in (
                QEvent.Resize,
                QEvent.WindowStateChange,
                QEvent.Show,
                QEvent.WindowActivate,
                QEvent.ActivationChange,
            ):
                # Skip while the user is mid-drag — region lag makes resize feel wonky.
                if getattr(window, "_resize_active", False):
                    return False
                try:
                    if not window.isVisible() and et != QEvent.Show:
                        return False
                except RuntimeError:
                    return False
                if et in (QEvent.WindowActivate, QEvent.ActivationChange):
                    # Force region rebuild — Windows may have painted light NC edges.
                    try:
                        if not window.isActiveWindow():
                            return False
                        window._round_corner_key = None  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    timer.setInterval(0)
                else:
                    timer.setInterval(80)
                timer.start()
            return False

    filt = CornerFilter(window)
    window.installEventFilter(filt)
    window._corner_filter = filt  # type: ignore[attr-defined]
    window._corner_timer = timer  # type: ignore[attr-defined]
    # Only after Show — eager winId() here caused CreateWindowEx failures.


def nchittest_resize(
    window: QWidget,
    *,
    border: int = RESIZE_BORDER,
    title_bar_height: int = theme.TITLE_BAR_HEIGHT,
    screen_pos: Optional[tuple] = None,
) -> Optional[int]:
    """Return an HT* code for edge resize, or None to use default handling.

    Port of CTk ``_enable_edge_resize`` hit-test — used from ``nativeEvent``
    ``WM_NCHITTEST`` so resize works from any edge, not only the title bar.

    ``screen_pos`` is the (x, y) screen coordinate Win32 passed in lParam, if
    available — preferred over re-querying QCursor (which can lag the message).
    """
    if is_window_filled(window):
        return None

    if screen_pos is not None:
        from PySide6.QtCore import QPoint

        pos = window.mapFromGlobal(QPoint(screen_pos[0], screen_pos[1]))
    else:
        from PySide6.QtGui import QCursor

        pos = window.mapFromGlobal(QCursor.pos())
    x, y = pos.x(), pos.y()
    w, h = window.width(), window.height()
    if w < 2 or h < 2:
        return None
    b = border
    left, right = x < b, x >= w - b
    top, bottom = y < b, y >= h - b
    if top and left:
        return HTTOPLEFT
    if top and right:
        return HTTOPRIGHT
    if bottom and left:
        return HTBOTTOMLEFT
    if bottom and right:
        return HTBOTTOMRIGHT
    if left:
        return HTLEFT
    if right:
        return HTRIGHT
    if top:
        return HTTOP
    if bottom:
        return HTBOTTOM
    # Allow native caption drag on the title-bar strip (not over buttons)
    if 0 <= y < title_bar_height and x < w - (46 * 3):
        return HTCAPTION
    return None


# Cursor shapes per edge for hover feedback.
_EDGE_CURSORS = {
    "left": Qt.CursorShape.SizeHorCursor,
    "right": Qt.CursorShape.SizeHorCursor,
    "top": Qt.CursorShape.SizeVerCursor,
    "bottom": Qt.CursorShape.SizeVerCursor,
    "topleft": Qt.CursorShape.SizeFDiagCursor,
    "bottomright": Qt.CursorShape.SizeFDiagCursor,
    "topright": Qt.CursorShape.SizeBDiagCursor,
    "bottomleft": Qt.CursorShape.SizeBDiagCursor,
}


def _edge_at(window: QWidget, pos, *, border: int = RESIZE_BORDER) -> Optional[str]:
    """Return which resize edge (or corner) the window-relative ``pos`` is on."""
    if is_window_filled(window):
        return None
    x, y = pos.x(), pos.y()
    w, h = window.width(), window.height()
    if w < 2 or h < 2:
        return None
    b = border
    left, right = x < b, x >= w - b
    top, bottom = y < b, y >= h - b
    if top and left:
        return "topleft"
    if top and right:
        return "topright"
    if bottom and left:
        return "bottomleft"
    if bottom and right:
        return "bottomright"
    if left:
        return "left"
    if right:
        return "right"
    if top:
        return "top"
    if bottom:
        return "bottom"
    return None


class _ResizeGrip(QWidget):
    """Invisible edge/corner band with a permanent resize cursor.

    Fallback when native WM_NCHITTEST is unavailable; also reinforces the
    resize cursor while hovering the border strip.
    """

    def __init__(self, host: QWidget, edge: str, controller: "_FrameResizeHandler") -> None:
        super().__init__(host)
        self._top = controller._window  # top-level MainWindow
        self._edge = edge
        self._controller = controller
        self.setObjectName("ResizeGrip")
        self.setCursor(_EDGE_CURSORS[edge])
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent; border: none;")

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if is_window_filled(self._top):
            return
        self._controller.begin_resize(self._edge, event.globalPosition().toPoint(), grab=self)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._controller.is_resizing:
            self._controller.do_resize(event.globalPosition().toPoint())
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._controller.is_resizing:
            self._controller.end_resize()
            event.accept()


class _FrameResizeHandler(QObject):
    """Owns invisible edge grips + shared resize drag state for a frameless window."""

    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self._window = window
        # Parent grips to the central widget (QMainWindow) so they sit above content.
        # MainWindow children outside the central layout are covered by centralWidget.
        host = window
        if hasattr(window, "centralWidget"):
            cw = window.centralWidget()
            if cw is not None:
                host = cw
        self._host = host
        self._resize_edge: Optional[str] = None
        self._resize_press_global: Optional[QPoint] = None
        self._resize_start: Optional[tuple[int, int, int, int]] = None
        self._grab_widget: Optional[QWidget] = None
        self._grips: list[_ResizeGrip] = []

        for edge in _EDGE_CURSORS:
            grip = _ResizeGrip(host, edge, self)
            self._grips.append(grip)

        window.installEventFilter(self)
        host.installEventFilter(self)
        self._layout_grips()
        self._raise_grips()

    @property
    def is_resizing(self) -> bool:
        return self._resize_edge is not None

    def begin_resize(self, edge: str, global_pos: QPoint, *, grab: Optional[QWidget] = None) -> None:
        g = self._window.geometry()
        self._resize_edge = edge
        self._resize_press_global = QPoint(global_pos)
        self._resize_start = (g.x(), g.y(), g.width(), g.height())
        self._window._resize_active = True  # type: ignore[attr-defined]
        target = grab or self._window
        target.grabMouse()
        self._grab_widget = target

    def do_resize(self, global_pos: QPoint) -> None:
        if self._resize_edge is None or self._resize_press_global is None:
            return
        if self._resize_start is None:
            return
        ox, oy, ow, oh = self._resize_start
        dx = global_pos.x() - self._resize_press_global.x()
        dy = global_pos.y() - self._resize_press_global.y()
        x, y, w, h = ox, oy, ow, oh
        e = self._resize_edge
        if e in ("right", "topright", "bottomright"):
            w = ow + dx
        if e in ("bottom", "bottomleft", "bottomright"):
            h = oh + dy
        if e in ("left", "topleft", "bottomleft"):
            w = ow - dx
            x = ox + ow - w
        if e in ("top", "topleft", "topright"):
            h = oh - dy
            y = oy + oh - h
        apply_window_geometry(self._window, x, y, w, h)

    def end_resize(self) -> None:
        self._resize_edge = None
        self._resize_press_global = None
        self._resize_start = None
        self._window._resize_active = False  # type: ignore[attr-defined]
        if self._grab_widget is not None:
            self._grab_widget.releaseMouse()
            self._grab_widget = None
        try:
            apply_window_corner_preference(self._window, theme.WINDOW_CORNER_RADIUS)
        except Exception:
            pass
        self._layout_grips()
        self._raise_grips()

    def _layout_grips(self) -> None:
        if is_window_filled(self._window):
            for grip in self._grips:
                grip.hide()
            return

        # Grips are parented to the host (central widget); size to that host.
        w = self._host.width()
        h = self._host.height()
        b = RESIZE_BORDER
        # Corners slightly larger than the side bands for easier diagonal grabs.
        c = max(b + 2, 10)

        geom = {
            "left": (0, c, b, max(0, h - 2 * c)),
            "right": (max(0, w - b), c, b, max(0, h - 2 * c)),
            "top": (c, 0, max(0, w - 2 * c), b),
            "bottom": (c, max(0, h - b), max(0, w - 2 * c), b),
            "topleft": (0, 0, c, c),
            "topright": (max(0, w - c), 0, c, c),
            "bottomleft": (0, max(0, h - c), c, c),
            "bottomright": (max(0, w - c), max(0, h - c), c, c),
        }
        for grip in self._grips:
            x, y, gw, gh = geom[grip._edge]
            grip.setGeometry(x, y, gw, gh)
            grip.setVisible(gw > 0 and gh > 0)

    def _raise_grips(self) -> None:
        for grip in self._grips:
            grip.raise_()

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is not self._window and obj is not self._host:
            return False
        etype = event.type()
        if etype in (QEvent.Type.Resize, QEvent.Type.Show):
            self._layout_grips()
            self._raise_grips()
        elif etype == QEvent.Type.WindowStateChange:
            self._layout_grips()
            self._raise_grips()
        elif etype == QEvent.Type.WindowDeactivate:
            if self.is_resizing:
                self.end_resize()
        elif etype == QEvent.Type.ChildAdded and obj is self._host:
            # Newly added content can stack above grips — re-raise next tick.
            QTimer.singleShot(0, self._raise_grips)
        return False


def install_frame_resize(window: QWidget) -> None:
    """Install native thick-frame resize + invisible edge grips.

    Call after the window's central widget / content tree exists so grips can
    parent above the content (not under QMainWindow's central widget).
    """
    if sys.platform != "win32":
        return
    old = getattr(window, "_frame_resize_handler", None)
    if old is not None:
        try:
            window.removeEventFilter(old)
        except Exception:
            pass
        host = getattr(old, "_host", None)
        if host is not None:
            try:
                host.removeEventFilter(old)
            except Exception:
                pass
        for grip in getattr(old, "_grips", []):
            grip.setParent(None)
            grip.deleteLater()
    window._resize_active = False  # type: ignore[attr-defined]
    handler = _FrameResizeHandler(window)
    window._frame_resize_handler = handler  # type: ignore[attr-defined]
    # Content added after install can cover grips — re-raise a few times.
    QTimer.singleShot(0, handler._raise_grips)
    QTimer.singleShot(100, handler._raise_grips)
    QTimer.singleShot(400, handler._raise_grips)


# ---------------------------------------------------------------------------
# Title bar widget — CTk-style glyphs: −  □  ×
# ---------------------------------------------------------------------------

_GLYPH_MIN = "\u2212"   # −
_GLYPH_MAX = "\u25a1"   # □
_GLYPH_RESTORE = "\u2750"  # ❐ overlapping squares when maximized
_GLYPH_CLOSE = "\u00d7"  # ×


def _make_title_button(text: str, *, danger: bool = False) -> QPushButton:
    """Plain min / max / close control (matches CTk title chrome)."""
    btn = QPushButton(text)
    btn.setObjectName("TitleClose" if danger else "TitleBtn")
    btn.setFixedSize(46, theme.TITLE_BAR_HEIGHT)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFocusPolicy(Qt.NoFocus)
    btn.setFlat(True)
    font = QFont(theme.FONT_FAMILY)
    font.setPixelSize(14)
    btn.setFont(font)
    if danger:
        btn.setStyleSheet(
            f"""
            QPushButton#TitleClose {{
                background: transparent;
                color: {theme.DARK['text_dim']};
                border: none;
            }}
            QPushButton#TitleClose:hover {{
                background-color: {theme.DARK['danger']};
                color: {theme.COLORS['log_fg']};
            }}
            """
        )
    else:
        btn.setStyleSheet(
            f"""
            QPushButton#TitleBtn {{
                background: transparent;
                color: {theme.DARK['text_dim']};
                border: none;
            }}
            QPushButton#TitleBtn:hover {{
                background-color: {theme.DARK['panel2']};
                color: {theme.DARK['fg']};
            }}
            """
        )
    return btn


class CustomTitleBar(QWidget):
    """Custom dark title bar with icon + title + min/max/close.

    Edge resize is owned by ``_FrameResizeHandler`` (invisible edge grips).
    Caption drag is handled here; title-bar double-click maximize is disabled
    (rounded-corner chrome looks wrong — use the maximize button instead).
    """

    close_requested: Callable[[], None]
    minimize_requested: Callable[[], None]
    maximize_requested: Callable[[], None]

    def __init__(self, parent_window: QWidget, *, height: int = theme.TITLE_BAR_HEIGHT) -> None:
        super().__init__(parent_window)
        self.setObjectName("TitleBar")
        self._win = parent_window
        self._height = height
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Drag state
        self._drag_press_pos: Optional[QPoint] = None
        self._drag_start_pos: Optional[QPoint] = None
        self._dragging = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(0)

        from PySide6.QtGui import QIcon
        from pathlib import Path

        icon_path = Path(__file__).resolve().parent.parent.parent / "logo.ico"
        icon_px = theme.TITLE_ICON_SIZE
        self._icon_lbl = QLabel()
        self._icon_lbl.setFixedSize(icon_px, icon_px)
        if icon_path.exists():
            self._icon_lbl.setPixmap(QIcon(str(icon_path)).pixmap(icon_px, icon_px))
        self._icon_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._icon_lbl)
        layout.addSpacing(8)

        self._title_lbl = QLabel("STEM organizer")
        self._title_lbl.setObjectName("Title")
        self._title_lbl.setStyleSheet(
            f"color: {theme.DARK['text_dim']}; background: transparent;"
        )
        title_font = QFont(theme.FONT_FAMILY)
        title_font.setPixelSize(theme.TITLE_LABEL_FONT_PX)
        self._title_lbl.setFont(title_font)
        layout.addWidget(self._title_lbl)
        layout.addStretch(1)

        self.min_btn = _make_title_button(_GLYPH_MIN)
        self.max_btn = _make_title_button(_GLYPH_MAX)
        self.close_btn = _make_title_button(_GLYPH_CLOSE, danger=True)
        self.min_btn.setToolTip("Minimize")
        self.max_btn.setToolTip("Maximize / restore")
        self.close_btn.setToolTip("Close")

        self.min_btn.clicked.connect(self._on_min)
        self.max_btn.clicked.connect(self._on_max)
        self.close_btn.clicked.connect(self._on_close)

        for btn in (self.min_btn, self.max_btn, self.close_btn):
            layout.addWidget(btn)

        self.setMouseTracking(True)
        self._win.setMouseTracking(True)
        self._win.installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self._win and event.type() == QEvent.WindowStateChange:
            self._sync_max_glyph()
        return super().eventFilter(obj, event)

    def _sync_max_glyph(self) -> None:
        filled = is_window_filled(self._win)
        self.max_btn.setText(_GLYPH_RESTORE if filled else _GLYPH_MAX)

    # ----- button handlers -----

    def _on_min(self) -> None:
        f = getattr(self, "minimize_requested", None)
        if f:
            f()

    def _on_max(self) -> None:
        f = getattr(self, "maximize_requested", None)
        if f:
            f()

    def _on_close(self) -> None:
        f = getattr(self, "close_requested", None)
        if f:
            f()

    # ----- mouse: caption drag only (edges → _FrameResizeHandler) -----

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 Qt name
        if event.button() != Qt.LeftButton:
            return
        self._drag_press_pos = event.globalPosition().toPoint()
        self._drag_start_pos = self._win.pos()
        self._dragging = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 Qt name
        if not self._dragging or self._drag_press_pos is None or self._drag_start_pos is None:
            return
        delta = event.globalPosition().toPoint() - self._drag_press_pos
        if is_window_filled(self._win):
            # Restore then continue drag (CTk-style work-area fill, or OS max).
            toggle = getattr(self._win, "_toggle_maximize", None)
            if callable(toggle) and getattr(self._win, "_custom_maximized", False):
                toggle()
            elif self._win.isMaximized():
                self._win.showNormal()
            self._drag_press_pos = event.globalPosition().toPoint()
            self._drag_start_pos = self._win.pos()
            return
        self._win.move(self._drag_start_pos + delta)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 Qt name
        self._dragging = False
        self._drag_press_pos = None
        self._drag_start_pos = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 Qt name
        # Intentionally no maximize — custom SetWindowRgn corners look wonky
        # when toggled via caption double-click (Qt path + native HTCAPTION).
        event.accept()
