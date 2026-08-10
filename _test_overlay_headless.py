"""Headless test of the HTDemucs warmup overlay in classify_tab.py.

Binds the REAL shipped methods (_show_warmup_overlay / _hide_warmup_overlay /
_wire_worker) onto a bare QWidget host, then verifies:
  1. show -> overlay visible, full-rect, label centered, text set
  2. GUI event loop stays responsive while a thread does blocking init work
  3. first progress update hides the overlay (real _wire_worker connection)
  4. finished_ok hides the overlay (safety net)
"""
import os
import sys
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"D:\STEM-organizer-Py6")

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication, QWidget

app = QApplication.instance() or QApplication([])

from stem_organizer.tabs.classify_tab import ClassifyTab


class OverlayHost(QWidget):
    pass


host = OverlayHost()
host.setGeometry(0, 0, 900, 600)
host.show()  # the real tab is a child of the visible main window
host._show_warmup_overlay = types.MethodType(ClassifyTab._show_warmup_overlay, host)
host._hide_warmup_overlay = types.MethodType(ClassifyTab._hide_warmup_overlay, host)

# --- 1. show -------------------------------------------------------------
text = "Initializing HTDemucs model\n(holds the GUI briefly — about 15–30 s)"
host._show_warmup_overlay(text)
assert host._warmup_overlay is not None, "overlay widget not created"
assert host._warmup_overlay.isVisible(), "overlay not visible after show"
assert host._warmup_overlay.geometry() == host.rect(), (
    f"overlay geometry {host._warmup_overlay.geometry()} != host rect {host.rect()}"
)
lab = host._warmup_label
assert lab.text() == text, f"label text wrong: {lab.text()!r}"
assert host._warmup_overlay.isAncestorOf(lab), "label not child of overlay"
cx, cy = lab.x() + lab.width() // 2, lab.y() + lab.height() // 2
assert abs(cx - host.width() // 2) <= 1 and abs(cy - host.height() // 2) <= 1, (
    f"label not centered: center=({cx},{cy}) host=({host.width()//2},{host.height()//2})"
)
print("PASS 1: visible, full-rect, label centered, text set")

# --- 2. GUI responsive during blocking init in a thread ------------------
class InitThread(QThread):
    def run(self):
        time.sleep(1.5)


ticks = []
timer = QTimer()
timer.setInterval(50)
timer.timeout.connect(lambda: ticks.append(time.time()))
timer.start()

th = InitThread()
th.start()
while th.isRunning():
    app.processEvents()
    time.sleep(0.01)
timer.stop()

assert len(ticks) >= 15, f"GUI starved: only {len(ticks)} event-loop ticks in 1.5s"
assert host._warmup_overlay.isVisible(), "overlay vanished during init block"
print(f"PASS 2: {len(ticks)} event-loop ticks during 1.5s blocking init; overlay stayed visible")

# --- 3. real _wire_worker: first progress hides overlay -------------------
class FakeWorker(QObject):
    log_line = Signal(str, str)
    progress = Signal(object)
    sdr_line = Signal(str, str)
    finished_ok = Signal()


host.request_progress = lambda *a, **k: None
host.request_sdr_log = lambda *a, **k: None
host._forward_worker_log = lambda *a, **k: None
host._on_worker_done = lambda: None
host.set_running = lambda r: None

host._wire_worker = types.MethodType(ClassifyTab._wire_worker, host)
fw = FakeWorker()
host._show_warmup_overlay(text)
host._wire_worker(fw)
assert host._warmup_overlay.isVisible(), "_wire_worker hid overlay prematurely"
fw.progress.emit(0.5)
assert not host._warmup_overlay.isVisible(), "progress did not hide overlay"
print("PASS 3: real _wire_worker connection hides overlay on first progress")

# --- 4. finished_ok safety net --------------------------------------------
host._show_warmup_overlay(text)
fw.finished_ok.emit()
assert not host._warmup_overlay.isVisible(), "finished_ok did not hide overlay"
print("PASS 4: finished_ok hides overlay")

print("ALL OVERLAY TESTS PASSED")
