"""PyInstaller / dev entry — mirrors the Tk original's run_stem_organizer.py."""
from __future__ import annotations

import multiprocessing
import sys


def main() -> None:
    from stem_organizer.main_entry import run

    run(sys.argv)


if __name__ == "__main__":
    # Required for ProcessPoolExecutor under a frozen Windows .exe
    # (Compression / Corruption / Convert). Without this, pool workers
    # re-enter the GUI and hit the single-instance dialog.
    multiprocessing.freeze_support()
    main()
