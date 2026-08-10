"""Smoke test: same backend classes the app runs (classify_backend.Worker /
SdrWorker) on the real folders, mirroring the app's param shapes.

Phase 1: RMS classify (Stem CNN6) E:\multitrack-test -> E:\multitrack-test_organized
Phase 2: SI-SDR (htdemucs, CUDA) on E:\Audio\!OGG  (compute-only: no deletes/tags)
"""
import queue
import sys
import time

sys.path.insert(0, r"D:\STEM-organizer-Py6")
import classify_backend as cb

# Mirror the app: splash.py calls _init_ml() at startup, which resolves
# ffmpeg + ML deps (needed to decode ogg/mp3 stems).
cb._init_ml()


def drain(name, worker, timeout_s=600):
    q = worker.q
    worker.start()
    errors = []
    sdr_lines = []
    done = False
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            msg = q.get(timeout=0.5)
        except queue.Empty:
            if not worker.is_alive():
                break
            continue
        if msg is cb.DONE_SENTINEL:
            done = True
            break
        if isinstance(msg, str):
            if "[error]" in msg.lower() or "[ERROR]" in msg:
                errors.append(msg)
            if "=== SI-SDR Summary ===" in msg or "DONE" in msg:
                pass
        elif isinstance(msg, tuple) and msg and msg[0] == cb.SDR_LOG_TAG:
            sdr_lines.append(msg)
    worker.join(timeout=5)
    return done, errors, sdr_lines


# ---- Phase 1: RMS classify with Stem CNN6 -------------------------------
rms_params = {
    "input_dir": r"E:\multitrack-test",
    "output_dir": r"E:\multitrack-test_organized",
    "use_cuda": True,
    "model_id": "vocal_cnn6",
    "stem_mode": "2 (instrumental/vocals)",
    "quality": "FLAC 16-bit",
    "threshold": 0.5,
    "min_margin": 0.2,
    "batch_size": 4,
    "peak_norm": True,
    "make_mixture": False,
    "dedup": False,
    "ambig_mode": "Skip ambiguous stem only",
    "scan_mode": "subfolders",
    "naming_mode": "Original folder name",
    "append_duration": False,
    "delete_if_short": True,
    "min_duration_sec": 8,
    "delete_if_incomplete": False,
    "skip_existing": True,
}
print("=== PHASE 1: RMS classify (Stem CNN6, CUDA) ===")
w1 = cb.Worker(rms_params, queue.Queue())
done1, errs1, _ = drain("rms", w1)
print(f"RMS done={done1} errors={len(errs1)}")
for e in errs1[:5]:
    print("  RMS err:", e[:160])

# ---- Phase 2: SI-SDR on the vocal-ogg folder ----------------------------
sdr_params = {
    "target_dir": r"E:\Audio\!OGG",
    "use_cuda": True,
    "model_id": "htdemucs",
    "stem_mode": "2 (instrumental/vocals)",
    "scan_mode": "Each subfolder (one level)",
    "sdr_thresholds": dict(cb.SDR_DEFAULT_THRESHOLDS),
    "sdr_delete_folder": False,
    "write_sdr_tags": False,
}
print("\n=== PHASE 2: SI-SDR (htdemucs, CUDA) on E:\\Audio\\!OGG ===")
w2 = cb.SdrWorker(sdr_params, queue.Queue())
done2, errs2, sdr_lines = drain("sdr", w2)
print(f"SDR done={done2} errors={len(errs2)} sdr_lines={len(sdr_lines)}")
for e in errs2[:8]:
    print("  SDR err:", e[:180])
for line in sdr_lines[:10]:
    print("  SDR score:", line[1], line[2], "->", line[0][:60])

print("\n=== RESULT ===")
print("RMS:", "PASS" if done1 and not errs1 else "FAIL")
print("SDR:", "PASS" if done2 and not errs2 else "FAIL")
sys.exit(0 if (done1 and not errs1 and done2 and not errs2) else 1)
