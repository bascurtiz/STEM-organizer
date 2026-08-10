# AGENTS.md — STEM organizer: analysis-first stem separation

> **Purpose:** Fast, deterministic, torch-free stem separation for **downstream
> analysis metrics** (RMS, SDR, future stem stats) — not studio-quality exports.
> Classify separators: **HTDemucs** (StemSplit ONNX) and **Vocal CNN6** (ONNX
> classifier) via ONNX Runtime. This file is the handoff for any coding agent.
> Read §0–§2 first.

---

## 0. Quick orientation (read this first)

- **What this app is:** a Windows PySide6 (Qt6) desktop app. Classify separates
  audio into stems, then computes analysis metrics and runs taggers
  (genre/gender/instrument/key/vocals). All ML is **inference-only**.
- **Separators:** **HTDemucs** (StemSplitio/htdemucs-onnx, 4-stem ONNX) and
  **Vocal CNN6** (a trained vocal/instrumental classifier that emits synthetic
  stems whose RMS encodes the vocal probability — no real separation).
  Optimize for throughput, VRAM, determinism, and broad hardware — **not**
  listening quality.
- **Retired separators:** UMX-L, X-UMXL, SCNet Tran, and BS-RoFormer were
  retired from Classify (2026-08) — they were **not faster than HTDemucs on the
  RTX 3060-class target** and added shipping complexity. Legacy settings ids
  (`umxl`/`xumxl`/`scnet_tran`/`bsroformer`/`demucs`) still load but route to
  HTDemucs (`classify_backend.load_demucs_model`). Their runner files
  (`umx_onnx.py` / `xumx_onnx.py` / `scnet_onnx.py` / `bs_roformer_onnx.py`),
  their weights, and `gpu_fft.py` (CuPy STFT helper) have been **deleted from
  the repo** — no CuPy is bundled in the freeze, and nothing imports them.
- **Hardware floor:** NVIDIA **RTX 3060 12 GB**. Do **not** tune for
  4080/4090/5090 as the success bar. AMD/Intel GPUs fall back to CPU for
  Classify (DirectML is rejected for Demucs, §13); taggers still use
  CUDA→DirectML fallback via `ort_util.py`.
- **GPU path:** ONNX Runtime + **CUDAExecutionProvider** (onnxruntime-gpu),
  **CPU fallback**. DirectML must never be mandatory for Classify.
- **HTDemucs weights:** [StemSplitio/htdemucs-onnx](https://huggingface.co/StemSplitio/htdemucs-onnx)
  — `htdemucs.onnx` (B=1 trace, ~316 MB) or the dynamic-batch
  `htdemucs.batch.onnx` (installer ships this as `htdemucs.onnx`).
- **Vocal CNN6 weight:** `vocal_classifier.onnx` (~24 MB, trained on user
  data). 32 kHz mono input → `vocal_prob` output; used for 2-stem
  instrumental/vocals classification.
- **Current codebase (v1.0.8):** Classify Model dropdown = `Vocal CNN6` |
  `HTdemucs (demucs)` (`classify_backend.MODELS`). DirectML-first runner
  files are gone from the shipped build.
- **App / models remotes:** `bascurtiz/STEM-organizer`,
  `bascurtiz/STEM-organizer-models` tag `models`. The installer downloads 9
  ONNX assets (htdemucs, vocal_classifier, cnn14, stem_cnn6, maest,
  discogs-effnet, gender, vocal_reverb, nf50). The release **also hosts the
  retired UMX-L / X-UMXL / SCNet Tran / BS-RoFormer weights (`_`-prefixed) as
  backup** — the app/build/installer do not use them; **do not delete them**
  (user keeps them for reference).
- **Do not ship:** `dist/`, experimental `.*-venv`, `nul`, `__pycache__/`,
  the retired `umxl_*`/`xumxl*`/`scnet_tran*`/`bs_roformer*` weights
  (build.bat no longer copies them).
- **Torch gotcha (still true):** never co-load torch + onnxruntime in one
  process on a workstation — export in isolated venvs only.
- **File encoding (Windows):** `.py` sources are **UTF-8 with BOM, CRLF**.
  Do NOT save them as ANSI/CP1252 or LF — a UTF-8→CP1252 round-trip
  double-encodes non-ASCII (`â€”` for `—`, `Â±` for `±`, `â‡§` for `⇧`), which
  renders as mojibake in the UI. `stem_player_window.py` was repaired from
  exactly this (2026-08-05); if the shortcuts footer / transport labels look
  garbled again, check file encoding before editing code.

---

## 1. Project goal

This project uses **HTDemucs ONNX (StemSplit)** as its **default and only**
stem-separation backend, with **Vocal CNN6** as the fast 2-stem alternative.

Repository / model card:

- https://huggingface.co/StemSplitio/htdemucs-onnx

The application is **not** intended to produce studio-quality stem exports.
Source separation exists solely to compute downstream analysis metrics.

Primary downstream metrics include:

- Per-stem RMS
- Per-stem SDR
- Additional future stem statistics

The separator is therefore optimized for:

1. Throughput
2. Memory efficiency
3. Deterministic output
4. Broad hardware compatibility

rather than maximum listening quality.

---

## 2. Hardware target

Development should always assume the following minimum supported GPU:

- NVIDIA RTX 3060 12 GB (CUDA EP)

The application must **not** assume high-end hardware such as RTX 4080, 4090,
or 5090.

Performance improvements should be evaluated against **RTX 3060-class**
hardware.

---

## 3. Inference backend

- ONNX Runtime (`onnxruntime-gpu==1.28.0` + CUDA 12.9 wheels, pinned)
- **CUDAExecutionProvider** — Classify GPU path
- **CPUExecutionProvider** — universal fallback
- FP16 models where available
- DirectML: used only as a **tagger fallback** (`ort_util.py`), never for
  Classify/Demucs

**Note:** DirectML was exhaustively **rejected for HTDemucs** (VRAM ~31 GB,
slow, inaccurate — historical §13). That verdict drove the retirement of the
UMX-L / X-UMXL / SCNet Tran DirectML-first separators as well: they were not
faster than HTDemucs on the 3060-class target, so the CUDA EP path became the
only Classify GPU path.

---

## 4. Source separation backend

**Backend:** HTDemucs ONNX (StemSplit trace), CUDA-or-CPU only.

| File | Notes | ~Size |
|---|---|---|
| `htdemucs.onnx` | B=1 trace (fallback) | ~316 MB |
| `htdemucs.batch.onnx` | dynamic batch (preferred; installer ships as `htdemucs.onnx`) | ~316 MB |

**I/O (StemSplit contract):**

- Domain: waveform, `sr=44100`, stereo
- Segment: 7.8 s (`SEGMENT_LENGTH = 343980` samples)
- Input / output: `mix (B,2,343980)` → `stems (B,4,2,343980)`
- Sources: `drums, bass, other, vocals`
- Reconstruction: overlap-add (OLA) with linear fades; Classify default
  overlap 10% (`STEM_CLASSIFY_OVERLAP`); window fades and OLA math live in
  `demucs_onnx.py` (`separate` / `_separate_ola`).
- The B=1 graph is promoted to a dynamic batch axis at load
  (`ensure_batch_dynamic_onnx`, cached `<stem>.batch.onnx`) so Classify can run
  true multi-file ORT forwards; `STEM_DEMUCS_MAX_BATCH` (default 4) caps the
  batch.

**Secondary backend:** Vocal CNN6 (`vocal_classifier_onnx.py`) — a 32 kHz mono
CNN6 classifier that duck-types the separator surface and emits **synthetic**
4-stem output whose RMS ratios encode the vocal probability (used for 2-stem
instrumental/vocals classification; SI-SDR should use HTDemucs).

---

## 5. Optimization philosophy

The application is designed for **analysis**, not audio production.

Whenever a trade-off exists between better-sounding stems and faster analysis,
**prefer faster analysis**, provided downstream metrics remain statistically
consistent.

The important outputs are the analysis metrics — not the perceptual quality of
reconstructed stems.

---

## 6. Success criteria

Future optimizations should be measured using these priorities:

1. Processing time per song
2. Songs processed per hour
3. Peak VRAM usage
4. RMS consistency
5. SDR consistency
6. Model size

Subjective listening quality is **not** a benchmark.

---

## 7. Benchmarking rules

Every new separator or optimization must be benchmarked against the current
baseline.

Record at minimum:

- total runtime
- average runtime per song
- peak GPU memory
- average GPU utilization
- model load time
- RMS correlation (vs prior separator / reference)
- SDR correlation (vs prior separator / reference)

Performance regressions are not acceptable unless accompanied by measurable
improvements in downstream analysis accuracy.

**Suggested corpus:** the existing
`F:\Speed comparison CUDA vs ONNX\1-Multitracks` (10 songs) plus a 3060-class
machine when available. Prior HTDemucs ONNX baseline on that set (5-folder
subset): ~2:33 wall on 3060-class (BS-RoFormer was ~12:35 and was retired).

---

## 8. Architecture principles

The separator is an **implementation detail**.

All downstream analysis must remain **model-agnostic**.

No component outside the separation layer should depend on:

- Demucs / HTDemucs internals
- Vocal CNN6 internals
- any specific model architecture

Only generic stem outputs should be exposed:

- vocals
- drums
- bass
- other

Future separator replacements should require changing **only** the inference
layer (duck-type the same surface as today’s `DemucsOnnxModel` /
`VocalClassifierOnnxModel`: `separate_numpy` / `sources` / `samplerate` /
`to(device)` / `cpu()`).

Classify UI / `STEM_MODES` settles on **2-stem and 4-stem** analysis layouts
derived from those four names (6-stem guitar/piano is not a first-class
separator output; a legacy 6-stem mode clamps to 4 for HTDemucs).

---

## 9. Future model evaluation

Any future candidate model should be evaluated before adoption:

- ONNX availability
- CUDA EP compatibility (DirectML-only models are suspect for Classify)
- 4-stem support
- Stable inference
- Runtime performance
- VRAM usage
- RMS consistency
- SDR consistency

A model should **not** replace HTDemucs solely because it scores higher SDR
on public listening benchmarks.

---

## 10. Coding guidelines

- Keep inference deterministic.
- Avoid unnecessary GPU memory allocations.
- Minimize model loading overhead.
- Reuse ONNX Runtime sessions whenever practical.
- Separate inference code from metric computation.
- Keep preprocessing / postprocessing independent of the chosen separator.
- Avoid model-specific assumptions outside the inference module.
- Ship `onnxruntime-gpu` + pinned CUDA wheels; DirectML stays a tagger-side
  fallback only (`ort_util.py`).
- Keep spike / throwaway export experiments out of the source tree (do not
  reintroduce an `_onnx_spike/`-style folder); do not co-load torch + ORT
  (§ historical).

---

## 11. Non-goals

This project is **not** intended to:

- produce the highest-quality stems
- compete with Ultimate Vocal Remover
- compete with Demucs leaderboards
- maximize SDR benchmark scores

The project exists to provide **fast, reliable stem generation for automated
audio analysis**. Every architectural decision should support that objective.

---

## 12. Migration status (codebase → this plan)

| Item | Status |
|---|---|
| Taggers ONNX (vocal_reverb, KeyNet, Cnn14, MAEST) | **Keep** — unaffected |
| Rename Auto-detect (PaSST OpenMIC → Stem CNN6) | **Done** (2026-08) — 11-class raw-waveform ONNX replaces the 20-class PaSST/hear21passt stack; 12 prefix categories → 11 (dropped Mallet/Percussion/Orchestra; split Flute/Organ into own prefixes) |
| Packaging (single Inno EXE, external model download) | **Done** — trimmed to 9 model assets |
| HTDemucs ONNX + CUDA EP Classify | **Current** — only shipped Classify separator |
| Vocal CNN6 ONNX | **Current** — 2-stem fast path |
| UMX-L / X-UMXL / SCNet Tran (DirectML-first) | **Retired** (2026-08) — not faster than HTDemucs on 3060-class; ids route to HTDemucs |
| BS-RoFormer 6-stem | **Retired** — not shipped |
| DirectML for Demucs / Classify | Historical **REJECTED** — do not revive |
| build.bat / spec / Inno | **Cleaned** — no retired runners, weights, configs, or download entries |
| `stem_organizer/` UI | **Clean** — only htdemucs/vocal_cnn6 references remain |
| `classify_backend` duck-type behind generic stems | **In place** |
| `_onnx_spike/` migration | **Done & removed** — folded into production then superseded |

### Key file map (production today)

```
run_stem_organizer.py / stem_organizer/   app entry + UI
classify_backend.py                       Classify load/apply/RMS/SDR
demucs_onnx.py                            HTDemucs StemSplit ONNX runner (live)
vocal_classifier_onnx.py                  Vocal CNN6 ONNX runner (live)
ort_util.py                               EP / GPU probes (taggers CUDA→DirectML fallback)
stem_organizer.iss / build.bat / spec     packaging (HTDemucs + Vocal CNN6 only)
*_tagger/                                 keep ONNX taggers
```

---

## 13. Historical footnotes (do not rediscover blindly)

These remain true for **spike hygiene** and for understanding why the
DirectML-first separators were abandoned; they are not a license to
re-litigate Demucs+DML:

1. Do not co-load torch Demucs + onnxruntime in one workstation process.
2. HTDemucs on `DmlExecutionProvider` ≈ 31 GB VRAM, ~60–120× slower than CUDA,
   and numerically wrong at short segments — **REJECTED for Demucs** (2026-08-01).
3. UMX-L / X-UMXL / SCNet Tran were adopted as DirectML-first separators, then
   **retired** (2026-08): they were not faster than HTDemucs on the 3060-class
   target and added ~1 GB of extra weights + three runners to ship. Their
   weights are not in the installer or build, but the GitHub `models` release
   keeps them (`_`-prefixed) as **backup — keep them there**.
4. BS-RoFormer is heavier than HTDemucs on the same multitrack corpus (~5× wall
   on 5090-class); fine for quality, wrong default for analysis throughput.
5. Windows Store `ffmpeg` stub is fake — use the `ffmpeg\ffmpeg.exe` fetched by
   `ffmpeg_bootstrap.py` at setup time (not bundled in the source tree).

---

*Plan adopted: HTDemucs (StemSplit ONNX) + Vocal CNN6, CUDA EP, analysis-first,
3060-class hardware floor. Current tree (v1.0.8) ships HTDemucs + Vocal CNN6
ONNX; UMX-L / X-UMXL / SCNet Tran / BS-RoFormer are retired from Classify and
the build.*
