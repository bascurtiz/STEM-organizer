# Source zip manifest (canonical upload set)

Canonical tree for GitHub upload lives in:

`dist/source-zip/STEM-organizer-1.0.8-src/`

(User-cleaned 2026-08-01. Next `STEM-organizer-*-src.zip` must mirror this, not the full repo.)

## Include (top-level)

- `.gitattributes`, `.gitignore`
- App / packaging: `run_stem_organizer.py`, `build.bat`, `requirements.txt`,
  `stem_organizer.iss`, `stem_organizer_py6.spec`, `install-deps.bat`,
  `README.md`, `SOURCE_ZIP_README.md`, `LICENSE`
- Core modules: `classify_backend.py`, `demucs_onnx.py`, `ort_util.py`,
  `deps_bootstrap.py`, `tagger_launch.py`, `update_checker.py`,
  `pair_matcher.py`, `stem_align.py`, `panns_enrich.py`,
  `audio_resample.py`, `resource_monitor.py`, `single_instance.py`,
  `done_sound.py`, `frozen_stdlib_imports.py`,
  `ffmpeg_bootstrap.py`, `flac_bootstrap.py`, `mp3val_bootstrap.py`
- Branding / installer art: `logo.png`, `logo.ico`, `wizard-image.bmp`,
  `wizard-small.bmp`, `screenshots-v107.gif`
- Packages: `stem_organizer/`, `genre_gender_tagger/`, `instrument_tagger/`,
  `key_tagger/`, `panns_tagger/`, `track_renamer/`
- Tiny model sidecars only (no `.onnx`):
  - `genre_gender_tagger/models/*.json`
  - `panns_tagger/models/*.csv`
- Spike scripts only: `_onnx_spike/*.py` (export / parity / bench / hash_weights)

## Exclude (do not put in next zip)

- `AGENTS.md` (local agent handoff; not for public repo upload set)
- `changelog.txt`, `genre_colors.png`, `theme-colors.png`
- `_onnx_spike/demucs_directml_vram_brief.{md,html,pdf}` (issue paste separately)
- `_onnx_spike/onnx_out/`, `demucs-gsoc/`, test audio, venvs
- All `.onnx` / `.pth` / `.pt` / large `.pb` weights
- `dist/`, `build/`, `.build-venv/`, `ffmpeg/`, `mp3val/`, `flac/`, `models/`
- `settings.json`, `python-version.txt`, `__pycache__/`, `_smoke/`, `nul`

## Packaging notes

- `install-deps.bat`: frozen EXE → tools only; source → `.venv` + `requirements.txt`.
- `build.bat`: installs freeze deps via `requirements.txt` (no duplicated pip list).
- Models are **not** in this zip — GitHub Release tag `models`.

## Next pack command intent

Copy only the paths that currently exist under
`dist/source-zip/STEM-organizer-1.0.8-src/`, or rebuild the zip **from that
folder** after syncing wanted source changes into it — do not re-robocopy the
whole repo without applying this exclude list.
