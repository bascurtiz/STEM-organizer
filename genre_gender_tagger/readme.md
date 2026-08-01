# Genre / Gender Tagger

Bundled with **STEM organizer** under `genre_gender_tagger\`.  
STEM’s Genre & Gender tab launches this folder’s `venv` (run `install-deps.bat` once).  
You can also double-click `run.bat` for the standalone CLI.

---

Local AI tagging for a music library:

- **Instrumental** — Discogs genre/style via Hugging Face
  `discogs-maest-30s-pw-129e-519l` (MAEST)
- **Acapella** — singing voice gender (female/male) via Essentia
  `gender-discogs-effnet` (ONNX Runtime; Essentia has no Windows wheels),
  plus dry/wet reverb via a lightweight mel-CNN (`models\vocal_reverb.pt`).
  Train it from your own packs: `reverb_data\dry` + `reverb_data\wet`, then
  `train-reverb.bat`.

**Vocal reverb classifier** — the world’s first FOSS dry/wet reverb classifier
**trained on singing vocals** (not speech or instruments), using **4.8k dry**
and **4.8k wet** vocals from sample packs, remix packs, and multitracks.
Best validation accuracy: **0.924**. Checkpoint: `models\vocal_reverb.pt`.

Scans recursively, writes tags to FLAC/MP3/M4A/WAV, and exports a CSV.
Nothing is uploaded.

---

## Features

- Local processing (no audio uploads)
- NVIDIA GPU acceleration (RTX 20/30/40 and RTX 50-series)
- **CPU fallback for GPU-less machines / VMs**
- Two content modes at startup (models named above)
- Recursive folder scanning
- Automatic 30-second audio sampling, multiple clips per song (instrumental)
- Top-5 genre predictions (instrumental)
- GENRE / STYLE separation (instrumental) or COMMENT / GENDER + optional REVERB (acapella)
- Batch and per-file run modes for both content types
- Metadata tagging (FLAC / MP3 / M4A / WAV)
- CSV export

---

## Supported formats

- FLAC
- MP3
- WAV
- M4A

All four are classified **and** tagged when write-metadata is enabled
(Vorbis on FLAC, ID3 on MP3/WAV, MP4 atoms on M4A).

---

## Install from scratch (Windows)

The fastest path is the provided installer. It asks whether you want
GPU or CPU, creates a virtual environment, and installs everything.

### Quick install

1. Install **Python 3.10.x or 3.11.x** from
   <https://www.python.org/downloads/> and tick
   **"Add python.exe to PATH"**.
2. Double-click **`install-deps.bat`** (or run it in a terminal).
3. Answer the prompts:
   - venv or global install
   - **GPU build** (pick 1 or 3) **or CPU** (pick 2)

That's it. Models: Essentia `.onnx` + trained `vocal_reverb.pt` live in
`models\`; Hugging Face MAEST downloads on first genre run.

### Train dry/wet reverb (required for Gender reverb)

Shipped model: world’s first FOSS dry/wet reverb classifier trained on
singing vocals (not speech or instruments) — 4.8k dry + 4.8k wet from
sample packs, remix packs, and multitracks; best val accuracy **0.924**
(`models\vocal_reverb.pt`).

1. Copy vocal samples into:
   - `reverb_data\dry\`
   - `reverb_data\wet\`
2. Double-click **`train-reverb.bat`** (or `python train_vocal_reverb.py`).
3. Checkpoint is written to `models\vocal_reverb.pt` and used by the
   Gender tab automatically.

### Offline / VM (no internet)

Gender models must already exist under `models\`:

- `models\discogs-effnet-bsdynamic-1.onnx`
- `models\gender-discogs-effnet-1.onnx`
- `models\vocal_reverb.pt` (train on a machine with your dry/wet packs,
  then copy)

Instrumental mode needs the Hugging Face MAEST cache.

### Manual install

If you prefer the command line:

```bat
python -m venv venv
venv\Scripts\activate

:: ---- GPU: pick ONE of these ----
:: RTX 20/30/40 (CUDA 12.4)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

:: RTX 50-series (CUDA 12.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

:: ---- CPU only ----
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

:: ---- project deps ----
pip install -r requirements.txt
```

Verify:

```bat
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

---

## Install in a VM (no GPU)

This is the CPU-only setup. No NVIDIA drivers or CUDA toolkit are
needed — PyTorch's CPU wheel includes everything.

1. In the VM, install **Python 3.10.x or 3.11.x** and add it to PATH.
2. Run **`install-deps.bat`** and choose:
   - venv: **yes**
   - GPU build: **2 = No NVIDIA GPU - CPU only**
3. Open `genre_gender_tagger.py` and set:
   ```python
   BATCH_MODE = False
   ```
   Per-file mode is the recommended way to run on CPU: it is simpler,
   you see each result live, and it avoids the large batched transfers
   that only pay off on a GPU.
4. Run the tagger (see below). It will report `Device: cpu`.

Note: CPU inference is much slower than GPU. For a large library,
expect roughly tens of seconds per file rather than fractions of a
second.

---

## Configure

Open `genre_gender_tagger.py`.

| Setting            | Default | Meaning                                                   |
| ------------------ | ------- | --------------------------------------------------------- |
| `RUNTIME_PROMPTS`  | `True`  | Ask at startup which mode to run in and how to write tags. Set `False` to use the defaults below without prompting. |
| `WRITE_METADATA`   | `True`  | Write tags at all. `False` = only export the CSV.         |
| `TAG_WRITE_MODE`   | `combined` | Instrumental: `combined` = `GENRE = "Rock/Surf"`. `split` = `GENRE` + `STYLE`. |
| `GENDER_TAG_FIELD` | `comment` | Acapella: `comment` or `gender` field for the gender value. |
| `REVERB_TAG_MODE`  | `combined` | Acapella: `combined` = `female/wet` in the gender field; `split` = gender field + `REVERB`. |
| `BATCH_MODE`       | `True`  | `True` = fast batched pipeline. `False` = per-file output. |
| `BATCH_SIZE`       | `64`    | Batch size (batch mode, GPU / EffNet patches).            |
| `AUDIO_WORKERS`    | `8`     | CPU decode workers (batch mode).                          |
| `NUMBER_OF_CLIPS`  | `3`     | 30 s clips sampled per song (instrumental).               |
| `OUTPUT_CSV`       | `genre_gender_results.csv` | Instrumental CSV.                    |
| `OUTPUT_CSV_GENDER`| `genre_gender_voice_results.csv` | Acapella CSV.                  |

The input folder is asked at runtime (drag-and-drop or paste the path).

---

## Run

```bat
run.bat
```

That uses the venv Python automatically (no need to `activate`).

Or manually:

```bat
venv\Scripts\activate
python genre_gender_tagger.py
```

You will be prompted for content type, then the input folder:

**Content type**
- `1 = Instrumental` — genre/style (`discogs-maest-30s-pw-129e-519l`)
- `2 = Acapella` — voice gender + dry/wet (`gender-discogs-effnet` + `vocal_reverb.pt`)

For instrumental (when `RUNTIME_PROMPTS = True`) you then get:

**Run mode**:
- `1 = Batch` — much faster, but no per-file overview.
- `2 = Per-file` — slower, prints `GENRE`/`STYLE`/`CONF` per file.

**Tag writing** (only if `WRITE_METADATA = True`):
- `1 = GENRE field only` — stored combined, e.g. `GENRE = Rock/Metal`.
- `2 = GENRE + STYLE fields` — separated, e.g. `GENRE = Rock`, `STYLE = Metal`.

For acapella (when `RUNTIME_PROMPTS = True`) you then get:

**Run mode**:
- `1 = Batch` — parallel decode + packed 64-patch TF batches (faster).
- `2 = Per-file` — prints `GENDER`/`REVERB`/`CONF` per file.

**Tag writing** (only if `WRITE_METADATA = True`):
- `1 = COMMENT field` — e.g. `COMMENT = female`.
- `2 = GENDER field` — e.g. `GENDER = female`.

**Reverb tagging** (vocal mel-CNN):
- `1 = Combined` — e.g. `COMMENT = female/wet`.
- `2 = Split` — e.g. `COMMENT = female`, `REVERB = wet`.

The tagger then runs and, when finished, writes the CSV.

### Batch mode (`BATCH_MODE = True`)

Fastest. Processes files in parallel and pushes batches to the model.
No per-file console output during the run — only the progress bar and
the final summary.

### Per-file mode (`BATCH_MODE = False`)

Each file is printed live as it finishes:

```
03 - Track Name.flac
GENRE: Rock
STYLE: Surf
CONF: 0.6206
```

or for acapella:

```
03 - Track Name.flac
GENDER: female
REVERB: wet
CONF: 0.8712 / 0.6401
```

---

## Output

A CSV is written with one row per file. Instrumental rows include genre,
style, confidence, and top-5. Acapella rows include gender and
reverb (`wet`/`dry`) probabilities.

When `WRITE_METADATA = True`, matching tags are updated on FLAC, MP3,
M4A, and WAV.

Set `WRITE_METADATA = False` in `genre_gender_tagger.py` to only export
CSV and leave files untouched.
