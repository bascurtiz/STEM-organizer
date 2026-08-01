# PANNs tagger (AudioSet / Cnn14)

Standalone CLI module for STEM organizer — same subprocess / JSONL pattern as
`instrument_tagger` (Rename Auto-detect).

Uses **PANNs Cnn14** via [`panns-inference`](https://pypi.org/project/panns-inference/)
(pretrained on AudioSet, 527 classes). No training required.

## Focus labels

| Class | Use |
|-------|-----|
| Singing | Sung vocals |
| Speech | Spoken voice |
| Rapping | Rap |
| Humming | Humming |
| Choir | Choir / ensemble vocals |

Also reports related AudioSet scores when present (Male/Female singing, Whispering, …).

## Install

**From source** (shared `genre_gender_tagger\venv`):

```bat
panns_tagger\install-deps.bat
```

Or run root `install-deps.bat` (wires this automatically).

**Frozen build:** root `install-deps.bat` beside `STEM-organizer.exe` installs
`panns_inference` into `site-packages\`.

First inference downloads **Cnn14_mAP=0.431.pth** (~300+ MB) via Python
(`urllib` — no `wget` required on Windows) into `panns_tagger\models\` and
`%USERPROFILE%\panns_data\`, plus AudioSet `class_labels_indices.csv`.

## CLI

```bat
genre_gender_tagger\venv\Scripts\python.exe panns_tagger\panns_tagger.py --file track.wav
genre_gender_tagger\venv\Scripts\python.exe panns_tagger\panns_tagger.py --folder stems --segment-sec 2
```

Stdout = one JSON object per file. Status on stderr.

### Useful flags

| Flag | Meaning |
|------|---------|
| `--segment-sec N` | Also emit per-window `segments[]` with vocal probs |
| `--hop-sec N` | Segment hop (default = window length) |
| `--focus-only` | Restrict `top[]` to vocal-related labels |
| `--device cuda\|cpu\|auto` | Inference device |

### Example JSON

```json
{
  "path": "D:\\stems\\vocals.wav",
  "model": "panns-cnn14",
  "label": "Singing",
  "score": 0.91,
  "vocal": {
    "Singing": 0.91,
    "Speech": 0.04,
    "Rapping": 0.01,
    "Humming": 0.02,
    "Choir": 0.08
  },
  "top": [["Singing", 0.91], ["Music", 0.55]],
  "segments": [
    {"start": 0.0, "end": 2.0, "label": "Speech", "score": 0.72, "vocal": {}}
  ]
}
```

## App integration

- Paths: `tagger_launch.panns_tagger_dir()` / `panns_tagger_script()`
- Thin client: `panns_enrich.classify_paths(...)` (subprocess, no torch in GUI)

UI hook (new tab / Classify option) can call `panns_enrich` the same way Rename
calls `instrument_enrich`.
