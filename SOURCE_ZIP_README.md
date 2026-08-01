# STEM organizer — source upload set (v1.0.8)

Shippable app source for [bascurtiz/STEM-organizer](https://github.com/bascurtiz/STEM-organizer).

## Runtime

- Default ML: **onnxruntime-gpu** (CUDA EP). Demucs does **not** use DirectML.
- Models: not in this zip. Installer downloads from
  [STEM-organizer-models](https://github.com/bascurtiz/STEM-organizer-models) tag `models`.

## From source

```bat
install-deps.bat
.venv\Scripts\python.exe run_stem_organizer.py
```

## Freeze / installer

```bat
build.bat
# then compile stem_organizer.iss with Inno Setup 6
```

See root `README.md` for tabs, tags, and layout.
