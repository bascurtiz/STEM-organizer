"""Export MelReverbNet (vocal_reverb.pt) to ONNX.

The mel-spectrogram + crop logic stays in numpy/librosa (outside the graph),
exactly as in the torch path. Only the Conv2d stack is exported.

Input:  (B, 1, 64, 250) float32   [B crops, 1 ch, n_mels=64, target_frames=250]
Output: (B, 2) float32 logits     [dry, wet]
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Import the model definition from the app's own module (no fork needed).
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / 'genre_gender_tagger'))
import torch
from vocal_reverb import MelReverbNet, DEFAULT_CONFIG, CLASS_NAMES, frames_for_clip

CKPT = APP / 'genre_gender_tagger' / 'models' / 'vocal_reverb.pt'
DEST = APP / '_onnx_spike' / 'onnx_out' / 'vocal_reverb.onnx'


def main():
    saved = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = dict(DEFAULT_CONFIG); cfg.update(saved.get('config') or {})
    classes = tuple(saved.get('classes') or CLASS_NAMES)
    channels = tuple(cfg.get('channels') or DEFAULT_CONFIG['channels'])
    target_frames = frames_for_clip(cfg)
    print(f"config: n_mels={cfg['n_mels']} channels={channels} "
          f"n_classes={len(classes)} target_frames={target_frames} classes={classes}")

    model = MelReverbNet(n_mels=int(cfg['n_mels']), channels=channels,
                         n_classes=len(classes))
    model.load_state_dict(saved['state_dict'])
    model.eval()  # bakes BatchNorm running stats -> deterministic export

    dummy = torch.randn(1, 1, int(cfg['n_mels']), target_frames, dtype=torch.float32)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting -> {DEST} ...")
    torch.onnx.export(
        model, dummy, DEST,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=['mel'], output_names=['logits'],
        dynamic_axes={'mel': {0: 'batch'}, 'logits': {0: 'batch'}},
        dynamo=False,
    )
    mb = DEST.stat().st_size / 1e6
    print(f"  done: {mb:.3f} MB")

    # Write a config sidecar JSON so the ONNX loader doesn't need the .pt
    # (the .onnx carries weights but not the cfg/classes metadata). Lets the
    # .pt be deleted in a torch-free build.
    import json
    sidecar = DEST.with_suffix('.config.json')
    # tuple -> list for JSON
    cfg_out = dict(cfg); cfg_out['channels'] = list(cfg_out['channels'])
    sidecar.write_text(json.dumps({'config': cfg_out, 'classes': list(classes)}),
                       encoding='utf-8')
    print(f"  sidecar: {sidecar.name} ({sidecar.stat().st_size} bytes)")
    print(f"  done: {mb:.3f} MB")

    # Verify I/O contract
    import onnxruntime as ort
    sess = ort.InferenceSession(str(DEST), providers=['CPUExecutionProvider'])
    for i in sess.get_inputs():
        print(f'  IN  {i.name}: {i.shape} {i.type}')
    for o in sess.get_outputs():
        print(f'  OUT {o.name}: {o.shape} {o.type}')


if __name__ == '__main__':
    main()
