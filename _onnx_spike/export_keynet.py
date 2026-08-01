"""Export KeyNet (musical-key CNN) to ONNX + verify I/O.

Preprocess (CQT + log1p) stays in numpy/librosa; only the conv forward is
exported. Real input shape from inference.py preproc: (B, 1, 136, 32).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / 'key_tagger'))
import torch
from model import KeyNet
from inference import MODEL_NF, MODEL_P, N_BINS, _chunk_frames

CKPT = APP / 'key_tagger' / 'checkpoints' / 'nf50-q05-221125.pt'
DEST = APP / '_onnx_spike' / 'onnx_out' / 'keynet.onnx'

FREQ = N_BINS            # 136
TIME = _chunk_frames()   # (8*44100)//11025 = 32


def main():
    model = KeyNet(num_classes=24, in_channels=1, Nf=MODEL_NF, p=MODEL_P)
    state = torch.load(str(CKPT), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()  # Dropout2d + BatchNorm frozen -> deterministic export

    # Real CQT chunk shape: (B, 1, 136, 32)
    dummy = torch.randn(1, 1, FREQ, TIME, dtype=torch.float32)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting KeyNet -> {DEST}  (input {tuple(dummy.shape)}) ...")
    torch.onnx.export(
        model, dummy, DEST,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=['cqt'], output_names=['logits'],
        dynamic_axes={'cqt': {0: 'batch', 3: 'time'}, 'logits': {0: 'batch'}},
        dynamo=False,
    )
    print(f"  done: {DEST.stat().st_size/1e6:.3f} MB")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(DEST), providers=['CPUExecutionProvider'])
    for i in sess.get_inputs():
        print(f'  IN  {i.name}: {i.shape} {i.type}')
    for o in sess.get_outputs():
        print(f'  OUT {o.name}: {o.shape} {o.type}')

    # Sanity: torch vs ONNX on the same input
    x = torch.randn(4, 1, FREQ, TIME)
    with torch.inference_mode():
        lt = model(x).numpy()
    lo = sess.run(['logits'], {'cqt': x.numpy()})[0]
    d = abs(lt - lo)
    print(f"\nparity (4 random chunks): max|d|={d.max():.3e} mean|d|={d.mean():.3e}")
    print(f"  argmax agree: {(lt.argmax(1)==lo.argmax(1)).mean()*100:.0f}%")


if __name__ == '__main__':
    main()
