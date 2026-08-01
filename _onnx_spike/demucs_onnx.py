"""Pure numpy + onnxruntime Demucs stem separator.

Drop-in replacement for the torch model's segmented inference path
(apply_model with split=True, shifts=0, overlap=0.25). No torch, no demucs
package needed at runtime — only the single exported htdemucs.onnx file.

Mirrors demucs/apply.py:257-301 (segment/stride/triangle-window blending).
"""
from __future__ import annotations
import numpy as np
import onnxruntime as ort

SEGMENT = 7.8          # seconds (htdemucs default; core.segment)
SAMPLERATE = 44100
SEGMENT_LENGTH = int(SEGMENT * SAMPLERATE)  # 343980
N_SOURCES = 4


def _triangle_window(segment_length: int, transition_power: float = 1.0) -> np.ndarray:
    half = segment_length // 2
    w = np.concatenate([np.arange(1, half + 1, dtype=np.float32),
                        np.arange(segment_length - half, 0, -1, dtype=np.float32)])
    assert len(w) == segment_length
    return (w / w.max()) ** transition_power


def separate_segment(session: ort.InferenceSession, chunk: np.ndarray) -> np.ndarray:
    """Run the ONNX model on one segment. chunk: (B,2,SEGMENT_LENGTH) float32."""
    out = session.run(['output'], {'input': chunk})[0]
    return out  # (B, N_SOURCES, 2, SEGMENT_LENGTH)


def separate(session: ort.InferenceSession, mix: np.ndarray,
             overlap: float = 0.25, transition_power: float = 1.0) -> np.ndarray:
    """Separate a full track into stems. mix: (B, 2, length) float32.

    Returns (B, N_SOURCES, 2, length). Mirrors apply_model split=True, shifts=0.
    """
    if mix.ndim == 2:
        mix = mix[np.newaxis]
    batch, channels, length = mix.shape
    assert channels == 2, f"expected stereo, got {channels}ch"

    stride = int((1 - overlap) * SEGMENT_LENGTH)
    weight = _triangle_window(SEGMENT_LENGTH, transition_power)

    out = np.zeros((batch, N_SOURCES, 2, length), dtype=np.float32)
    sum_weight = np.zeros(length, dtype=np.float32)

    offsets = list(range(0, length, stride))
    for offset in offsets:
        # Extract chunk, zero-pad to SEGMENT_LENGTH on the right.
        chunk = np.zeros((batch, 2, SEGMENT_LENGTH), dtype=np.float32)
        chunk_len = min(SEGMENT_LENGTH, length - offset)
        chunk[:, :, :chunk_len] = mix[:, :, offset:offset + chunk_len]

        chunk_out = separate_segment(session, chunk)  # (B,S,2,SEGMENT_LENGTH)
        chunk_out = chunk_out[..., :chunk_len]        # trim to valid part

        w = weight[:chunk_len].astype(np.float32)
        out[:, :, :, offset:offset + chunk_len] += w * chunk_out
        sum_weight[offset:offset + chunk_len] += w

    out /= sum_weight  # broadcasts over (B,S,2,length)
    return out


if __name__ == '__main__':
    import sys, time, warnings
    warnings.filterwarnings('ignore')

    onnx_path = sys.argv[1] if len(sys.argv) > 1 else 'onnx_out/htdemucs.onnx'
    audio_path = sys.argv[2] if len(sys.argv) > 2 else 'demucs-gsoc/test.mp3'

    import soundfile as sf
    print(f"Loading {onnx_path} ...")
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"  providers active: {sess.get_providers()}")

    # demucs reads at 44.1kHz stereo. soundfile can't decode mp3 without ffmpeg/libsnd,
    # so decode via the installed demucs audio loader (spike only — runtime will use ffmpeg).
    sys.path.insert(0, 'demucs-gsoc')
    from demucs.audio import AudioFile
    print(f"Decoding {audio_path} ...")
    wav = AudioFile(audio_path).read(samplerate=SAMPLERATE, channels=2)
    wav = wav.numpy() if hasattr(wav, 'numpy') else np.asarray(wav)
    wav = wav.astype(np.float32)
    if wav.ndim == 2:
        wav = wav[np.newaxis]
    length = wav.shape[-1]
    print(f"  decoded: shape={wav.shape} ({length/SAMPLERATE:.1f}s)")

    print("Running ONNX separation (shifts=0, split, overlap=0.25) ...")
    t0 = time.time()
    stems_onnx = separate(sess, wav)
    t1 = time.time()
    print(f"  done in {t1-t0:.1f}s -> {stems_onnx.shape}")
