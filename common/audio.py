import numpy as np
import librosa


def load_audio(path, sample_rate=16000):
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def normalize_audio(audio):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    return audio / peak if peak > 0 else audio


def fix_audio_length(audio, target_samples):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    current_samples = len(audio)

    if current_samples < target_samples:
        padding = target_samples - current_samples

        pad_front = padding // 2
        pad_back = padding - pad_front

        return np.pad(audio, (pad_front, pad_back))

    if current_samples > target_samples:
        excess = current_samples - target_samples

        trim_front = excess // 2
        trim_back = excess - trim_front

        return audio[trim_front : current_samples - trim_back]

    return audio


def rms_energy(audio):
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(audio * audio) + 1e-12))
