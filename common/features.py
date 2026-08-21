import numpy as np
import librosa
from common.audio import normalize_audio


def extract_features(audio, sample_rate=16000, n_mels=64, n_fft=512, hop_length=160):
    audio = normalize_audio(audio)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=50, fmax=sample_rate // 2
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return ((log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)).astype(np.float32)


def stack_features(features):
    shapes = {x.shape for x in features}
    if len(shapes) != 1:
        raise RuntimeError(f"Feature shapes are not identical: {shapes}")
    return np.stack(features).astype(np.float32)
