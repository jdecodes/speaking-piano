import torch
from common.audio import load_audio, fix_audio_length
from common.features import extract_features
from common.model_io import load_model


class SpeechPredictor:
    def __init__(self, model_path, metadata_path, device=None):
        self.model, self.metadata, self.device = load_model(model_path, metadata_path, device)

    @property
    def sample_rate(self):
        return self.metadata["sample_rate"]

    def predict_audio(self, audio):
        audio = fix_audio_length(audio, self.metadata["target_samples"])
        feature = extract_features(
            audio,
            self.metadata["sample_rate"],
            self.metadata["n_mels"],
            self.metadata["n_fft"],
            self.metadata["hop_length"],
        )
        x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        i = int(probs.argmax().item())
        return self.metadata["classes"][i], float(probs[i].item())

    def predict_file(self, path):
        return self.predict_audio(load_audio(path, self.sample_rate))
