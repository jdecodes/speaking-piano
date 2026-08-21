from inference.predictor import SpeechPredictor


class AudioClassifier:
    """One trained model. No microphone, threads, or speech detection."""

    def __init__(self, name, model_path, metadata_path, confidence_threshold=0.70):
        self.name = name
        self.predictor = SpeechPredictor(model_path, metadata_path)
        self.confidence_threshold = confidence_threshold

    @property
    def sample_rate(self):
        return self.predictor.sample_rate

    def predict(self, audio):
        label, confidence = self.predictor.predict_audio(audio)
        accepted = confidence >= self.confidence_threshold
        return {
            "category": self.name,
            "label": label,
            "confidence": confidence,
            "accepted": accepted,
        }
