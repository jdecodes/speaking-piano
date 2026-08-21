import threading

from audio.recorder import AudioRecorder
from inference.speech_detector import SpeechDetector
from recognition.audio_classifier import AudioClassifier


class SpeechRecognizer:
    """
    Orchestration layer:
        AudioRecorder -> SpeechDetector -> all classifiers -> consolidated result

    This class owns the live recording session. Individual classifiers remain
    independent of live audio and can also be used for file/offline inference.
    """

    def __init__(
        self,
        classifiers,
        callback=None,
        block_duration=0.05,
        detector_kwargs=None,
        input_device=None,
    ):
        if not classifiers:
            raise ValueError("At least one classifier is required")

        self.classifiers = classifiers
        self.callback = callback

        sample_rates = {classifier.sample_rate for classifier in classifiers.values()}
        if len(sample_rates) != 1:
            raise ValueError(f"All classifiers must use the same sample rate, got: {sample_rates}")

        self.sample_rate = sample_rates.pop()
        self.recorder = AudioRecorder(
            sample_rate=self.sample_rate,
            block_duration=block_duration,
            input_device=input_device,
        )
        self.detector = SpeechDetector(**(detector_kwargs or {}))
        self.running = False
        self.worker = None

    @classmethod
    def from_model_configs(cls, model_configs, **kwargs):
        classifiers = {}
        for name, config in model_configs.items():
            classifiers[name] = AudioClassifier(
                name=name,
                model_path=config["model_path"],
                metadata_path=config["metadata_path"],
                confidence_threshold=config.get("confidence_threshold", 0.70),
            )
        return cls(classifiers=classifiers, **kwargs)

    def _consolidate(self, audio):
        predictions = {}
        accepted = {}

        for name, classifier in self.classifiers.items():
            result = classifier.predict(audio)
            predictions[name] = result
            print(f"for shape {audio.shape} :: {name} {predictions[name]}  {result}")
            if result["accepted"]:
                accepted[name] = result

        return {
            "audio": audio,
            "predictions": predictions,
            "accepted": accepted,
        }

    def _worker(self):
        while self.running:
            chunk = self.recorder.read(timeout=0.1)
            if chunk is None:
                continue

            utterance = self.detector.process_chunk(chunk)
            if utterance is None:
                continue

            result = self._consolidate(utterance)

            if self.callback is not None:
                self.callback(result)

    def start(self):
        if self.running:
            return
        self.running = True
        self.recorder.start()
        self.worker = threading.Thread(
            target=self._worker,
            daemon=True,
            name="speech-recognition-session",
        )
        self.worker.start()
        print("Speech recognizer started.")

    def stop(self):
        self.running = False
        self.recorder.stop()
        if self.worker is not None:
            self.worker.join(timeout=1.0)
            self.worker = None
        print("Speech recognizer stopped.")
