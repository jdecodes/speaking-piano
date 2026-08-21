import queue, threading
import sounddevice as sd
from inference.predictor import SpeechPredictor
from inference.speech_detector import SpeechDetector


class LiveRecognizer:
    def __init__(
        self,
        model_path,
        metadata_path,
        callback=None,
        confidence_threshold=0.70,
        block_duration=0.05,
        detector_kwargs=None,
        input_device=None,
    ):
        self.predictor = SpeechPredictor(model_path, metadata_path)
        self.callback = callback
        self.confidence_threshold = confidence_threshold
        self.blocksize = int(self.predictor.sample_rate * block_duration)
        self.detector = SpeechDetector(**(detector_kwargs or {}))
        self.input_device = input_device
        self.audio_queue = queue.Queue(maxsize=100)
        self.utterance_queue = queue.Queue(maxsize=20)
        self.running = False
        self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        try:
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _detector_worker(self):
        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            utterance = self.detector.process_chunk(chunk)
            if utterance is not None:
                try:
                    self.utterance_queue.put_nowait(utterance)
                except queue.Full:
                    print("Utterance queue full, dropping audio.")

    def _predictor_worker(self):
        while self.running:
            try:
                utterance = self.utterance_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            label, confidence = self.predictor.predict_audio(utterance)
            if confidence >= self.confidence_threshold:
                if self.callback:
                    self.callback(label, confidence)
                else:
                    print(f"Recognized: {label} | confidence={confidence:.3f}")
            else:
                print(f"Ignored: {label} | confidence={confidence:.3f}")

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._detector_worker, daemon=True, name="speech-detector").start()
        threading.Thread(target=self._predictor_worker, daemon=True, name="speech-predictor").start()
        self.stream = sd.InputStream(
            samplerate=self.predictor.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._audio_callback,
            device=self.input_device,
        )
        self.stream.start()
        print("Live recognizer started.")

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("Live recognizer stopped.")
