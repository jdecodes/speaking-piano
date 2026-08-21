import queue
import sounddevice as sd


class AudioRecorder:
    """Microphone capture only. Produces float32 mono chunks."""

    def __init__(self, sample_rate=16000, block_duration=0.05, input_device=None, queue_size=100):
        self.sample_rate = sample_rate
        self.blocksize = int(sample_rate * block_duration)
        self.input_device = input_device
        self.queue = queue.Queue(maxsize=queue_size)
        self.stream = None
        self.running = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        try:
            self.queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def start(self):
        if self.running:
            return
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
            device=self.input_device,
        )
        self.stream.start()
        print("Audio recorder started.")

    def read(self, timeout=0.1):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("Audio recorder stopped.")
