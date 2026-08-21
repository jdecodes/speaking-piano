import queue
import threading

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 44100
BLOCK_SIZE = 512


NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def parse_note(note):
    note = note.strip().upper()

    if len(note) < 2:
        raise ValueError(f"Invalid note: {note}")

    if note[1] in ("#", "B"):
        note_name = note[:2]
        octave_text = note[2:]
    else:
        note_name = note[0]
        octave_text = note[1:]

    if note_name not in NOTE_TO_SEMITONE:
        raise ValueError(f"Invalid note name: {note_name}")

    octave = int(octave_text)

    return note_name, octave


def note_to_midi(note):
    note_name, octave = parse_note(note)

    midi = (octave + 1) * 12 + NOTE_TO_SEMITONE[note_name]

    if midi < 21 or midi > 108:
        raise ValueError(f"{note} is outside the piano range A0-C8")

    return midi


def midi_to_frequency(midi, reference_frequency=440.0):
    return reference_frequency * 2 ** ((midi - 69) / 12)


def note_to_frequency(note):
    midi = note_to_midi(note)

    return midi_to_frequency(midi)


def generate_piano_note(
    note,
    sample_rate=SAMPLE_RATE,
    duration=2.5,
):
    frequency = note_to_frequency(note)

    t = np.arange(int(sample_rate * duration)) / sample_rate

    harmonics = [
        (1, 1.00),
        (2, 0.50),
        (3, 0.30),
        (4, 0.18),
        (5, 0.10),
        (6, 0.06),
        (7, 0.04),
        (8, 0.025),
    ]

    signal = np.zeros_like(t)

    for harmonic, amplitude in harmonics:
        harmonic_frequency = frequency * harmonic

        signal += amplitude * np.sin(2 * np.pi * harmonic_frequency * t)

    # Attack
    attack_time = 0.01

    attack = np.minimum(
        t / attack_time,
        1.0,
    )

    # Decay
    decay_time = 1.2

    decay = np.exp(-t / decay_time)

    signal *= attack * decay

    peak = np.max(np.abs(signal))

    if peak > 0:
        signal /= peak

    return signal.astype(np.float32)


class NoteCache:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def get(self, note):
        with self.lock:
            if note not in self.cache:
                print(f"Generating piano note: {note}")

                self.cache[note] = generate_piano_note(note)

            return self.cache[note]


class PianoPlayer:
    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        block_size=BLOCK_SIZE,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size

        self.note_queue = queue.Queue()

        self.note_cache = NoteCache()

        self.stop_event = threading.Event()

        self.thread = None

    # =========================================================
    # PUBLIC API
    # =========================================================

    def play_note(self, note):
        """
        This is the callback API.

        SequentialNoteSession will call:

            piano_player.play_note("C4")
        """

        note = str(note).strip()

        # Validate before putting it in the queue.
        note_to_midi(note)

        print(f"Queued piano note: {note}")

        self.note_queue.put(note)

    # =========================================================

    def start(self):
        if self.thread is not None:
            return

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._audio_worker,
            daemon=True,
        )

        self.thread.start()

        print("Piano player started.")

    # =========================================================

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join()

            self.thread = None

        print("Piano player stopped.")

    # =========================================================
    # AUDIO WORKER
    # =========================================================

    def _audio_worker(self):
        print("Piano audio thread started.")

        active_notes = []

        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
        ) as stream:
            while not self.stop_event.is_set():
                # ---------------------------------------------
                # Get all newly received notes.
                # ---------------------------------------------

                while True:
                    try:
                        note = self.note_queue.get_nowait()

                    except queue.Empty:
                        break

                    audio = self.note_cache.get(note)

                    active_notes.append(
                        {
                            "note": note,
                            "audio": audio,
                            "position": 0,
                        }
                    )

                    print(f"Playing: {note}")

                # ---------------------------------------------
                # Create output block.
                # ---------------------------------------------

                output = np.zeros(
                    self.block_size,
                    dtype=np.float32,
                )

                still_active = []

                # ---------------------------------------------
                # Mix all active notes.
                # ---------------------------------------------

                for voice in active_notes:
                    audio = voice["audio"]
                    position = voice["position"]

                    remaining = len(audio) - position

                    count = min(
                        self.block_size,
                        remaining,
                    )

                    output[:count] += audio[position : position + count]

                    voice["position"] += count

                    if voice["position"] < len(audio):
                        still_active.append(voice)

                active_notes = still_active

                # ---------------------------------------------
                # Prevent clipping.
                # ---------------------------------------------

                peak = np.max(np.abs(output))

                if peak > 1.0:
                    output /= peak

                # ---------------------------------------------
                # Play.
                # ---------------------------------------------

                stream.write(output.reshape(-1, 1))
