# sequential_note_session.py

import time

from inference.speech_detector import SpeechDetector
from recognition.audio_classifier import AudioClassifier
from audio.recorder import AudioRecorder
from pathlib import Path
import soundfile as sf
from piano_player.paino_player import PianoPlayer


class DebugAudioDumper:
    def __init__(self, debug_dir="debug", sample_rate=16000):
        self.debug_dir = Path(debug_dir)
        self.sample_rate = sample_rate
        self.counter = 0

        self.create_debug_folder()
        self.counter = self.get_last_counter()

    def create_debug_folder(self):
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def get_last_counter(self):
        max_counter = 0

        for file_path in self.debug_dir.glob("*.wav"):
            try:
                counter = int(file_path.stem.split("_")[0])
                max_counter = max(max_counter, counter)
            except (ValueError, IndexError):
                continue

        return max_counter

    def debug(self, audio, result):
        """
        Save audio with prediction/debug information.

        Expected result example:

        {
            "label": "A",
            "confidence": 0.98,
            "accepted": True
        }
        """

        self.counter += 1

        label = result.get("label", "unknown")
        confidence = result.get("confidence", 0.0)
        accepted = result.get("accepted", False)

        status = "accepted" if accepted else "rejected"

        filename = f"{self.counter:06d}_{label}_{confidence:.2f}_{status}.wav"

        output_path = self.debug_dir / filename

        sf.write(
            output_path,
            audio,
            self.sample_rate,
        )

        return output_path


class SequentialNoteSession:
    def __init__(
        self,
        recorder,
        speech_detector,
        letter_classifier,
        digit_classifier,
        accidental_classifier,
        play_callback,
        accidental_timeout=1.0,
    ):

        self.recorder = recorder
        self.speech_detector = speech_detector

        self.letter_classifier = letter_classifier
        self.digit_classifier = digit_classifier
        self.accidental_classifier = accidental_classifier

        self.play_callback = play_callback
        self.accidental_timeout = accidental_timeout

        self.running = False
        self.debug_audio = DebugAudioDumper(
            debug_dir="debug",
            sample_rate=16000,
        )

    def start(self):

        if self.running:
            return

        self.running = True

        # Microphone starts once and remains active.
        self.recorder.start()

        print()
        print("Sequential note recognizer started.")
        print("Microphone is active.")
        print("Press Ctrl+C to stop.")

        try:
            while self.running:
                self.run_one_note()

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            self.stop()

    # =========================================================

    def stop(self):

        if not self.running:
            return

        self.running = False

        self.recorder.stop()

        print("\nSequential note recognizer stopped.")

    def run_one_note(self):
        print("\nSpeak a letter: A to G")

        letter = self.get_required_prediction(
            classifier=self.letter_classifier,
            description="letter",
        )

        if letter is None:
            return

        print(f"Detected letter: {letter}")

        print("\nSpeak the octave: 0 to 8")

        digit = self.get_required_prediction(
            classifier=self.digit_classifier,
            description="octave",
        )

        if digit is None:
            return

        print(f"Detected octave: {digit}")

        # print(
        #     f"\nSpeak sharp or flat, "
        #     f"or remain silent for "
        #     f"{self.accidental_timeout} second(s)"
        # )
        accidental = None
        # accidental = self.get_optional_prediction(
        #     classifier=self.accidental_classifier,
        #     timeout=self.accidental_timeout,
        # )

        # if accidental is None:

        #     print("No accidental detected.")

        # else:

        #     print(f"Detected accidental: {accidental}")

        note = self.build_note(
            letter=letter,
            digit=digit,
            accidental=None,
        )

        if note is None:
            print(f"Could not create note from: {letter}, {digit}, {accidental}")

            return

        print()
        print(f"FINAL NOTE: {note}")

        self.play_callback(note)

    def get_required_prediction(
        self,
        classifier,
        description,
    ):

        self.speech_detector.reset()

        while self.running:
            chunk = self.get_next_chunk()

            if chunk is None:
                continue

            audio = self.speech_detector.process_chunk(chunk)

            if audio is None:
                continue

            result = classifier.predict(audio)

            label = result["label"]
            confidence = result["confidence"]
            accepted = result["accepted"]
            self.debug_audio.debug(audio, result)
            print(f"{description}: {label} ({confidence:.3f}) {'OK' if accepted else 'ignored'}")

            if accepted:
                return label

            print(f"Could not confidently recognize {description}. Try again.")

            self.speech_detector.reset()

        return None

    def get_optional_prediction(
        self,
        classifier,
        timeout,
    ):

        self.speech_detector.reset()

        deadline = time.monotonic() + timeout

        while self.running:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                return None

            chunk = self.get_next_chunk(timeout=remaining)

            if chunk is None:
                continue

            audio = self.speech_detector.process_chunk(chunk)

            if audio is None:
                continue

            result = classifier.predict(audio)

            label = result["label"]
            confidence = result["confidence"]
            accepted = result["accepted"]
            self.debug_audio.debug(audio, result)
            print(f"accidental: {label} ({confidence:.3f}) {'OK' if accepted else 'ignored'}")

            if accepted:
                return label

            print("Accidental was not confidently recognized.")

            # Continue looking for another utterance
            # until timeout expires.
            self.speech_detector.reset()

        return None

    # =========================================================
    # GET AUDIO CHUNK
    # =========================================================

    def get_next_chunk(self, timeout=None):

        return self.recorder.read(timeout=timeout)

    def build_note(
        self,
        letter,
        digit,
        accidental,
    ):

        letter = str(letter).upper()
        digit = str(digit)

        if accidental is None:
            return f"{letter}{digit}"

        accidental = str(accidental).lower()

        if accidental == "sharp":
            return f"{letter}#{digit}"

        if accidental == "flat":
            return f"{letter}b{digit}"

        print(f"Unknown accidental: {accidental}")

        return None


if __name__ == "__main__":
    piano_player = PianoPlayer()
    piano_player.start()

    def play_piano(note):
        print(f"Sending to piano: {note}")
        piano_player.play_note(note)

    recorder = AudioRecorder(
        sample_rate=16000,
        block_duration=0.05,
        input_device=None,
    )

    speech_detector = SpeechDetector(
        energy_threshold=0.03,
        start_chunks=3,
        end_silence_chunks=6,
        min_speech_chunks=3,
        max_chunks=80,
        pre_roll_chunks=8,
    )

    letter_classifier = AudioClassifier(
        "letters",
        model_path="models/letters_cnn.pt",
        metadata_path="metadata/letters_metadata.json",
        confidence_threshold=0.70,
    )

    digit_classifier = AudioClassifier(
        "digits",
        model_path="models/digits_cnn.pt",
        metadata_path="metadata/digits_metadata.json",
        confidence_threshold=0.70,
    )

    accidental_classifier = AudioClassifier(
        "accidentls",
        model_path="models/accidentals_cnn.pt",
        metadata_path="metadata/accidentals_metadata.json",
        confidence_threshold=0.70,
    )

    session = SequentialNoteSession(
        recorder=recorder,
        speech_detector=speech_detector,
        letter_classifier=letter_classifier,
        digit_classifier=digit_classifier,
        accidental_classifier=accidental_classifier,
        play_callback=play_piano,
        accidental_timeout=1.0,
    )

    session.start()
