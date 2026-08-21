from recognition.speech_recognizer import SpeechRecognizer


class NoteRecognizer:
    """Piano-specific layer on top of the generic SpeechRecognizer."""

    def __init__(self, speech_recognizer, note_callback=None, assembler=None):
        self.speech_recognizer = speech_recognizer
        self.note_callback = note_callback
        self.assembler = assembler
        self.speech_recognizer.callback = self._on_speech_result

    def _on_speech_result(self, result):
        # For now expose the consolidated token-level result.
        # A NoteAssembler can later consume accepted predictions and build C#4.
        if self.assembler is not None:
            note = self.assembler.add_result(result)
            if note is not None and self.note_callback is not None:
                self.note_callback(note)
        elif self.note_callback is not None:
            self.note_callback(result)

    def start(self):
        self.speech_recognizer.start()

    def stop(self):
        self.speech_recognizer.stop()
