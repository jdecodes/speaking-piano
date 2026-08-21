from recognition.speech_recognizer import SpeechRecognizer


def on_result(result):
    print("\n--- speech segment ---")
    for category, prediction in result["predictions"].items():
        mark = "OK" if prediction["accepted"] else "ignored"
        print(f"{category:12} {prediction['label']:8} {prediction['confidence']:.3f}  {mark}")


recognizer = SpeechRecognizer.from_model_configs(
    {
        "letter": {
            "model_path": "models/letters_cnn.pt",
            "metadata_path": "metadata/letters_metadata.json",
            "confidence_threshold": 0.70,
        },
        "digit": {
            "model_path": "models/digits_cnn.pt",
            "metadata_path": "metadata/digits_metadata.json",
            "confidence_threshold": 0.70,
        },
        "accidental": {
            "model_path": "models/accidentals_cnn.pt",
            "metadata_path": "metadata/accidentals_metadata.json",
            "confidence_threshold": 0.70,
        },
    },
    callback=on_result,
    block_duration=0.05,
    detector_kwargs={
        "energy_threshold": 0.03,
        "start_chunks": 2,
        "end_silence_chunks": 3,
        "pre_roll_chunks": 4,
    },
)

recognizer.start()

try:
    input("Press Enter to stop...\n")
finally:
    recognizer.stop()
