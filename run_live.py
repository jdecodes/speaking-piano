from inference.live_recognizer import LiveRecognizer


def on_prediction(label, confidence):
    print(f">>> {label} ({confidence:.3f})")


recognizer = LiveRecognizer(
    model_path="models/digits_cnn.pt",
    metadata_path="metadata/digits_metadata.json",
    callback=on_prediction,
    confidence_threshold=0.70,
    detector_kwargs={"energy_threshold": 0.01, "start_chunks": 2, "end_silence_chunks": 6, "pre_roll_chunks": 2},
)
recognizer.start()
try:
    input("Press Enter to stop...\n")
finally:
    recognizer.stop()
