import sys
from inference.predictor import SpeechPredictor

if len(sys.argv) != 4:
    print("Usage: python test_model_file.py MODEL METADATA AUDIO.wav")
    raise SystemExit(1)
label, confidence = SpeechPredictor(sys.argv[1], sys.argv[2]).predict_file(sys.argv[3])
print("Prediction:", label)
print(f"Confidence: {confidence:.3f}")
