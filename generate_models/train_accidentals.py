import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from classifier import train_classifier


def train_accidentals():

    train_classifier(
        dataset_path="data/accidentals",
        classes=["flat", "sharp"],
        model_path="models/accidentals_cnn.pt",
        metadata_path=("metadata/accidentals_metadata.json"),
    )


if __name__ == "__main__":
    train_accidentals()
