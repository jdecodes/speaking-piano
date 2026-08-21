import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from classifier import train_classifier


def train_letters():
    pass
    train_classifier(
        dataset_path="data/letters",
        classes=["A", "B", "C", "D", "E", "F", "G"],
        model_path="models/letters_cnn.pt",
        metadata_path=("metadata/letters_metadata.json"),
    )


if __name__ == "__main__":
    train_letters()
