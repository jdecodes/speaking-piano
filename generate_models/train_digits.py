import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from classifier import train_classifier


def train_digits():

    train_classifier(
        dataset_path="data/digits",
        classes=["0", "1", "2", "3", "4", "5", "6", "7"],
        model_path="models/digits_cnn.pt",
        metadata_path=("metadata/digits_metadata.json"),
    )


if __name__ == "__main__":
    train_digits()
