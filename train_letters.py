from classifier import train_classifier


def train_letters():

    train_classifier(
        dataset_path="data/letters",
        classes=["A", "B", "C", "D", "E", "F", "G"],
        model_path="models/letters_cnn.pt",
        metadata_path=("metadata/letters_metadata.json"),
    )


if __name__ == "__main__":
    train_letters()
