    """
    Train a speech classifier.

    Expected folder structure:

        dataset_path/
            train/
                class_1/
                class_2/
                ...
            test/
                class_1/
                class_2/
                ...

    Example:

        train_classifier(
            dataset_path="data/letters",
            classes=["A", "B", "C", "D", "E", "F", "G"],
            model_path="models/letters_cnn.pt",
            metadata_path="metadata/letters_metadata.json"
        )
    """
