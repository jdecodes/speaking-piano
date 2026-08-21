import json
import os
import random

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


def ensure_parent_directory(file_path):
    """
    Create the parent directory for a file path if it doesn't exist.
    Example: models/digits/digits_cnn.pt
    Creates: models/digits/
    """

    parent_directory = os.path.dirname(file_path)

    if parent_directory:
        os.makedirs(parent_directory, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000

# Every recording presented to the CNN will be exactly
# this many seconds long.
TARGET_DURATION = 1.0

TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)

N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 150
BATCH_SIZE = 8
LEARNING_RATE = 0.001


# ============================================================
# Reproducibility
# ============================================================


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Load audio
# ============================================================


def load_audio(path, sample_rate=SAMPLE_RATE):

    audio, sr = librosa.load(path, sr=sample_rate, mono=True)

    return audio.astype(np.float32)


# ============================================================
# Normalize audio
# ============================================================


def normalize_audio(audio):

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    return audio


# ============================================================
# Force audio to fixed duration
# ============================================================


def fix_audio_length(audio, target_samples=TARGET_SAMPLES):
    """
    Make every recording exactly the same length.

    Short recording:
        zero padded

    Long recording:
        truncated

    This makes the CNN input shape deterministic.
    """

    current_samples = len(audio)

    if current_samples < target_samples:
        padding = target_samples - current_samples

        audio = np.pad(audio, (0, padding), mode="constant")

    elif current_samples > target_samples:
        audio = audio[:target_samples]

    return audio


# ============================================================
# Feature extraction
# ============================================================


def extract_features(audio, sample_rate=SAMPLE_RATE):

    audio = normalize_audio(audio)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=50,
        fmax=sample_rate // 2,
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize the spectrogram
    # independently for each recording.

    mean = log_mel.mean()

    std = log_mel.std() + 1e-8

    log_mel = (log_mel - mean) / std

    return log_mel.astype(np.float32)


# ============================================================
# Load dataset
# ============================================================


def load_dataset(dataset_path, classes, class_to_index):

    features = []

    labels = []

    paths = []

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            raise FileNotFoundError(f"Missing class folder: {class_path}")

        for filename in sorted(os.listdir(class_path)):
            if not filename.lower().endswith(".wav"):
                continue

            path = os.path.join(class_path, filename)

            try:
                audio = load_audio(path)

                audio = fix_audio_length(audio)

                feature = extract_features(audio)

                features.append(feature)

                labels.append(class_to_index[class_name])

                paths.append(path)

                print(f"Loaded: {path}")

            except Exception as e:
                print(f"Failed: {path}")

                print(f"  {e}")

    if not features:
        raise RuntimeError(f"No WAV files found in {dataset_path}")

    return (features, np.asarray(labels, dtype=np.int64), paths)


# ============================================================
# Stack features
# ============================================================


def stack_features(features):

    shapes = {feature.shape for feature in features}

    if len(shapes) != 1:
        raise RuntimeError(f"Feature shapes are not identical: {shapes}")

    return np.stack(features)


# ============================================================
# Data augmentation
# ============================================================


def augment_spectrogram(feature):

    feature = feature.copy()

    # Time shift

    if random.random() < 0.5:
        shift = random.randint(-5, 5)

        feature = np.roll(feature, shift, axis=1)

    # Frequency shift

    if random.random() < 0.3:
        shift = random.randint(-2, 2)

        feature = np.roll(feature, shift, axis=0)

    # Small noise

    if random.random() < 0.5:
        noise = np.random.normal(0, 0.02, feature.shape).astype(np.float32)

        feature += noise

    return feature


# ============================================================
# PyTorch Dataset
# ============================================================


class SpeechDataset(Dataset):
    def __init__(self, features, labels, augment=False):

        self.features = features

        self.labels = labels

        self.augment = augment

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        feature = self.features[index].copy()

        label = self.labels[index]

        if self.augment:
            feature = augment_spectrogram(feature)

        # CNN input: [channels, mel, time]
        # Currently: [1, 64, ~200]

        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return (feature, label)


# ============================================================
# CNN
# ============================================================


class SpeechCNN(nn.Module):
    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(
            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # ------------------------------------------------
            # Global average pooling
            # ------------------------------------------------
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, num_classes))

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# Train one epoch
# ============================================================


def train_one_epoch(model, loader, optimizer, criterion):

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        predictions = output.argmax(dim=1)
        correct += (predictions == y).sum().item()
        total += X.size(0)

    average_loss = total_loss / total
    accuracy = correct / total
    return (average_loss, accuracy)


# ============================================================
# Train model
# ============================================================


def train_model(X, y, num_classes, model_path):

    dataset = SpeechDataset(X, y, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = SpeechCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    for epoch in range(EPOCHS):
        loss, accuracy = train_one_epoch(model, loader, optimizer, criterion)
        print(f"Epoch {epoch + 1:03d}/{EPOCHS} | Loss: {loss:.4f} | Train Acc: {accuracy:.3f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    return model


# ============================================================
# Evaluate test set
# ============================================================


def evaluate_test_set(model, X_test, y_test, test_paths, classes, index_to_class):

    model.eval()
    dataset = SpeechDataset(X_test, y_test, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []
    probabilities = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            output = model(X)
            probs = torch.softmax(output, dim=1)
            prediction = probs.argmax(dim=1).item()
            confidence = probs[0, prediction].item()
            predictions.append(prediction)
            probabilities.append(confidence)

    predictions = np.asarray(predictions)
    accuracy = np.mean(predictions == y_test)

    print("FINAL TEST RESULTS")
    print(f"Accuracy: {accuracy:.3f}")
    print()

    # --------------------------------------------------------
    # Individual predictions
    # --------------------------------------------------------

    for path, actual, predicted, confidence in zip(test_paths, y_test, predictions, probabilities):
        actual_name = index_to_class[actual]
        predicted_name = index_to_class[predicted]
        status = "OK" if actual == predicted else "WRONG"
        print(
            f"{status:5s} | Actual: {actual_name} | Predicted: {predicted_name} | Confidence: {confidence:.3f} | {path}"
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("Classification report:")
    print()

    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("Confusion matrix:")

    print()

    matrix = confusion_matrix(y_test, predictions, labels=list(range(len(classes))))

    print(matrix)

    return accuracy


# ============================================================
# Save metadata
# ============================================================


def save_metadata(metadata_path, classes):

    metadata = {
        "sample_rate": SAMPLE_RATE,
        "target_duration": TARGET_DURATION,
        "target_samples": TARGET_SAMPLES,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "classes": classes,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved: {metadata_path}")


# ============================================================
# Generic classifier trainer
# ============================================================


def train_classifier(dataset_path, classes, model_path, metadata_path):
    # Ensure output directories exist
    ensure_parent_directory(model_path)
    ensure_parent_directory(metadata_path)
    set_seed()
    print(f"starting classifier")
    class_to_index = {name: i for i, name in enumerate(classes)}

    index_to_class = {i: name for name, i in class_to_index.items()}

    train_path = os.path.join(dataset_path, "train")

    test_path = os.path.join(dataset_path, "test")

    print(f"Using device: {DEVICE}")

    print(f"Classes: {classes}")

    print(f"Target duration: {TARGET_DURATION} seconds")

    print(f"Target samples: {TARGET_SAMPLES}")

    print()

    # ========================================================
    # Training data
    # ========================================================

    print("LOADING TRAINING DATA")

    (train_features, train_labels, train_paths) = load_dataset(train_path, classes, class_to_index)

    print()

    print(f"Loaded {len(train_features)} training recordings.")

    X_train = stack_features(train_features)

    print("Training feature shape:", X_train.shape)

    # ========================================================
    # Test data
    # ========================================================

    print()

    print("=" * 60)

    print("LOADING TEST DATA")

    print("=" * 60)

    (test_features, test_labels, test_paths) = load_dataset(test_path, classes, class_to_index)

    print()

    print(f"Loaded {len(test_features)} test recordings.")

    X_test = stack_features(test_features)

    print("Test feature shape:", X_test.shape)

    # ========================================================
    # Sanity check
    # ========================================================

    if X_train.shape[1:] != X_test.shape[1:]:
        raise RuntimeError("Training and test feature shapes differ.")

    # ========================================================
    # Training
    # ========================================================

    print()

    print("=" * 60)

    print("TRAINING CNN")

    print("=" * 60)

    model = train_model(X_train, train_labels, num_classes=len(classes), model_path=model_path)

    # ========================================================
    # Metadata
    # ========================================================

    save_metadata(metadata_path, classes)

    # ========================================================
    # Final test
    # ========================================================

    print()

    print("=" * 60)

    print("EVALUATING UNSEEN TEST DATA")

    print("=" * 60)

    evaluate_test_set(model, X_test, test_labels, test_paths, classes, index_to_class)

    print()

    print("Done.")
