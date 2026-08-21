import os, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from common.audio import load_audio, fix_audio_length
from common.features import extract_features, stack_features
from common.model import SpeechCNN
from common.model_io import save_model, save_metadata


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_path, classes, sample_rate, target_samples, n_mels, n_fft, hop_length):
    features = []
    labels = []
    paths = []
    for label, name in enumerate(classes):
        folder = os.path.join(dataset_path, name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith(".wav"):
                path = os.path.join(folder, filename)
                audio = fix_audio_length(load_audio(path, sample_rate), target_samples)
                features.append(extract_features(audio, sample_rate, n_mels, n_fft, hop_length))
                labels.append(label)
                paths.append(path)
                print("Loaded:", path)
    return stack_features(features), np.asarray(labels, dtype=np.int64), paths


def augment(x):
    x = x.copy()
    if random.random() < 0.5:
        x = np.roll(x, random.randint(-5, 5), axis=1)
    if random.random() < 0.3:
        x = np.roll(x, random.randint(-2, 2), axis=0)
    if random.random() < 0.5:
        x += np.random.normal(0, 0.02, x.shape).astype(np.float32)
    return x


class SpeechDataset(Dataset):
    def __init__(self, X, y, augment_data=False):
        self.X = X
        self.y = y
        self.augment_data = augment_data

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = augment(self.X[i]) if self.augment_data else self.X[i]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(self.y[i], dtype=torch.long)


def train_classifier(
    dataset_path,
    classes,
    model_path,
    metadata_path,
    sample_rate=16000,
    target_duration=1.0,
    n_mels=64,
    n_fft=512,
    hop_length=160,
    epochs=150,
    batch_size=8,
    learning_rate=0.001,
):
    set_seed()
    target_samples = int(sample_rate * target_duration)
    X, y, _ = load_dataset(dataset_path, classes, sample_rate, target_samples, n_mels, n_fft, hop_length)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpeechCNN(len(classes)).to(device)
    loader = DataLoader(SpeechDataset(X, y, True), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total = correct = 0
        loss_sum = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total += len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            loss_sum += loss.item() * len(yb)
        print(f"Epoch {epoch + 1:03d}/{epochs} | Loss: {loss_sum / total:.4f} | Train Acc: {correct / total:.3f}")
    metadata = {
        "sample_rate": sample_rate,
        "target_duration": target_duration,
        "target_samples": target_samples,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "classes": classes,
    }
    save_model(model, model_path)
    save_metadata(metadata_path, metadata)
    print("Model saved:", model_path)
    print("Metadata saved:", metadata_path)
    return model
