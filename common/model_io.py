import json
from pathlib import Path
import torch
from common.model import SpeechCNN


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_model(model, path):
    ensure_parent(path)
    torch.save(model.state_dict(), path)


def save_metadata(path, metadata):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def load_model(model_path, metadata_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    model = SpeechCNN(len(metadata["classes"])).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, metadata, device
