import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

try:
    from cog import BasePredictor, Input, Path
except ImportError:
    class BasePredictor:
        pass

    def Input(*args, **kwargs):
        return kwargs.get("default", None)

    from pathlib import Path

CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Predictor(BasePredictor):
    def setup(self):
        """Loads both baseline and optimized models for CIFAR-10 inference."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SETUP] Initializing CIFAR-10 Predictor on device: {self.device}", flush=True)

        self.model_optimized = Net().to(self.device)
        try:
            self.model_optimized.load_state_dict(
                torch.load("optimized_model.pth", map_location=self.device, weights_only=True)
            )
            print("[SETUP] Successfully loaded 'optimized_model.pth'!", flush=True)
        except Exception as e:
            print(f"[SETUP] Warning: Could not load optimized_model.pth ({e})", flush=True)
        self.model_optimized.eval()

        self.model_baseline = Net().to(self.device)
        try:
            self.model_baseline.load_state_dict(
                torch.load("baseline_model.pth", map_location=self.device, weights_only=True)
            )
            print("[SETUP] Successfully loaded 'baseline_model.pth'!", flush=True)
        except Exception as e:
            print(f"[SETUP] Warning: Could not load baseline_model.pth ({e})", flush=True)
        self.model_baseline.eval()

        print("[SETUP] CIFAR-10 Classifier Predictor ready!", flush=True)

    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """
        Preprocesses an image for CIFAR-10:
        1. RGB conversion
        2. Resize to 32x32
        3. Normalize to [-1, 1] using (0.5, 0.5, 0.5)
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize((32, 32), Image.Resampling.BILINEAR)

        np_img = np.array(img, dtype=np.float32) / 255.0
        # Normalize (img - 0.5) / 0.5
        np_img = (np_img - 0.5) / 0.5
        # [H, W, C] -> [C, H, W]
        np_img = np.transpose(np_img, (2, 0, 1))

        tensor = torch.tensor(np_img, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    def predict(
        self,
        image: Path = Input(description="CIFAR-10 image to classify"),
        model_version: str = Input(
            description="Model to use: 'optimized', 'baseline', or 'both'",
            default="optimized"
        ),
    ) -> dict:
        """Runs CIFAR-10 classification inference."""
        print("\n==========================================", flush=True)
        print("[INFERENCE] CIFAR-10 classification request received!", flush=True)
        print(f"[INFERENCE] Image path: {image}", flush=True)

        tensor = self.preprocess_image(image)

        with torch.no_grad():
            opt_logits = self.model_optimized(tensor)
            opt_probs = torch.softmax(opt_logits, dim=1).squeeze(0)

            base_logits = self.model_baseline(tensor)
            base_probs = torch.softmax(base_logits, dim=1).squeeze(0)

        opt_dict = {CLASSES[i]: float(opt_probs[i].item()) for i in range(len(CLASSES))}
        base_dict = {CLASSES[i]: float(base_probs[i].item()) for i in range(len(CLASSES))}

        opt_top_idx = opt_probs.argmax().item()
        base_top_idx = base_probs.argmax().item()

        print(f"[INFERENCE] Optimized: {CLASSES[opt_top_idx]} ({opt_probs[opt_top_idx]*100:.2f}%)", flush=True)
        print(f"[INFERENCE] Baseline:  {CLASSES[base_top_idx]} ({base_probs[base_top_idx]*100:.2f}%)", flush=True)
        print("==========================================\n", flush=True)

        return {
            "optimized": {
                "top_class": CLASSES[opt_top_idx],
                "confidence": round(float(opt_probs[opt_top_idx].item()), 4),
                "probabilities": opt_dict,
            },
            "baseline": {
                "top_class": CLASSES[base_top_idx],
                "confidence": round(float(base_probs[base_top_idx].item()), 4),
                "probabilities": base_dict,
            },
        }
