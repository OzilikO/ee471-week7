import torch
from torch import nn
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

# FashionMNIST 10 Class Labels
CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Predictor(BasePredictor):
    def setup(self):
        """Loads both baseline and optimized models for inference."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SETUP] Initializing Predictor on device: {self.device}", flush=True)

        # 1. Load Optimized (HPO) Model
        self.model_optimized = NeuralNetwork().to(self.device)
        try:
            self.model_optimized.load_state_dict(
                torch.load("optimized_model.pth", map_location=self.device, weights_only=True)
            )
            print("[SETUP] Successfully loaded 'optimized_model.pth'!", flush=True)
        except Exception as e:
            print(f"[SETUP] Warning: Could not load optimized_model.pth ({e})", flush=True)
        self.model_optimized.eval()###

        # 2. Load Baseline Model
        self.model_baseline = NeuralNetwork().to(self.device)
        try:
            self.model_baseline.load_state_dict(
                torch.load("baseline_model.pth", map_location=self.device, weights_only=True)
            )
            print("[SETUP] Successfully loaded 'baseline_model.pth'!", flush=True)
        except Exception as e:
            print(f"[SETUP] Warning: Could not load baseline_model.pth ({e})", flush=True)
        self.model_baseline.eval()###

        print("[SETUP] Garment Classifier Cog Predictor is ready!", flush=True)

    def preprocess_image(self, image_path: Path) -> torch.Tensor:###
        """
        Preprocesses an uploaded image to match FashionMNIST specs:
        1. Convert to Grayscale ('L')
        2. Resize to 28x28
        3. Background brightness check (invert if background is bright/white)
        4. Normalize pixels to [0.0, 1.0]
        5. Convert to tensor shape [1, 1, 28, 28]
        """
        img = Image.open(image_path).convert("L")
        img = img.resize((28, 28), Image.Resampling.BILINEAR)

        np_img = np.array(img, dtype=np.float32)

        # Border check: if the average edge brightness is > 128, invert colors
        # so clothing is bright (white) on dark (black) background
        border = np.concatenate([np_img[0, :], np_img[-1, :], np_img[:, 0], np_img[:, -1]])
        if border.mean() > 128.0:
            np_img = 255.0 - np_img

        # Normalize to [0.0, 1.0]
        tensor = torch.tensor(np_img / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict(
        self,
        image: Path = Input(description="Garment image to classify"),
        model_version: str = Input(
            description="Model to use: 'optimized', 'baseline', or 'both'",
            default="optimized"
        ),
    ) -> dict:
        """Runs garment classification inference and logs predictions."""
        print("\n==========================================", flush=True)
        print("[INFERENCE] Garment classification request received!", flush=True)
        print(f"[INFERENCE] Image path: {image}", flush=True)
        print(f"[INFERENCE] Selected model version: {model_version}", flush=True)

        tensor = self.preprocess_image(image)

        with torch.no_grad():
            # Run optimized model
            opt_logits = self.model_optimized(tensor)
            opt_probs = torch.softmax(opt_logits, dim=1).squeeze(0) ###

            # Run baseline model
            base_logits = self.model_baseline(tensor)
            base_probs = torch.softmax(base_logits, dim=1).squeeze(0)###

        # Format probabilities
        opt_dict = {CLASSES[i]: float(opt_probs[i].item()) for i in range(len(CLASSES))}
        base_dict = {CLASSES[i]: float(base_probs[i].item()) for i in range(len(CLASSES))}

        opt_top_idx = opt_probs.argmax().item()
        base_top_idx = base_probs.argmax().item()

        print(f"[INFERENCE] Optimized Model Prediction: {CLASSES[opt_top_idx]} ({opt_probs[opt_top_idx]*100:.2f}%)", flush=True)
        print(f"[INFERENCE] Baseline Model Prediction:  {CLASSES[base_top_idx]} ({base_probs[base_top_idx]*100:.2f}%)", flush=True)
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
