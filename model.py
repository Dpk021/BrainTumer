"""
Model definition shared by train.py and main.py.

Mirrors the original Keras architecture:
    VGG16 (ImageNet, no top, last 3 conv layers trainable)
    -> Flatten -> Dropout(0.3) -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(num_classes)
"""
import torch
import torch.nn as nn
from torchvision import models

IMAGE_SIZE = 128


def get_device() -> torch.device:
    """Pick the best available device: Apple MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TumorClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16(weights=weights)

        # Convolutional backbone only (equivalent to include_top=False)
        self.features = vgg.features

        # Freeze everything...
        for p in self.features.parameters():
            p.requires_grad = False
        # ...then unfreeze block5's three conv layers (Keras layers[-2], [-3], [-4]).
        # In torchvision's vgg16.features these are indices 24, 26 and 28.
        for idx in (24, 26, 28):
            for p in self.features[idx].parameters():
                p.requires_grad = True

        # 128x128 input -> 512 x 4 x 4 feature map after 5 max-pools
        flat_dim = 512 * (IMAGE_SIZE // 32) ** 2

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),  # logits; softmax applied in loss / at inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def save_checkpoint(model: nn.Module, class_names: list[str], path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "class_names": class_names}, path)


def load_checkpoint(path: str, device: torch.device | None = None):
    """Returns (model in eval mode, class_names)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    class_names = ckpt["class_names"]
    model = TumorClassifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, class_names
