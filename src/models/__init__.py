from src.models.cnn import CNN
from src.models.crnn import CRNN
from src.models.effnet import EfficientNetSmall
from src.models.vit import ViTSmall
from src.models.resnet import ResNetSmall
from src.models.mobilenet import MobileNetSmall

MODEL_REGISTRY = {
    "cnn": CNN,
    "crnn": CRNN,
    "efficientnet": EfficientNetSmall,
    "vit": ViTSmall,
    "resnet": ResNetSmall,
    "mobilenet": MobileNetSmall,
}


def build_model(name: str, n_classes: int = 2, in_channels: int = 1):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo desconhecido: {name}. Opções: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_classes=n_classes, in_channels=in_channels)
