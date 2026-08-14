from src.models.cnn import CNN
from src.models.bigru import BiGRU
from src.models.bilstm import BiLSTM
from src.models.effnet import EfficientNet
from src.models.vit import ViT
from src.models.resnet import ResNet
from src.models.mobilenet import MobileNet
from src.models.ast_model import AST
from src.models.swin import Swin
from src.models.conformer import Conformer

MODEL_REGISTRY = {
    "cnn": CNN,
    "crnn": BiGRU,
    "bilstm": BiLSTM,
    "efficientnet": EfficientNet,
    "resnet": ResNet,
    "mobilenet": MobileNet,
    "vit": ViT,
    "ast": AST,
    "swin": Swin,
    "conformer": Conformer,
}

def build_model(name: str, n_classes: int = 2, in_channels: int = 1):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo desconhecido: {name}. Opções: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_classes=n_classes, in_channels=in_channels)
