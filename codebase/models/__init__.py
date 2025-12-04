#Export all models for easy importing

from .base_model import BaseModel
from .neural_networks import SimpleCNN, ResNetModel, TransformerModel

#Factory pattern for creating models
def create_model(model_type, config):
    if model_type == 'cnn':
        return SimpleCNN(config)
    elif model_type == 'resnet':
        return ResNetModel(config)
    elif model_type == 'transformer':
        return TransformerModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")