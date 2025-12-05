#Makes improting models easy
from .base_model import BaseModel
from .neural_networks import ResNetModel

#Factory pattern for creating models
def create_model(model_type, config):
    if model_type == 'resnet':
        return ResNetModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Only 'resnet' is supported.")