#Export all models for easy importing

from .base_model import BaseModel
from .neural_networks import SimpleCNN, ResNetModel, TransformerModel
from .interpretable_models import LogisticRegressionModel, RandomForestModel
from .ensemble import EnsembleModel
from .explainability import ModelExplainer

#Factory pattern for creating models
def create_model(model_type, config):
    if model_type == 'cnn':
        return SimpleCNN(config)
    elif model_type == 'resnet':
        return ResNetModel(config)
    elif model_type == 'transformer':
        return TransformerModel(config)
    elif model_type == 'logistic':
        return LogisticRegressionModel(config)
    elif model_type == 'random_forest':
        return RandomForestModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")