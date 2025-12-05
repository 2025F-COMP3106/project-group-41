#Abstract class that defines a common interface for all models
#Purpose: Template that simple CNN, ResNetModel, transform inherit from making sure they all have same basic methods.
import torch
from torch.nn import Module

class BaseModel(Module):
    """
    Abstract base class, defines interface for all models.
    Inherits from nn.Models for PyTorch compatibility.
    """

    def __init__(self, config):
        """
        Initialize base class, defines interface for all models.
        (Stores config hyperparameters and initializes input shape and number of classes)
        Parameters:
            config: configuration dictionary (hyperparameters)
        Returns:
            None
        """

        super().__init__()          #call nn.Module init
        self.config = config        #store config for later use
        self.model = None           #initialize model
        self.is_trained = False     #set trained to false
        #initialize model attributes
        self.input_shape = config.get('input_shape', None)  
        self.num_classes = config.get('num_classes', 2)