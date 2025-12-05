#Abstract base class for all models
import torch
from torch.nn import Module

class BaseModel(Module):
    """
    Abstract base class, defines interface for all models.
    Inherits from nn.Module for PyTorch compatibility.
    """
    
    def __init__(self, config):
        """
        Initialize the model with configuration parameters.

        Parameters:
            config : configuration dictionary (hyperparameters)

        Returns:
            None
        """
        super().__init__()                                      #call nn.Module init
        self.config = config                                    #store config for later use
        self.model = None                                       #init model 
        self.is_trained = False                                 #set trained false
        self.input_shape = config.get('input_shape', None)      #init model attributes
        self.num_classes = config.get('num_classes', 2)