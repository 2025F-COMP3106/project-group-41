# Abstract base class for all models
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
from torch.nn import Module
from sklearn.metrics import accuracy_score, precision_score, recall_score
from codebase.utils.helpers import create_batches

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

    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
            X           : Input features, shape (n_samples, n_features)
        
        Returns:
            predictions : Predicted class labels or probabilities; For classification: shape (n_samples,) or (n_samples, n_classes)
                                                                    For regression: shape (n_samples,)
        """

        if not self.is_trained:                                     #check if model is trained
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()                                           #set to evaluation mode (disables dropout/batch norm training behavior)
        
        #convert numpy to tensor if needed
        is_numpy = isinstance(X, np.ndarray)                        #check if input is numpy array
        if is_numpy:
            X = torch.from_numpy(X).float()                         #convert tensor
        
        #predictions without gradients 
        with torch.no_grad():                                       #disable gradient computation
            predictions = self.model(X)                             #forward pass through model
            
            #apply activation based on task type
            if self.config.get('task') == 'classification':         #for classification tasks
                predictions = torch.softmax(predictions, dim=1)     #convert logits to probabilities
        
        if is_numpy:
            predictions = predictions.numpy()                       #convert back to numpy            
        
        return predictions                                          #return the prediction

    def save(self, path):
        """
        Save model info on device.
        
        Parameters:
            path : File path where model should be saved
        
        Returns:
            None
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)       #create directory if not exist
        
        model_type = self.config.get('model_type', 'pytorch')   #get model type 
        
        if model_type == 'pytorch':                             #pytorch models
            torch.save({
                'model_state_dict': self.model.state_dict(),    #save model weights
                'config': self.config                           #save configuration
            }, path)
        elif model_type == 'sklearn':                           #sklearn models
            with open(path, 'wb') as f:                         #open file in binary write mode
                pickle.dump(self.model, f)                      #serialize & save model
        
        print(f"Model saved to {path}")                         #confirmation

    def load(self, path):
        """
        Load model info on device.
        
        Parameters:
            path : File path to saved model 
        
        Returns:
            None (modifies self.model in place)
        """
        if not os.path.exists(path):                            #check if file exists            
            raise FileNotFoundError(f"Model file not found: {path}")

        model_type = self.config.get('model_type', 'pytorch')   #get model type
        
        if model_type == 'pytorch':                             #pytorch models
            checkpoint = torch.load(path)                       #load save
            self.config = checkpoint['config']                  #restore config
            self.build_model()                                  #rebuild model
            self.model.load_state_dict(checkpoint['model_state_dict'])  #load weights into model

        elif model_type == 'sklearn':                           #sklearn models
            with open(path, 'rb') as f:                         #open file in binary read mode
                self.model = pickle.load(f)                     #deserialize & load model
        
        self.is_trained = True                                  #set model = trained
        print(f"Model loaded from {path}")                      #confirmation message