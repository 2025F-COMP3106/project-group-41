import numpy as np
import torch
from codebase.models.base_model import BaseModel

class EnsembleModel(BaseModel):
    def __init__(self, models_list, config=None):
        """
        Initialize ensemble with multiple trained models
        
        Parameters:
            models_list : list of trained model objects (SimpleCNN, ResNet, RandomForest, etc)
            config      : optional configuration dict
        
        Returns:
            None
        """
        #initialize base class
        if config is None:
            config = {}
        super().__init__(config)                                #call parent class init
        
        #store models
        self.models = models_list                               #list of trained models
        self.num_models = len(models_list)                      #num of models in ensemble
        
        #make sure models are trained
        for i, model in enumerate(self.models):
            if not model.is_trained:
                raise ValueError(f"Model {i} not trained. All models must be trained")
        
        self.is_trained = True                                  #ensemble = trained
        self.model = self                                       #self = model(base class compatibility)
    
    def predict(self, X):
        """
        Get predictions from each model and combine via majority voting
        
        Parameters:
            X           : input features
        
        Returns:
            predictions : ensemble predictions (majority vote for classification)
        """
        all_predictions = []                                    #collect predictions from all models
        
        #get predictions from each model
        for model in self.models:
            pred = model.predict(X)                             #predictions from current model
            
            #convert to class labels if probabilities
            if len(pred.shape) > 1 and pred.shape[1] > 1:       #if shape is (batch, num_classes)
                if isinstance(pred, torch.Tensor):
                    pred = pred.argmax(dim=1).numpy()           #get class with max probability
                else:
                    pred = pred.argmax(axis=1)                  #numpy version
            
            all_predictions.append(pred)                        #add to collection
        
        #stack predictions: shape (num_models, num_samples)
        all_predictions = np.array(all_predictions)
        
        #majority voting: for each sample, pick most common prediction
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),                  #count votes, pick max
            axis=0,                                             #vote across models (axis 0)
            arr=all_predictions
        )
        
        return ensemble_pred                                    #final ensemble predictions
    
    def weighted_ensemble(self, X, weights):
        """
        Weight each models prediction differently based on validation performance
        
        Parameters:
            X           : input features
            weights     : list/array of weights for each model (should sum to 1.0)
        
        Returns:
            predictions : weighted ensemble predictions
        """
        if len(weights) != self.num_models:                     #num of weights dont match num of models
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({self.num_models})")
        
        #normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()                       #ensure weights sum to 1.0
        
        all_predictions = []                                    #predictions
        
        #predictions from each model
        for model in self.models:
            pred = model.predict(X)                             #get current prediction
            
            #convert to probabilities if needed
            if len(pred.shape) == 1 or pred.shape[1] == 1:      #if class labels

                #convert to one-hot probabilities
                num_classes = self.config.get('num_classes', 2)
                prob = np.zeros((len(pred), num_classes))
                prob[np.arange(len(pred)), pred] = 1.0
                pred = prob
            
            if isinstance(pred, torch.Tensor):
                pred = pred.numpy()                             #convert to numpy
            
            all_predictions.append(pred)                        #add to collection
        
        #weighted average of probabilities
        weighted_pred = np.zeros_like(all_predictions[0])       #initialize result
        for i, pred in enumerate(all_predictions):
            weighted_pred += weights[i] * pred                  #add weighted contribution
        
        #final class predictions
        ensemble_pred = weighted_pred.argmax(axis=1)            #pick class with highest weighted probability
        
        return ensemble_pred                                    #weighted ensemble predictions