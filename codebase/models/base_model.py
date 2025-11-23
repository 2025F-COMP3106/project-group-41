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

    def train(self, train_data, val_data):
        """
        Train model on training data & validate on validation data.

        Parameters:
            train_data  : (X_train, y_train) tuple where X is features, y is labels
            val_data    : (X_val, y_val) tuple for validation during training

        Returns:
            dict        : losses and metrics per epoch
        """

        #unpack data
        X_train, y_train = train_data
        X_val, y_val = val_data

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}    #init history

        num_epochs = self.config.get('epochs', 100)                                     #get number of epochs
        batch_size = self.config.get('batch_size', 32)                                  #get batch size 
        
        #train loop
        for epoch in range(num_epochs):
            epoch_loss = 0                                  #total loss for this epoch
            num_batches = 0                                 #count batches for averaging
            
            #train on mini batches 
            for batch_X, batch_y in create_batches(X_train, y_train, batch_size):
                predictions = self.model.forward(batch_X)   #forward pass: input -> model -> predictions
                loss = compute_loss(predictions, batch_y)   #compute error between predictions & true labels
                gradients = compute_gradients(loss)         #backward pass: compute how to adjust weights
                update_weights(gradients)                   #apply weight updates using optimizer
                epoch_loss += loss                          #accumulate loss for this epoch
                num_batches += 1                            #increment batch counter

            avg_train_loss = epoch_loss/num_batches         #compute avg train loss across all batches

            #compute train accuracy on full training set
            train_predictions = self.predict(X_train)       #get predictions for entire training set
            train_acc = accuracy_score(y_train, train_predictions)  #compare predictions to true labels

            val_metrics = self.evaluate((X_val, y_val))     #evaluate model performance on validation set (once per epoch)

            #store metrics for this epoch
            history['train_loss'].append(avg_train_loss)            #record average training loss
            history['val_loss'].append(val_metrics['loss'])         #record validation loss (detect overfitting)
            history['train_acc'].append(train_acc)                  #record training accuracy
            history['val_acc'].append(val_metrics['accuracy'])      #record validation accuracy (true performance indicator)

        self.is_trained = True
        return history

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
        

    def evaluate(self, test_data):
        """
        Evaluate model performance on test data.
        
        Parameters:
            test_data   : features, true labels
        
        Returns:
            dict        : Dictionary of evaluation metrics
        """
        X_test, y_test = test_data                              #unpack test data
        predictions = self.predict(X_test)                      #get predictions through forward pass
        
        loss = compute_loss(predictions, y_test)                #compute loss 
        
        #compute metrics
        accuracy = accuracy_score(y_test, predictions)          #accuracy: ratio of correct predictions
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)    #precision: TP / (TP + FP), how many predicted positives are correct
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)          #recall: TP / (TP + FN),    how many actual positives we found
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0    #F1-score: harmonic mean of precision and recall
        
        #put all into metrics dictionary
        metrics = {
            'accuracy': accuracy,       #overall correctness
            'precision': precision,     #positive prediction quality
            'recall': recall,           #positive detection rate
            'f1_score': f1,             #balanced precision/recall score
            'loss': loss                #model error/loss value
        }
        
        return metrics

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