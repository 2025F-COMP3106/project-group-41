import numpy as np
import torch
import random
import os


def create_batches(X, y, batch_size):
    """
    Create mini-batches from data for training 
    (FOR MODELS)

    Parameters:
        X                       : features 
        y                       : labels 
        batch_size              : Size of each batch
        
    Returns:
        batch_X, batch_y        : (batch_X, batch_y) for each batch
    """
    n_samples = len(X)                                      #total num of samples
    indices = np.arange(n_samples)                          #array of indices
    np.random.shuffle(indices)                              #shuffle indices
    
    #iterate through data in batch_size chunks
    for start_idx in range(0, n_samples, batch_size):      
        
        end_idx = min(start_idx + batch_size, n_samples)    #handle last batch 
        batch_indices = indices[start_idx:end_idx]          #get indices for this batch
        
        batch_X = X[batch_indices]                          #extract batch features
        batch_y = y[batch_indices]                          #extract batch labels
        
        yield batch_X, batch_y                              #return batch