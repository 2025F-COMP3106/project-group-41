# Training callbacks (early stopping, model checkpointing, logging)

import os
import torch
from torch.nn import Module


class EarlyStopping:

    #Monitors the validation loss and stops training when itstops improving for a certain number of epochs (patience).
    

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience         
        self.min_delta = min_delta        
        self.counter = 0                  
        self.best_loss = None            
        self.should_stop = False          

    def __call__(self, val_loss: float):
       # Update state after each epoch using the new validation loss.

        if self.best_loss is None:
            # first time we see a loss
            self.best_loss = val_loss
            return

        # if val_loss is NOT significantly better than best_loss
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            # improvement: reset patience counter and update best_loss
            self.best_loss = val_loss
            self.counter = 0


class ModelCheckpoint:
    #Saves the model parameters whenever validation loss improves.Only keeps the best version of the model on disk.

    def __init__(self, save_path: str = "checkpoints/best_model.pth"):
        self.save_path = save_path
        self.best_loss = None

    def __call__(self, model: Module, val_loss: float):
        #Check if the current validation loss is the best so far.If yes, save the model weights to disk.
        if (self.best_loss is None) or (val_loss < self.best_loss):
            self.best_loss = val_loss
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            print(f"[Checkpoint] Saved improved model → {self.save_path}")
