#PyTorch neural network implementations (SimpleCNN, ResNet, Transformer)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import models                          
from codebase.models.base_model import BaseModel
from codebase.utils.helpers import create_batches

class SimpleCNN(BaseModel):
    def __init__(self, config):
        """
        Initialize CNN with convolutional layers
        
        Parameters:
            config : configuration 
        
        Returns:
            None
        """
        super().__init__(config)                                    #call parent class init 
        
        #extract config parameters
        input_channels = config.get('input_channels', 3)            #RGB images = 3 channels
        num_classes = config.get('num_classes', 2)                  #num of output classes
        
        #feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)    #input -> 32 feat maps
        self.relu1 = nn.ReLU()                                                  #activation function(relu)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                      #reduce spatial dimensions (by 2x)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)        #32 -> 64 feat maps (deeper)
        self.relu2 = nn.ReLU()                                          #activation again
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)              #downsample by 2x again
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)       #64 -> 128 feat maps (even deeper)
        self.relu3 = nn.ReLU()                                          #activation again
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)              #downsample by 2x again
        
        self.flatten = nn.Flatten()                                     #flatten layer (3d tensor -> 1d for dense layers)
        
        #calculate flattened size (depends on input image size)
        #Ex: input: 224x224, after 3 poolings (each divides by 2)(2^3 = 8): 224/8 = 28x28
        #flattened_size = 128 channels * 28 * 28 = 100352
        flattened_size = config.get('flattened_size', 128 * 28 * 28)
        
        #fully connected (dense) layers (classification)
        self.fc1 = nn.Linear(flattened_size, 256)                       #1st dense layer (256 neurons)
        self.relu4 = nn.ReLU()                                          #activation
        self.dropout = nn.Dropout(0.5)                                  #dropout layer (prevents overfitting)
        
        self.fc2 = nn.Linear(256, num_classes)                          #output layer (class logits)

        #SimpleCNN IS the model (inherits from nn.Module via BaseModel)
        #No need for self.model = self (causes recursion issues)

        self.criterion = nn.CrossEntropyLoss()                          #loss function -> classification(CE loss)
        self.optimizer = optim.Adam(self.parameters(), lr=config.get('learning_rate', 0.001))  #optimizer
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Parameters:
            x       : input tensor, shape (batch_size, channels, height, width)
        
        Returns:
            output  : logits -> each class, shape (batch_size, num_classes)
        """
        #convolutional block 1
        x = self.conv1(x)           #1st convolution
        x = self.relu1(x)           #activation
        x = self.pool1(x)           #downsample 
        
        #convolutional block 2
        x = self.conv2(x)           #2nd convolution
        x = self.relu2(x)           #activation
        x = self.pool2(x)           #downsample
        
        #convolutional block 3
        x = self.conv3(x)           #3rd convolution
        x = self.relu3(x)           #activation
        x = self.pool3(x)           #downsample
        
        #flatten & dense layers
        x = self.flatten(x)         #flatten 3D tensor -> 1D (batch_size, flattened_size)
        x = self.fc1(x)             #1st dense layer
        x = self.relu4(x)           #activation
        x = self.dropout(x)         #dropout (regularization)
        x = self.fc2(x)             #output layer (logits(!activation))
        
        return x                    #class scores (logits)
    
    def fit(self, train_data, val_data):
        """
        Train CNN using pytorch training loop (standalone training without Trainer class)
        Note: Use Trainer.fit() for full training with callbacks. This is for direct training.
        
        Parameters:
            train_data  : (X_train, y_train) tuple
            val_data    : (X_val, y_val) tuple
        
        Returns:
            history     : dict with training metrics
        """
        
        X_train, y_train = train_data                           #unpack train data
        X_val, y_val = val_data                                 #unpack validation data
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}    #init history
        
        num_epochs = self.config.get('epochs', 100)             #num of epochs
        batch_size = self.config.get('batch_size', 32)          #batch size
        
        for epoch in range(num_epochs):
            super().train()                                     #mode = training mode (allows dropout/batch norm)
            epoch_loss = 0                                      #total loss for this epoch
            num_batches = 0                                     #count batches for averaging
            
            #training on mini-batches
            for batch_X, batch_y in create_batches(X_train, y_train, batch_size):
                
                #convert to tensors if numpy
                if isinstance(batch_X, np.ndarray):
                    batch_X = torch.from_numpy(batch_X).float() #X to float tensor
                    batch_y = torch.from_numpy(batch_y).long()  #y to long tensor (CE Loss)
                
                self.optimizer.zero_grad()                      #reset prev grads 
                predictions = self.forward(batch_X)             #forward pass (input -> model -> predictions)
                loss = self.criterion(predictions, batch_y)     #compute loss (predictions & true labels)
                loss.backward()                                 #compute grad (backpropagation)
                self.optimizer.step()                           #update weights (apply grad)
                
                epoch_loss += loss.item()                       #sum loss (tensor -> python number)
                num_batches += 1                                #increase batch
            
            avg_train_loss = epoch_loss / num_batches           #compute avg train loss
            
            #compute train accuracy (full train set)
            super().eval()                                      #mode = evaluation mode (!dropout)

            with torch.no_grad():                               #!gradient computation (saves memory)

                train_predictions = self.predict(X_train)       #predictions (entire training set)

                if isinstance(train_predictions, torch.Tensor):
                    train_predictions = train_predictions.numpy()  #->numpy if needed

                train_pred_classes = train_predictions.argmax(axis=1)       #get class with highest probability
                train_acc = accuracy_score(y_train, train_pred_classes)     #compare predictions to true labels
            
            #validation metrics (once per epoch)
            val_metrics = self.evaluate((X_val, y_val))         #eval model performance on val set
            
            #store metrics for this epoch
            history['train_loss'].append(avg_train_loss)        #record average training loss
            history['val_loss'].append(val_metrics['loss'])     #record validation loss (detect overfitting)
            history['train_acc'].append(train_acc)              #record training accuracy
            history['val_acc'].append(val_metrics['accuracy'])  #record validation accuracy (true performance indicator)
            
            #print progress every 10 epochs (logging)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        self.is_trained = True                                  #model = trained
        return history                                          #training history

class ResNetModel(BaseModel):
    def __init__(self, config):
        """
        Initialize ResNet using pretrained model from torchvision
        
        Parameters:
            config : configuration 
        
        Returns:
            None
        """
        super().__init__(config)                                    #call parent class init
        
        # extract config parameters
        num_classes = config.get('num_classes', 2)                  #num of output classes
        pretrained = config.get('pretrained', True)                 #whether to use pretrained weights
        resnet_version = config.get('resnet_version', 'resnet18')   #resnet version (resnet18, resnet50, etc)
        
        #load pretrained ResNet model
        if resnet_version == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)    #resnet-18 (11M param, fastest)
        elif resnet_version == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)    #resnet-34 (21M param)
        elif resnet_version == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)    #resnet-50 (25M param, good balance)
        else:
            self.resnet = models.resnet18(pretrained=pretrained)    #default: resnet-18
        
        #num of feat from last layer
        num_features = self.resnet.fc.in_features               #original resnet outputs 1000 classes (imagenet)
        
        #replace final fully connected layer for task
        self.resnet.fc = nn.Linear(num_features, num_classes)   #adapt to our number of classes
        
        #store as model (base class compatibility)
        self.model = self.resnet                                #self.model points to resnet
        
        #training components
        self.criterion = nn.CrossEntropyLoss()                  #loss function (classification)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))  #optimizer
    
    def forward(self, x):
        """
        Forward pass through resenet
        
        Parameters:
            x       : input tensor, shape (batch_size, channels, height, width)
        
        Returns:
            output  : logits for each class, shape (batch_size, num_classes)
        """
        return self.resnet(x)       #resnet handles forward pass (all residual blocks)

class TransformerModel(BaseModel):
    def __init__(self, config):
        """
        Initialize vision transformer for image classification
        
        Parameters:
            config : configuration 
        
        Returns:
            None
        """
        super().__init__(config)                                #call parent class init
        
        #config parameters
        input_channels = config.get('input_channels', 3)        #RGB = 3 channels
        num_classes = config.get('num_classes', 2)              #num of output classes
        image_size = config.get('image_size', 224)              #input image dimensions
        patch_size = config.get('patch_size', 16)               #size of image patches (16x16)
        embed_dim = config.get('embed_dim', 256)                #embedding dimension
        num_heads = config.get('num_heads', 8)                  #num of attention heads
        num_layers = config.get('num_layers', 6)                #num of transformer layers
        mlp_dim = config.get('mlp_dim', 512)                    #feedforward dimension
        dropout = config.get('dropout', 0.1)                    #dropout rate
        
        #calculate num of patches
        self.num_patches = (image_size // patch_size) ** 2      #Ex: 224/16 = 14, so 14*14 = 196 patches
        
        #patch embedding layer (image patches -> embeddings)
        self.patch_embed = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  #each patch -> embed_dim vector
        
        #positional embeddings (learnable position encoding)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))  #+1 for class token
        
        #class token (classification)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  #learnable class token
        
        #transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,                                  #embedding dimension
            nhead=num_heads,                                    #num of attention heads
            dim_feedforward=mlp_dim,                            #feedforward network dimension
            dropout=dropout,                                    #dropout rate
            activation='gelu',                                  #GELU activation (better than relu for transformers)
            batch_first=True                                    #batch dimension first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  #stack multiple layers
        
        #classification head 
        self.norm = nn.LayerNorm(embed_dim)                     #layer norm
        self.fc = nn.Linear(embed_dim, num_classes)             #final classification layer
        
        self.dropout = nn.Dropout(dropout)                      #dropout
        
        #TransformerModel IS the model (inherits from nn.Module via BaseModel)
        #No need for self.model = self (causes recursion issues)
        
        # training components
        self.criterion = nn.CrossEntropyLoss()                  #loss function
        self.optimizer = optim.Adam(self.parameters(), lr=config.get('learning_rate', 0.001))  #optimizer
    
    def forward(self, x):
        """
        Forward pass through vision transformer
        
        Parameters:
            x       : input tensor, shape (batch_size, channels, height, width)
        
        Returns:
            output  : logits for each class, shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]                                 #batch size
        
        #patch embedding: (batch, 3, 224, 224) -> (batch, embed_dim, 14, 14)
        x = self.patch_embed(x)                                 #splits image into patches & embed
        
        #flatten patches: (batch, embed_dim, 14, 14) -> (batch, 196, embed_dim)
        x = x.flatten(2).transpose(1, 2)                        #reshape to sequence of patch embeddings
        
        #add class token: (batch, 196, embed_dim) -> (batch, 197, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  #replicate token for each batch
        x = torch.cat([cls_tokens, x], dim=1)                   #prepend token to sequence
        
        #add positional embeddings
        x = x + self.pos_embed                                  #add learned position information
        x = self.dropout(x)                                     #dropout for regularization
        
        #transformer encoder (multi-head attention + feedforward)
        x = self.transformer(x)                                 #process sequence through transformer layers
        
        #classification (use class token output)
        x = self.norm(x[:, 0])                                  #layer norm on token (first token)
        x = self.fc(x)                                          #classify using token representation
        
        return x                                                #class logits