#PyTorch neural network implementations (ResNet)
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from codebase.models.base_model import BaseModel


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
        
        #load pretrained ResNet model using new weights API
        if resnet_version == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet18(weights=weights)    #resnet-18 (11M param, fastest)
        elif resnet_version == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet34(weights=weights)    #resnet-34 (21M param)
        elif resnet_version == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet50(weights=weights)    #resnet-50 (25M param, good balance)
        else:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet18(weights=weights)    #default: resnet-18
        
        #num of feat from last layer
        num_features = self.resnet.fc.in_features               #original resnet outputs 1000 classes (imagenet)
        
        #replace final fully connected layer for task
        self.resnet.fc = nn.Linear(num_features, num_classes)   #adapt to our number of classes
        
        #store as model (base class compatibility)
        self.model = self.resnet                                #self.model points to resnet
    
    def forward(self, x):
        """
        Forward pass through resenet
        
        Parameters:
            x       : input tensor, shape (batch_size, channels, height, width)
        
        Returns:
            output  : logits for each class, shape (batch_size, num_classes)
        """
        return self.resnet(x)       #resnet handles forward pass (all residual blocks)