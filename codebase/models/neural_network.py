#Defines the neural network model for skin cancer classification ResNet-18
import torch.nn as nn
from torchvision import models                          
from codebase.models.base_model import BaseModel

class ResNetModel(Basemodel):
    def __init__(self, config):
        """
        Initialize ResNet using pre-trained model from torchvision
        Parameters:
            config: Model settings (num_classes, pretrained, resnet_version)
        Returns:
            None
        """
        super().__init()__(config)              #call parent initalize class

        #extract config parameters
        num_classes = config.get('num_classes', 2)   #num of output classes
        pretrained = config.get('pretrained', True)  #whether to use pre-trained weights
        resnet_version = config.get('resnet_version', 'resnet18')   #resnet version (resnet18, resnet50, etc)

        #load pre-trained ResNet model
        if resnet_version == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)    #resnet-18 (11M param, fastest)
        elif resnet_version == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)    #resnet-34 (21M param)
        elif resnet_version == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)    #resnet-50 (25M param, good balance)
        else:
            self.resnet = models.resnet18(pretrained=pretrained)    #default: resnet-18
        
        #num of features from last layer
        num_features = self.resnet.fc.in_features   #original resnet outputs 1000 classes (imagenet)

        #replace final fully connected layer for task
        self.resnet.fc = nn.Linear(num_features, num_classes)       #adapt to number of classes

        #store as model (base class compatibility)
        self.model = self.resnet                     #model points to resnet    

    def foward(self, x):
        """
        Foward pass through resnet
        Parameters:
            x: input tensor, shape(batch_size, channels, height, width)
        Returns:
            output: logits for each class, shape(batch_size, num_classes)
        """
        return self.resnet(x)