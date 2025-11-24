#Model interpretability tools (Grad-CAM, LIME, SHAP) for explaining predictions
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import shap

class ModelExplainer:
    def __init__(self, model):
        """
        Initialize explainability tools for a trained model
        
        Parameters:
            model : trained model object (CNN, ResNet, Transformer, etc)
        
        Returns:
            None
        """
        self.model = model                                      #store the model
        if hasattr(model, 'model'):                             #if model wraps another model
            self.core_model = model.model                       #get the core model (for pytorch models)
        else:
            self.core_model = model
    
    def grad_cam(self, image, target_layer=None, target_class=None):
        """
        Grad-CAM: Gradient-weighted class activation mapping for cnns
        Shows which parts of the image matter for the prediction
        
        Parameters:
            image        : input image tensor, shape (1, channels, height, width)
            target_layer : layer to visualize (default: last conv layer)
            target_class : class to explain (default: predicted class)
        
        Returns:
            heatmap      : activation heatmap overlayed on image
        """
        #make sure model is eval mode
        self.core_model.eval()
        
        #if target layer not specified; try to find last conv layer
        if target_layer is None:
            #for simplecnn, use conv3; for ResNet, use layer4
            if hasattr(self.core_model, 'conv3'):
                target_layer = self.core_model.conv3
            elif hasattr(self.core_model, 'layer4'):
                target_layer = self.core_model.layer4
            else:
                raise ValueError("Cannot auto-detect target layer. Please specify manually.")
        
        #forward pass with hook to capture activations
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)                          #save activation maps
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])                    #save gradients
        
        #register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        #forward pass
        output = self.core_model(image)
        
        #get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        #backward pass for target class
        self.core_model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        #remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        #compute grad-cam
        activation = activations[0].squeeze(0)                  #shape: (channels, height, width)
        gradient = gradients[0].squeeze(0)                      #shape: (channels, height, width)
        
        #global average pooling on gradients to get weights
        weights = gradient.mean(dim=(1, 2))                     #shape: (channels,)
        
        #weighted combination of activation maps
        cam = torch.zeros(activation.shape[1:], dtype=torch.float32)  #shape: (height, width)
        for i, w in enumerate(weights):
            cam += w * activation[i]                            #weighted sum
        
        #apply relu (only positive contributions)
        cam = F.relu(cam)
        
        #normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        #resize to original image size
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))     #resize to (width, height)
        
        return cam                                                  #heatmap
    
    def lime_explain(self, instance, num_samples=1000, num_features=10):
        """
        Perturbs input to see how output changes, shows local feature importance
        
        Parameters:
            instance     : single input sample to explain (image)
            num_samples  : number of perturbed samples to generate
            num_features : number of top features to show
        
        Returns:
            explanation  : LIME explanation object with feature importance
        """
        #lime image explainer
        explainer = lime_image.LimeImageExplainer()
        
        #prediction function wrapper
        def predict_fn(images):
            """Wrapper for model prediction that lime can use"""
            predictions = []
            for img in images:
                #convert to tensor if needed
                if isinstance(img, np.ndarray):
                    img_tensor = torch.from_numpy(img).float()
                    #add batch and channel dims if needed
                    if len(img_tensor.shape) == 3:                              #(H, W, C)
                        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)   #(1, C, H, W)
                else:
                    img_tensor = img
                
                #get prediction
                pred = self.model.predict(img_tensor)
                
                #convert to probabilities
                if isinstance(pred, torch.Tensor):
                    pred = pred.numpy()
                
                #ensure 2D (batch, num_classes)
                if len(pred.shape) == 1:
                    #convert class labels to probabilities
                    num_classes = self.model.config.get('num_classes', 2)
                    prob = np.zeros(num_classes)
                    prob[int(pred[0])] = 1.0
                    pred = prob
                else:
                    pred = pred.squeeze()
                
                predictions.append(pred)
            
            return np.array(predictions)
        
        #prepare image for lime (needs to be H x W x C format)
        if isinstance(instance, torch.Tensor):
            instance = instance.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        
        #normalize to [0, 1] if needed
        if instance.max() > 1.0:
            instance = instance / 255.0
        
        #explain the instance
        explanation = explainer.explain_instance(
            instance,
            predict_fn,
            top_labels=1,                                   #explain top predicted class
            hide_color=0,                                   #use black for hidden regions
            num_samples=num_samples,                        #number of perturbed samples
            segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)
        )
        
        return explanation                                  # return lime explanation object
    
    def shap_explain(self, instances, background_samples=100):
        """
        Shows global feature importance across multiple instances
        
        Parameters:
            instances           : multiple input samples to explain
            background_samples  : number of background samples for SHAP
        
        Returns:
            shap_values         : SHAP values for each instance
        """
        #create wrapper for model prediction
        def predict_wrapper(X):
            """Wrapper that converts inputs and returns predictions"""
            predictions = []
            for sample in X:
                #reshape if needed
                if len(sample.shape) == 1:                  #if flattened
                    #try to reshape to image format
                    #assume square images for simplicity
                    side = int(np.sqrt(len(sample) / 3))
                    sample = sample.reshape(3, side, side)
                
                #get prediction
                pred = self.model.predict(sample[np.newaxis, ...])
                
                if isinstance(pred, torch.Tensor):
                    pred = pred.numpy()
                
                #ensure we return probabilities
                if len(pred.shape) == 1 or pred.shape[1] == 1:
                    num_classes = self.model.config.get('num_classes', 2)
                    prob = np.zeros((1, num_classes))
                    prob[0, int(pred[0])] = 1.0
                    pred = prob
                
                predictions.append(pred.squeeze())
            
            return np.array(predictions)
        
        #flatten instances for shap
        if len(instances.shape) == 4:                       #(batch, C, H, W)
            batch_size = instances.shape[0]
            instances_flat = instances.reshape(batch_size, -1)
        else:
            instances_flat = instances
        
        #use subset as background
        background = instances_flat[:min(background_samples, len(instances_flat))]
        
        #create shap explainer (kernelexplainer is model-agnostic)
        explainer = shap.KernelExplainer(predict_wrapper, background)
        
        # compute SHAP values
        shap_values = explainer.shap_values(instances_flat, nsamples=100)
        
        #reshape back to image format if needed
        if len(instances.shape) == 4:
            original_shape = instances.shape
            shap_values_reshaped = []
            for sv in shap_values:                          #for each class
                sv_reshaped = sv.reshape(original_shape)
                shap_values_reshaped.append(sv_reshaped)
            return shap_values_reshaped
        
        return shap_values                                  #shap values
    
    def visualize_heatmap(self, image, heatmap, alpha=0.4, save_path=None):
        """
        Overlay heatmap on original image for visualization
        
        Parameters:
            image           : original image (numpy array or tensor)
            heatmap         : grad-cam or attention heatmap
            alpha           : transparency of overlay (0=invisible, 1=opaque)
            save_path       : optional path to save visualization
        
        Returns:
            overlayed_image : image with heatmap overlay
        """
        #convert image to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        
        #normalize image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        #convert heatmap to RGB
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]         #apply jet colormap
        
        #overlay
        overlayed = image * (1 - alpha) + heatmap_colored * alpha
        overlayed = np.clip(overlayed, 0, 1)
        
        #visualize
        if save_path:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap, cmap='jet')
            plt.title('Heatmap')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlayed)
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        return overlayed                                        #overlayed image