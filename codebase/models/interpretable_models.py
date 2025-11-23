import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from codebase.models.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, config):
        """
        Initialize Logistic Regression wrapper (interpretable baseline)
        
        Parameters:
            config : configuration
        
        Returns:
            None
        """
        super().__init__(config)                                #call parent class init
        
        #config parameters
        max_iter = config.get('max_iter', 1000)                 #max iterations(optimization)
        C = config.get('C', 1.0)                                #regularization strength (smaller = stronger)
        solver = config.get('solver', 'lbfgs')                  #optimization algorithm
        
        #create sklearn logistic regression model
        self.model = LogisticRegression(
            max_iter=max_iter,                                  #iterations
            C=C,                                                #regularization
            solver=solver,                                      #optimizer
            random_state=42                                     #for reproducibility
        )
        
        #set model type for base class save/load
        self.config['model_type'] = 'sklearn'                   #tell base class this is sklearn model
    
    def train(self, train_data, val_data):
        """
        Train using sklearns fit method
        
        Parameters:
            train_data  : (X_train, y_train) tuple
            val_data    : (X_val, y_val) tuple
        
        Returns:
            history     : dict with training metrics
        """
        X_train, y_train = train_data                           #unpack train data
        X_val, y_val = val_data                                 #unpack validation data
        
        if len(X_train.shape) > 2:                              #flatten images if needed (sklearn needs 2D input)
            X_train = X_train.reshape(X_train.shape[0], -1)     #flatten to (batch, features)
        if len(X_val.shape) > 2:
            X_val = X_val.reshape(X_val.shape[0], -1)           #flatten validation 
        
        #train model
        self.model.fit(X_train, y_train)                        #sklearn fit method (trains model)
        self.is_trained = True                                  #trained = true
        
        #compute metrics for history
        train_score = self.model.score(X_train, y_train)        #train accuracy
        val_metrics = self.evaluate((X_val, y_val))             #validation metrics
        
        #create history dict (single epoch for sklearn)
        history = {
            'train_loss': [0.0],                                #sklearn !(expose loss)
            'val_loss': [val_metrics['loss']],
            'train_acc': [train_score],
            'val_acc': [val_metrics['accuracy']]
        }
        
        return history
    
    def get_feature_importance(self):
        """
        Return most important features
        
        Returns:
            coefficients : feature (importance weights)
        """
        if not self.is_trained:                                 #untrained
            raise ValueError("Model must be trained first")
        
        #coefficients (weights for each feature)
        return self.model.coef_                                 #shape: (num_classes, num_features)

class RandomForestModel(BaseModel):
    def __init__(self, config):
        """
        Initialize random forest wrapper (interpretable baseline)
        
        Parameters:
            config : configuration
        
        Returns:
            None
        """
        super().__init__(config)                                #call parent class init
        
        #config parameters
        n_estimators = config.get('n_estimators', 100)          #num of trees in forest
        max_depth = config.get('max_depth', None)               #max depth of trees (None = unlimited)
        min_samples_split = config.get('min_samples_split', 2)  #min samples to split node
        
        #create sklearn random forest model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,                          #num of decision trees
            max_depth=max_depth,                                #depth limit
            min_samples_split=min_samples_split,                #split threshold
            random_state=42,                                    #for reproducibility
            n_jobs=-1                                           #use all CPU cores
        )
        
        #set model type for base class save/load
        self.config['model_type'] = 'sklearn'                   #tell base class this is sklearn model
    
    def get_feature_importance(self):
        """
        Return built in feature importance from random forest
        
        Returns:
            importance : feature importance scores (higher = more important)
        """
        if not self.is_trained:                                 #untrained model
            raise ValueError("Model must be trained first")
        
        #feature importance (averaged across all trees)
        return self.model.feature_importances_                  #shape: (num_features)