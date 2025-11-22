# Import sklearn models

class LogisticRegressionModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Wrap sklearn's LogisticRegression
        # These are baselines to compare against
    
    def train(self, train_data, val_data):
        # Use sklearn's .fit() method
        pass
    
    def get_feature_importance(self):
        # Return coefficient weights
        # Shows which features matter most
        pass

class RandomForestModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Wrap sklearn's RandomForestClassifier
    
    def get_feature_importance(self):
        # Built-in feature importance
        pass