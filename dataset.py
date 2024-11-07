import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(self, n_samples=1000, n_features=10):
        self.X, self.y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2, random_state=42)
        
        # Normalize the features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        # Convert to PyTorch tensors
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y).unsqueeze(1)
        
        # Split into training, validation, and test sets
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 20% of 80% for validation (i.e., 16% of total data)
    
    def get_train_data(self):
        return self.X_train, self.y_train
    
    def get_validation_data(self):
        return self.X_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.y_test
