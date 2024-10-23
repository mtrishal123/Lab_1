from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch

class Dataset:
    def __init__(self, n_samples=1000, n_features=10):
        self.X, self.y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2, random_state=42)
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y).unsqueeze(1)  # Convert to PyTorch tensors and add extra dimension for binary output
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    
    def get_train_data(self):
        return self.X_train, self.y_train
    
    def get_test_data(self):
        return self.X_test, self.y_test
