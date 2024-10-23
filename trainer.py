import torch
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
    
    def train(self, X_train, y_train, epochs=1000):
        for epoch in range(epochs):
            self.model.train()  # Set the model to training mode
            self.optimizer.zero_grad()  # Clear gradients

            # Forward pass
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def test(self, X_test):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = self.model(X_test)
        return predictions
