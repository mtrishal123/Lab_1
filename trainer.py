import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(self, model, learning_rate=0.001, batch_size=32, patience=10):
        self.model = model
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=patience)
        self.batch_size = batch_size

    def train(self, X_train, y_train, X_val, y_val, epochs=1000):
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            val_loss = self.evaluate(X_val, y_val)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
            self.scheduler.step(val_loss)

    def evaluate(self, X_val, y_val):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
            val_loss = self.criterion(outputs, y_val).item()
        return val_loss

    def test(self, X_test):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test)
        return predictions

    def calculate_accuracy(self, y_true, y_pred):
        predicted_classes = (y_pred > 0.5).float()
        accuracy = (predicted_classes == y_true).float().mean().item()
        return accuracy
