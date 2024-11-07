import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

class Trainer:
    def __init__(self, model, learning_rate=0.0005, batch_size=32, step_size=500, gamma=0.5, patience=100):
        self.model = model
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.batch_size = batch_size
        self.early_stopping_patience = patience

    def train(self, X_train, y_train, X_val, y_val, epochs=3000):
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        early_stopping_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            
            # Validation loss
            val_loss = self.evaluate(X_val, y_val)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Update learning rate
            self.scheduler.step()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

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
