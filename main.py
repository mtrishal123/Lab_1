from dataset import Dataset
from model import BinaryClassificationNN
from trainer import Trainer
import torch

def main():
    # Step 1: Create the dataset
    dataset = Dataset(n_samples=1000, n_features=10)
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_validation_data()  # New validation data split
    X_test, y_test = dataset.get_test_data()

    # Step 2: Initialize the model
    model = BinaryClassificationNN()

    # Step 3: Initialize the trainer with early stopping and learning rate scheduling
    trainer = Trainer(model, learning_rate=0.005, step_size=500, gamma=0.5, patience=100)

    # Step 4: Train the model
    trainer.train(X_train, y_train, X_val, y_val, epochs=3000)

    # Step 5: Test the model and evaluate accuracy
    predictions = trainer.test(X_test)
    accuracy = trainer.calculate_accuracy(y_test, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Predictions for the test set: {predictions}")

if __name__ == "__main__":
    main()
