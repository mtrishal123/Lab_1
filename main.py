from dataset import Dataset
from model import BinaryClassificationNN
from trainer import Trainer

def main():
    # Step 1: Create the dataset
    dataset = Dataset(n_samples=1000, n_features=10)
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()

    # Step 2: Initialize the model
    model = BinaryClassificationNN()

    # Step 3: Initialize the trainer
    trainer = Trainer(model, learning_rate=0.01)

    # Step 4: Train the model
    trainer.train(X_train, y_train, epochs=3000)

    # Step 5: Test the model
    predictions = trainer.test(X_test)
    print(f"Predictions for the test set: {predictions}")

if __name__ == "__main__":
    main()
