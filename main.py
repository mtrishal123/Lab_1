from dataset import Dataset
from model import BinaryClassificationNN
from trainer import Trainer

def main():
    dataset = Dataset(n_samples=1500, n_features=10)
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_validation_data()
    X_test, y_test = dataset.get_test_data()

    model = BinaryClassificationNN()

    trainer = Trainer(model, learning_rate=0.001, patience=10)

    trainer.train(X_train, y_train, X_val, y_val, epochs=1000)

    predictions = trainer.test(X_test)
    accuracy = trainer.calculate_accuracy(y_test, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Predictions for the test set: {predictions}")

if __name__ == "__main__":
    main()
