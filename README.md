## Lab-1
# Binary Classification Neural Network

This project implements a multilayer (deep) neural network for binary classification. The network is built using custom code and trained on a dataset using various configurations of learning rates and epochs.

## Project Structure

- **main.py**: The main script to execute the neural network training and testing.
- **dataset.py**: Contains the code to generate, split, and load the dataset used for training, validation, and testing.
- **model.py**: Implements the structure of the neural network including layers and activation functions.
- **trainer.py**: Manages the training process, including forward and backward propagation and optimization (using gradient descent).

## Model Architecture

The neural network used in this project follows this architecture:

- Layer 1: 10 neurons with **ReLU** activation
- Layer 2: 8 neurons with **ReLU** activation
- Layer 3: 8 neurons with **ReLU** activation
- Layer 4: 4 neurons with **ReLU** activation
- Layer 5: 1 neuron with **Sigmoid** activation (for binary classification)

## Training Process

The network is trained using **binary cross-entropy loss** and **Adam optimizer**.

### Example Output
Below is a sample output during training:

Epoch [100/3000], Loss: 0.7232 Epoch [200/3000], Loss: 0.7213 Epoch [300/3000], Loss: 0.7195 ...

After training, the network will output predictions for the test set, like:

tensor([[0.1288], [0.9723], [0.2907], ...])


### Prediction Output
The predictions represent probabilities. Values near `0` classify the input as class `0`, while values near `1` classify the input as class `1`.

## Requirements

- Python 3.10+
- Required libraries:
  - `numpy`
  - `torch` (PyTorch)

Install the dependencies with:

pip install numpy torch

How to Run

1. Clone the repository:
(https://github.com/mtrishal123/Lab_1.git)
cd Lab_1

2. Prepare the dataset. Either generate random samples or place your images in the appropriate directories.

3. Run the training:
python main.py

Configuration
Learning Rate: Set in trainer.py, default is 0.01.
Epochs: Modify the epochs parameter in main.py to control the number of iterations (default is 3000).

## Lab-2
# Adjustments in Binary Classification Neural Network

In Lab-2, the following modifications were made to enhance the model's training:

Changes Made

Learning Rate Adjustment: The learning rate was modified to 0.005 to improve the training stability and control the loss fluctuation.
Early Stopping: The training process now includes early stopping, terminating at epoch 1830 to avoid overfitting and reduce unnecessary computation time.
Test Accuracy: With these changes, the model achieved a test accuracy of 83.5%.
Example Output for Lab-2
Epoch [1800/3000], Loss: 0.2972, Val Loss: 0.3004 ...

After implementing early stopping, the training halted at epoch 1830 with a test accuracy of 83.5%.

## Key Updates in Lab 3
### Dataset Improvements
1. Introduced slight random noise to input features to improve robustness.
2. Ensured proper normalization using StandardScaler.

### Model Architecture
The neural network architecture is designed as follows:

1. Input Layer: 10 neurons
2. Hidden Layers:
    1. Layer 1: 16 neurons, ReLU activation
    2. Layer 2: 16 neurons, ReLU activation
    3. Layer 3: 8 neurons, ReLU activation
3. Output Layer: 1 neuron, Sigmoid activation (for binary classification)
4. Regularization: Dropout layers were added to prevent overfitting.
### Hyperparameter Tuning
1. Learning Rate: Tuned dynamically using a scheduler with initial learning_rate=0.005, decaying by a factor of gamma=0.8 every step_size=200 epochs.
2. Batch Size: Set to 32 for balanced optimization.
3. Optimization Algorithm: Switched to Adam optimizer for faster convergence.
### Training Enhancements
1. Early stopping was initially used but later removed to allow complete utilization of 1000 epochs.
2. Regular monitoring of validation loss ensured no overfitting.
3. Final configuration achieved convergence at 1000 epochs.

### Results
 1. Training Epochs: 1000
 2. Final Test Accuracy: 94.67%
 3. Output for Lab 3 is lab_3_output.png
