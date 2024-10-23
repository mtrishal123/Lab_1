import torch.nn as nn

class BinaryClassificationNN(nn.Module):
    def __init__(self):
        super(BinaryClassificationNN, self).__init__()
        self.layer1 = nn.Linear(10, 10)      # Input layer with 10 neurons
        self.layer2 = nn.Linear(10, 8)       # First hidden layer with 8 neurons
        self.layer3 = nn.Linear(8, 8)        # Second hidden layer with 8 neurons
        self.layer4 = nn.Linear(8, 4)        # Third hidden layer with 4 neurons
        self.output_layer = nn.Linear(4, 1)  # Output layer with 1 neuron

        self.relu = nn.ReLU()                # ReLU activation for hidden layers
        self.sigmoid = nn.Sigmoid()          # Sigmoid activation for output layer
    
    def forward(self, x):
        x = self.relu(self.layer1(x))        # First layer activation
        x = self.relu(self.layer2(x))        # Second layer activation
        x = self.relu(self.layer3(x))        # Third layer activation
        x = self.relu(self.layer4(x))        # Fourth layer activation
        x = self.sigmoid(self.output_layer(x))  # Output layer activation (Sigmoid)
        return x
