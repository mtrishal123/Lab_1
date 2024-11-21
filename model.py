import torch.nn as nn

class BinaryClassificationNN(nn.Module):
    def __init__(self):
        super(BinaryClassificationNN, self).__init__()
        self.layer1 = nn.Linear(10, 16)      # Increased neurons
        self.bn1 = nn.BatchNorm1d(16)       # Batch Normalization
        self.layer2 = nn.Linear(16, 12)
        self.bn2 = nn.BatchNorm1d(12)
        self.layer3 = nn.Linear(12, 10)
        self.bn3 = nn.BatchNorm1d(10)
        self.layer4 = nn.Linear(10, 6)
        self.bn4 = nn.BatchNorm1d(6)
        self.output_layer = nn.Linear(6, 1) # Output layer

        self.leaky_relu = nn.LeakyReLU()    # Changed activation
        self.sigmoid = nn.Sigmoid()         # Sigmoid for binary output
        self.dropout = nn.Dropout(0.2)     # Reduced dropout

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn4(self.layer4(x)))
        x = self.sigmoid(self.output_layer(x))
        return x
