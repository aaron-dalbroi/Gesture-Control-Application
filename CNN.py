# Credit for model goes to https://peyrone.medium.com/comparing-svm-and-cnn-in-recognizing-handwritten-digits-an-overview-5ef06b20194e

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First convolution layer: input channels = 1 (grayscale), output = 32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 64)  # Input size = (64 * 7 * 7)
        self.fc2 = nn.Linear(64, 4)  # Output layer with 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> MaxPool (28x28 -> 14x14)
        x = F.relu(self.conv2(x))  # Conv2 -> ReLU (14x14 -> 14x14)
        x = self.pool2(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> MaxPool (14x14 -> 7x7)
        x = x.view(-1, 64 * 7 * 7)  # Flatten (7x7 feature map to vector)
        x = F.relu(self.fc1(x))  # Fully connected layer
        x = self.fc2(x)  # Output layer
        return x

