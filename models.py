import torch
from torch import nn


def load_model(config):
    if config['model'] == "basic":
        return BasicNet()
    elif config['model'] == "2NN":
        return MNIST_2NN()
    elif config['model'] == "LSTM":
        return NextWordModel()
    elif config['model'] == "MLP":
        return MLP()


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x


class MNIST_2NN(nn.Module):
    def __init__(self):
        super(MNIST_2NN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x


class NextWordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(85, 8)
        self.lstm = nn.LSTM(input_size=8, hidden_size=256, num_layers=2, batch_first=True)
        self.linear = nn.Linear(256, 85)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
        # # take only the last output
        # x = x[:, -1, :]
        # return self.softmax(self.linear(x))
