import random
import numpy as np
import torch
from torch import nn
import os


def initialize_model():
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_latest_model(config):
    folder_path = f"pre_computed_models/{config['dataset']}/{config['model']}/{config['solver']}"
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
        return 0, None
    else:
        files = sorted(os.listdir(folder_path))
        return int(files[-1].split(".")[0]), torch.load(os.path.join(folder_path, files[-1]))


def print_time(end_time, start_time):
    elapsed_time = int(end_time - start_time)
    hr = elapsed_time // 3600
    mi = (elapsed_time - hr * 3600) // 60
    sec = elapsed_time - hr * 3600 - mi * 60
    print(f"training done in {hr} H {mi} M {sec} S")


def evaluate_server(model, loaders, device):
    model = model.to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    total_acc = 0
    total_loss = 0

    for loader in loaders:
        len_train = len(loader)

        cur_acc = 0
        cur_loss = 0

        with torch.no_grad():
            for idx, (data, target) in enumerate(loader):
                data = data.to(device)
                target = target.to(device)

                scores = model(data)

                _, predicted = torch.max(scores, dim=1)
                correct = (predicted == target).sum()
                samples = scores.shape[0]
                cur_acc += correct / (samples * len_train)
                cur_loss += loss_fn(scores, target) / (samples * len_train)

        total_acc += cur_acc / len(loaders)
        total_loss += cur_loss / len(loaders)

    print(f"test_acc: {total_acc:.3f}, test_loss: {total_loss:.3f}")

    return total_acc, total_loss


def load_model(config):
    if config['model'] == "basic":
        return BasicNet()
    elif config['model'] == "2NN":
        return MNIST_2NN()
    elif config['model'] == "LSTM":
        return NextWordModel()


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
        # take only the last output
        x = x[:, -1, :]
        return self.softmax(self.linear(x))
