import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from models import load_model
from utils import evaluate_server, print_time


class BaseLine:
    def __init__(self, train_loader, test_loader, config):
        self.device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.client_iters = config['client_iterations']
        self.total_iters = config['total_iterations']
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.mdl = load_model(config['model']).to(self.device)
        self.lr = config['lr']
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        accs = []
        t_start_time = time.perf_counter()

        for idx in range(self.total_iters):
            i_start_time = time.perf_counter()

            print(f"iteration [{idx + 1}/{self.total_iters}]")

            self.opt = optim.Adam(self.mdl.parameters(), lr=self.lr)

            self.mdl.train()
            len_train = len(self.train_loader)

            cur_loss = 0
            cur_acc = 0
            for idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                scores = self.mdl(data)
                loss = self.loss_fn(scores, target)
                cur_loss += loss.item() / len_train

                scores = F.softmax(scores, dim=1)
                _, predicted = torch.max(scores, dim=1)
                correct = (predicted == target).sum()
                samples = scores.shape[0]
                cur_acc += correct / (samples * len_train)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            print(f"train_acc: {cur_acc:.3f} train_loss: {cur_loss:.3f}")

            single_acc = evaluate_server(self.mdl, self.test_loader, self.device)
            print(f'cur_acc: {single_acc["acc"]:.3f}')
            accs.append(single_acc["acc"].item())

            i_end_time = time.perf_counter()
            print_time(i_end_time, i_start_time)

        t_end_time = time.perf_counter()
        print_time(t_end_time, t_start_time)

        return accs
