import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import copy

from utils import evaluate_server, load_model, print_time


class Client:
    def __init__(self, train_loader, config):
        self.lr = config['lr']
        self.mom = config['momentum']
        self.train_loader = train_loader
        self.loss_fn = nn.CrossEntropyLoss()

    def train_client(self, c_mdl, n_epochs, device):
        self.opt = optim.SGD(c_mdl.parameters(), lr=self.lr, momentum=self.mom)

        c_mdl.train()
        len_train = len(self.train_loader)

        for epochs in range(n_epochs):
            cur_loss = 0
            cur_acc = 0
            for idx, (data, target) in enumerate(self.train_loader):
                data = data.to(device)
                target = target.to(device)

                scores = c_mdl(data)
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

            print(f"epochs: [{epochs + 1}/{n_epochs}] train_acc: {cur_acc:.3f} train_loss: {cur_loss:.3f}")

        return c_mdl


class CWT:
    def __init__(self, clients_data_loaders, test_loader, config):
        self.device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.n_clients = config['num_clients']
        self.client_iters = config['client_iterations']
        self.total_iters = config['total_iterations']
        self.config = config
        self.test_loader = test_loader
        self.sample_cli = int(config['sample_clients'] * self.n_clients)
        self.clients = [Client(clients_data, config) for clients_data in clients_data_loaders]
        self.center_mdl = load_model(self.config['model'], self.config['n_class'], channel=self.config['channel']).to(self.device)

    def train(self):
        accs = []
        t_start_time = time.perf_counter()

        for idx in range(self.total_iters):
            i_start_time = time.perf_counter()

            print(f"iteration [{idx + 1}/{self.total_iters}]")
            clients_selected = random.sample([i for i in range(self.n_clients)], self.sample_cli)

            for jdx in clients_selected:
                print(f"############## client {jdx} ##############")
                mdl = self.clients[jdx].train_client(self.center_mdl, self.client_iters, self.device)
                self.center_mdl = copy.deepcopy(mdl)

            print("############## server ##############")
            single_acc = evaluate_server(self.center_mdl, self.test_loader, self.device)
            print(f'cur_acc: {single_acc["acc"]:.3f}')
            accs.append(single_acc["acc"].item())

            i_end_time = time.perf_counter()
            print_time(i_end_time, i_start_time)

            if (idx + 1) % 10 == 0:
                save_checkpoint("fed_avg", self.center_mdl, self.config)

        t_end_time = time.perf_counter()
        print_time(t_end_time, t_start_time)

        return accs
