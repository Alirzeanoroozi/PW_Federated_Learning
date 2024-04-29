import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import copy
from collections import OrderedDict

from models import load_model
from utils import evaluate_server, print_time, get_latest_model


class Client:
    def __init__(self, train_loader, config):
        self.lr = config['lr']
        self.train_loader = train_loader
        self.mdl = load_model(config)
        self.n_epochs = config['client_iterations']
        self.config = config

    def train_client(self, device):
        self.mdl = self.mdl.to(device)
        self.opt = optim.Adam(self.mdl.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.mdl.train()
        len_train = len(self.train_loader)

        for epochs in range(self.n_epochs):
            cur_loss = 0
            cur_acc = 0
            for idx, (data, target) in enumerate(self.train_loader):
                data = data.to(device)
                target = target.to(device)

                self.opt.zero_grad()
                scores = self.mdl(data)

                if self.config['dataset'] == "Sheks":
                    B, T, C = scores.shape
                    logits = scores.reshape(B * T, C)
                    targets = target.view(B * T)

                    loss = self.loss_fn(logits, targets)
                    cur_loss += loss.item() / len_train

                    scores = scores[:, -1, :]  # becomes (B, C)
                    # apply softmax to get probabilities
                    probs = F.softmax(scores, dim=-1)  # (B, C)
                    _, predicted = torch.max(probs, dim=1)
                    target = target[:, -1]
                else:
                    loss = self.loss_fn(scores, target)
                    cur_loss += loss.item() / len_train

                    _, predicted = torch.max(scores, dim=1)

                correct = (predicted == target).sum()
                samples = scores.shape[0]
                cur_acc += correct / (samples * len_train)

                loss.backward()
                self.opt.step()

            print(f"epochs: [{epochs + 1}/{self.n_epochs}] train_acc: {cur_acc:.3f} train_loss: {cur_loss:.3f}")

    def replace_mdl(self, server_mdl):
        self.mdl = copy.deepcopy(server_mdl)


class Server:
    def __init__(self, config):
        self.mdl = load_model(config)

    def aggregate_models(self, clients_model, training_samples):
        update_state = OrderedDict()
        
        for k, client in enumerate(clients_model):
            local_state = client.state_dict()
            for key in self.mdl.state_dict().keys():
                # Updated to be a weighted average: w_k * n_k / N
                if k == 0:
                    update_state[key] = (local_state[key] * training_samples[k]) / sum(training_samples)
                else:
                    update_state[key] += (local_state[key] * training_samples[k]) / sum(training_samples)
 
        self.mdl.load_state_dict(update_state)


class FedAvg:
    def __init__(self, train_loaders, test_loaders, config):
        self.clients = [Client(train_loader, config) for train_loader in train_loaders]
        self.test_loaders = test_loaders
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.server = Server(config)
        self.config = config

        self.client_iters = config['client_iterations']
        self.n_clients = len(train_loaders)
        self.sample_cli = int(config['sample_clients'] * self.n_clients)

    def train(self):
        accs = []
        loss_s = []
        t_start_time = time.perf_counter()

        train_index, server_model = get_latest_model(self.config)
        if server_model is not None:
            self.server.mdl = server_model

        idx = train_index
        while 1:
            i_start_time = time.perf_counter()

            print(f"iteration [{idx + 1}]")
            clients_selected = random.sample([i for i in range(self.n_clients)], self.sample_cli)
            for i, jdx in enumerate(clients_selected):
                print(f"############## client {jdx} - {i + 1} / {len(clients_selected)} ##############")
                self.clients[jdx].replace_mdl(self.server.mdl)
                self.clients[jdx].train_client(self.device)

            print(f"############## server ##############")
            self.server.aggregate_models([self.clients[i].mdl for i in clients_selected],
                                         [len(self.clients[i].train_loader.dataset) for i in clients_selected])

            acc, loss = evaluate_server(self.server.mdl, [self.test_loaders[i] for i in clients_selected], self.device, self.config)
            accs.append(acc)
            loss_s.append(loss)

            i_end_time = time.perf_counter()
            print_time(i_end_time, i_start_time)

            torch.save(self.server.mdl, f"pre_computed_models/{self.config['dataset']}/{self.config['model']}/{self.config['solver']}/{idx}.pkl")
            if acc > .97:
                break
            idx += 1

        t_end_time = time.perf_counter()
        print_time(t_end_time, t_start_time)

        return accs, loss_s, idx
