import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import copy
from collections import OrderedDict

from utils import evaluate_server, load_model, print_time


class Client:
    def __init__(self, train_loader, config):
        self.lr = config['lr']
        self.mom = config['momentum']
        self.train_loader = train_loader
        self.mdl = load_model(config['model'], config['n_class'], channel=config['channel'])
        self.loss_fn = nn.CrossEntropyLoss()
        self.landa = config['lambda']

    def train_client(self, best_mdl, n_epochs, device):
        self.mdl = self.mdl.to(device)
        self.opt = optim.SGD(self.mdl.parameters(), lr=self.lr, momentum=self.mom)

        self.mdl.train()
        len_train = len(self.train_loader)

        for epochs in range(n_epochs):
            cur_loss = 0
            cur_acc = 0
            for idx, (data, target) in enumerate(self.train_loader):
                data = data.to(device)
                target = target.to(device)

                scores = self.mdl(data)
                loss = self.loss_fn(scores, target) + self.ditto(best_mdl)
                cur_loss = self.loss_fn(scores, target)
                cur_loss += cur_loss.item() / len_train

                scores = F.softmax(scores, dim=1)
                _, predicted = torch.max(scores, dim=1)
                correct = (predicted == target).sum()
                samples = scores.shape[0]
                cur_acc += correct / (samples * len_train)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            print(f"epochs: [{epochs + 1}/{n_epochs}] train_acc: {cur_acc:.3f} train_loss: {cur_loss:.3f}")

        return self.mdl, cur_loss

    def replace_mdl(self, server_mdl):
        self.mdl = copy.deepcopy(server_mdl)

    # @torch.no_grad()
    def ditto(self, best_mdl):
        params1 = dict(self.mdl.named_parameters())
        params2 = dict(best_mdl.named_parameters())

        loss_val = 0

        for name, params in params1.items():
            norm_val = torch.norm(params1[name] - params2[name]) ** 2
            loss_val += self.landa * 0.5 * norm_val

        return loss_val


class Server:
    def __init__(self, config):
        self.mdl = load_model(config['model'], config['n_class'], channel=config['channel'])

    def aggregate_models(self, clients_model):
        update_state = OrderedDict()
        n_clients = len(clients_model)
        for k, client in enumerate(clients_model):
            local_state = client.state_dict()
            for key in self.mdl.state_dict().keys():
                if k == 0:
                    update_state[key] = local_state[key] / n_clients
                else:
                    update_state[key] += local_state[key] / n_clients

        self.mdl.load_state_dict(update_state)


class DITTO:
    def __init__(self, clients_data_loaders, test_loader, config):
        self.device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.n_clients = config['num_clients']
        self.client_iters = config['client_iterations']
        self.total_iters = config['total_iterations']
        self.config = config
        self.test_loader = test_loader
        self.sample_cli = int(config['sample_clients'] * self.n_clients)
        self.clients = [Client(clients_data, config) for clients_data in clients_data_loaders]
        self.server = Server(config)
        self.best_mdl = load_model(self.config['model'], self.config['n_class'], channel=self.config['channel']).to(self.device)

    def train(self):
        accs = []
        t_start_time = time.perf_counter()

        for idx in range(self.total_iters):
            i_start_time = time.perf_counter()

            print(f"iteration [{idx + 1}/{self.total_iters}]")
            clients_selected = random.sample([i for i in range(self.n_clients)], self.sample_cli)

            for pdx in clients_selected:
                self.clients[pdx].replace_mdl(self.server.mdl)

            models = {}
            for jdx in clients_selected:
                print(f"############## client {jdx} ##############")
                mdl, loss = self.clients[jdx].train_client(self.best_mdl, self.client_iters, self.device)
                models[mdl] = loss
            self.best_mdl = copy.deepcopy(min(models, key=models.get))

            print("############## server ##############")
            self.server.aggregate_models([self.clients[i].mdl for i in clients_selected])

            single_acc = evaluate_server(self.server.mdl, self.test_loader, self.device)
            print(f'cur_acc: {single_acc["acc"]:.3f}')
            accs.append(single_acc["acc"].item())

            i_end_time = time.perf_counter()
            print_time(i_end_time, i_start_time)

            if (idx + 1) % 10 == 0:
                save_checkpoint("fed_avg", self.server.mdl, self.config)

        t_end_time = time.perf_counter()
        print_time(t_end_time, t_start_time)

        return accs
