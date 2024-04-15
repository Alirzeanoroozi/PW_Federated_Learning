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
        self.ci = {name: torch.zeros_like(params) for name, params in self.mdl.named_parameters()}
        self.cpu = torch.device('cpu')

    def train_client(self, server_mdl, c,  n_epochs, device):
        self.put_to_device(c, device)
        c_diff = self.c_diff(c)

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
                loss = self.loss_fn(scores, target)
                cur_loss += loss.item() / len_train

                scores = F.softmax(scores, dim=1)
                _, predicted = torch.max(scores, dim=1)
                correct = (predicted == target).sum()
                samples = scores.shape[0]
                cur_acc += correct / (samples * len_train)

                self.opt.zero_grad()
                loss.backward()
                for name, params in self.mdl.named_parameters():
                    params.grad += c_diff[name]
                self.opt.step()

            print(f"epochs: [{epochs + 1}/{n_epochs}] train_acc: {cur_acc:.3f} train_loss: {cur_loss:.3f}")

        mdl_diff = self.diff_mdl(server_mdl)
        self.ci, diff_c = self.update_c(c, c_diff, mdl_diff, n_epochs)
        self.off_device(c, diff_c)
        return mdl_diff, diff_c

    def put_to_device(self, c, device):
        for name, params in c.items():
            c[name] = c[name].to(device)
            self.ci[name] = self.ci[name].to(device)

    def off_device(self, c, diff_c):
        for name, params in c.items():
            c[name] = c[name].to(self.cpu)
            self.ci[name] = self.ci[name].to(self.cpu)
            diff_c[name] = diff_c[name].to(self.cpu)

    def c_diff(self, c):
        # c - c_i
        c_diff = OrderedDict()
        for name, params in c.items():
            c_diff[name] = c[name] - self.ci[name]
        return c_diff

    def diff_mdl(self, serv_mdl):
        # x - y_i
        mdl_diff = OrderedDict()
        server_parms = dict(serv_mdl.named_parameters())
        for name, params in self.mdl.named_parameters():
            mdl_diff[name] = params - server_parms[name]
        return mdl_diff

    def update_c(self, c, c_diff, mdl_diff, K):
        # c+
        alpha = 1 / (K * self.lr)

        update_c = OrderedDict()
        diff_c = OrderedDict()
        for name, params in c_diff.items():
            val = alpha * mdl_diff[name]
            update_c[name] = val - c_diff[name]
            diff_c[name] = val - c[name]
        return update_c, diff_c

    def replace_mdl(self, server_mdl):
        self.mdl = copy.copy(server_mdl)


class Server:
    def __init__(self, config, device):
        self.lr = config['lr']
        self.total_clients = config['num_clients']
        self.mdl = load_model(config['model'], config['n_class'], channel=config['channel']).to(device)
        self.c = {name: torch.zeros_like(params) for name, params in self.mdl.named_parameters()}

    @torch.no_grad()
    def aggregate_models(self, clients_model, c_diff):
        update_state = OrderedDict()
        avg_cv = OrderedDict()

        n_clients = len(clients_model)

        for k, client in enumerate(clients_model):
            for key in client.keys():
                if k == 0:
                    update_state[key] = client[key] / n_clients
                else:
                    update_state[key] += client[key] / n_clients

        for k, cv_diff in enumerate(c_diff):
            for name, params in cv_diff.items():
                if k == 0:
                    avg_cv[name] = cv_diff[name]
                else:
                    avg_cv[name] += cv_diff[name]

        for name, params in avg_cv.items():
            self.c[name] += avg_cv[name] / self.total_clients

        for name, params in self.mdl.named_parameters():
            params.copy_(params - self.lr * update_state[name])


class SCAFFOLD:
    def __init__(self, clients_data_loaders, test_loader, config):
        self.device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.n_clients = config['num_clients']
        self.client_iters = config['client_iterations']
        self.total_iters = config['total_iterations']
        self.config = config
        self.test_loader = test_loader
        self.sample_cli = int(config['sample_clients'] * self.n_clients)
        self.clients = [Client(clients_data, config) for clients_data in clients_data_loaders]
        self.server = Server(config, self.device)

    def train(self):
        accs = []
        t_start_time = time.perf_counter()

        for idx in range(self.total_iters):
            i_start_time = time.perf_counter()

            print(f"iteration [{idx + 1}/{self.total_iters}]")
            clients_selected = random.sample([i for i in range(self.n_clients)], self.sample_cli)

            for pdx in clients_selected:
                self.clients[pdx].replace_mdl(self.server.mdl)

            mdl_diffs = []
            diff_cs = []
            for jdx in clients_selected:
                print(f"############## client {jdx} ##############")
                mdl_diff, diff_c = self.clients[jdx].train_client(self.server.mdl, self.server.c, self.client_iters, self.device)
                mdl_diffs.append(mdl_diff)
                diff_cs.append(diff_c)

            print("############## server ##############")
            self.server.aggregate_models(mdl_diffs, diff_cs)

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
