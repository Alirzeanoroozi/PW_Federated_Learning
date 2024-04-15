import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import copy
from collections import OrderedDict

from utils import evaluate_server, load_model


class Client:
    def __init__(self, mdl_name, train_loader, config):
        self.lr = config['lr']
        self.mom = config['momentum']
        self.train_loader = train_loader
        self.mdl = load_model(mdl_name, config['n_class'], channel=config['channel'])
        self.loss_fn = nn.CrossEntropyLoss()
        self.serv_mdl = load_model(mdl_name, config['n_class'], channel=config['channel'])
        self.cpu = torch.device('cpu')
        self.alpha = config['alpha']

    def train_client(self, n_epochs, device):
        self.mdl = self.mdl.to(device)
        self.serv_mdl = self.serv_mdl.to(device)
        self.opt = optim.SGD(self.mdl.parameters(), lr=self.lr, momentum=self.mom)
        t_val = {'train_acc': [], "train_loss": []}

        self.mdl.train()
        len_train = len(self.train_loader)

        for epochs in range(n_epochs):
            cur_loss = 0
            cur_acc = 0
            for idx, (data, target) in enumerate(self.train_loader):
                data = data.to(device)
                target = target.to(device)

                scores = self.mdl(data)
                loss = self.loss_fn(scores, target) + self.prox_reg()
                cur_loss += loss.item() / len_train

                scores = F.softmax(scores, dim=1)
                _, predicted = torch.max(scores, dim=1)
                correct = (predicted == target).sum()
                samples = scores.shape[0]
                cur_acc += correct / (samples * len_train)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            t_val['train_acc'].append(float(cur_acc))
            t_val['train_loss'].append(float(cur_loss))

            print(f"epochs: [{epochs + 1}/{n_epochs}] train_acc: {cur_acc:.3f} train_loss: {cur_loss:.3f}")

        self.mdl = copy.deepcopy(self.mdl)
        return t_val

    def replace_mdl(self, server_mdl):
        self.mdl = copy.deepcopy(server_mdl)
        self.serv_mdl = copy.deepcopy(server_mdl)
        
    @torch.no_grad()
    def prox_reg(self):
        params1 = dict(self.mdl.named_parameters())
        params2 = dict(self.serv_mdl.named_parameters())
        
        loss_val = 0
        
        for name, params in params1.items():
            norm_val = torch.norm(params1[name] - params2[name]) ** 2
            loss_val += self.alpha * 0.5 * norm_val
            
        return loss_val


def diff_mdl(server_model, client_model):
    # x - y_i
    mdl_diff = OrderedDict()
    server_parms = dict(server_model.named_parameters())
    for name, params in client_model.named_parameters():
        mdl_diff[name] = params - server_parms[name]
    return mdl_diff


class Server:
    def __init__(self, config):
        self.mdl = load_model(config['model'], config['n_class'], channel=config['channel'])

    def aggregate_models(self, server_model, clients_model):
        delta_t = OrderedDict()
        n_clients = len(clients_model)
        for k, client in enumerate(clients_model):
            local_state = diff_mdl(server_model, client)
            for key in self.mdl.state_dict().keys():
                if k == 0:
                    delta_t[key] = local_state[key] / n_clients
                else:
                    delta_t[key] += local_state[key] / n_clients

        print(self.mdl.load_state_dict(update_state))


class FedOPT:
    def __init__(self, clients_data_loaders, test_loader, config):
        self.device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.n_clients = config['num_clients']
        self.client_iters = config['client_iterations']
        self.total_iters = config['total_iterations']
        self.test_loader = test_loader
        self.sample_cli = int(config['sample_clients'] * self.n_clients)
        self.clients = [Client(config['model'], clients_data, config) for clients_data in clients_data_loaders]
        self.server = Server(config)

    def train(self):
        accs = []
        start_time = time.perf_counter()

        for idx in range(self.total_iters):
            print(f"iteration [{idx + 1}/{self.total_iters}]")
            clients_selected = random.sample([i for i in range(self.n_clients)], self.sample_cli)

            for jdx in clients_selected:
                print(f"############## client {jdx} ##############")
                self.clients[jdx].train_client(self.client_iters, self.device)

            print("############## server ##############")
            self.server.aggregate_models(self.server.mdl, [self.clients[i].mdl for i in clients_selected])

            single_acc = evaluate_server(self.server.mdl, self.test_loader, self.device)
            print(f'cur_acc: {single_acc["acc"]:.3f}')
            accs.append(single_acc["acc"].item())

            for pdx in clients_selected:
                self.clients[pdx].replace_mdl(self.server.mdl)

        end_time = time.perf_counter()
        elapsed_time = int(end_time - start_time)
        hr = elapsed_time // 3600
        mi = (elapsed_time - hr * 3600) // 60
        print(f"training done in {hr} H {mi} M")
        return accs
