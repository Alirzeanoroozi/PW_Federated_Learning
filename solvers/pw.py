import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import copy
from collections import OrderedDict

from utils import evaluate_server, load_model, print_time, get_latest_model


class Client:
    def __init__(self, train_loader, config):
        self.lr = config['lr']
        self.train_loader = train_loader
        self.mdl = load_model(config)
        self.n_epochs = config['client_iterations']

    def train_client(self, device):
        self.mdl = self.mdl.to(device)
        self.opt = optim.Adam(self.mdl.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

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

                loss = self.loss_fn(scores, target)
                cur_loss += loss.item() / len_train

                _, predicted = torch.max(scores, dim=1)
                correct = (predicted == target).sum()
                samples = scores.shape[0]
                cur_acc += correct / (samples * len_train)

                loss.backward()
                self.opt.step()

            print(f"epochs: [{epochs + 1}/{self.n_epochs}] train_acc: {cur_acc:.3f} train_loss: {cur_loss:.3f}")

        return self.opt.state

    def replace_mdl(self, server_mdl):
        self.mdl = copy.deepcopy(server_mdl)


class Server:
    def __init__(self, config):
        self.mdl = load_model(config)

    def aggregate_models(self, clients_model, training_samples, optimizers):
        eps = 1e-20
        update_state = OrderedDict()

        # sums = {}

        # for k, optimizer in enumerate(optimizers):
        #     i = 0
        #     for _, key in enumerate(self.mdl.state_dict().keys()):
        #         if key.endswith("weight") or key.endswith("bias"):
        #             if k == 0:
        #                 sums[i] = optimizer[list(optimizer.keys())[i]]['exp_avg_sq'].cpu()
        #             else:
        #                 sums[i] += optimizer[list(optimizer.keys())[i]]['exp_avg_sq'].cpu() + eps
        #             i += 1

        # for k, client in enumerate(clients_model):
        #     local_state = client.state_dict()
        #     i = 0
        #     for _, key in enumerate(self.mdl.state_dict().keys()):
        #         if key.endswith("weight") or key.endswith("bias"):
        #             if k == 0:
        #                 update_state[key] = np.multiply(local_state[key].cpu(), optimizers[k][list(optimizers[k].keys())[i]]['exp_avg_sq'].cpu()) / sums[i]
        #             else:
        #                 update_state[key] += np.multiply(local_state[key].cpu(), optimizers[k][list(optimizers[k].keys())[i]]['exp_avg_sq'].cpu()) / sums[i]
        #             i += 1
        #         else:
        #             print("\n* Warning: A layer with weights was detected. Should we reset it?\n")
        #             update_state[key] = local_state[key].cpu()

        # JR: This code works only for the Basic model
        sum_v = OrderedDict()
        sum_weighted_v = OrderedDict()

        # for i, key in enumerate(self.mdl.state_dict().keys()):
        for c_ix, c_mdl in enumerate(clients_model):
            i = 0
            # Access the second momentum ('v') from the optimizer's state
            for layer_ix, layer_w in enumerate(c_mdl.state_dict()):
                if layer_w.endswith("weight") or layer_w.endswith("bias"):
                    m = training_samples[c_ix]
                    w = c_mdl.state_dict()[layer_w]
                    v = optimizers[c_ix][list(optimizers[c_ix].keys())[i]]['exp_avg_sq'] + eps
                    # print("key: ", key, v.shape, w.shape)

                    if layer_w not in sum_v.keys():
                        sum_v[layer_w] = v * m
                        sum_weighted_v[layer_w] = v * m * w
                    else:
                        sum_v[layer_w] += v * m
                        sum_weighted_v[layer_w] += v * m * w
                    i += 1
                else:
                    # print("\n* Warning: There are no weights nor estimated variance for layer '{0}'. FedAvg for this layer?\n".format(layer_w))
                    # import ipdb; ipdb.set_trace()
                    # sum_v[layer_w] = self.mdl.state_dict()[layer_w]
                    # sum_weighted_v[layer_w] = self.mdl.state_dict()[layer_w]
                    # update_state[layer_w] = self.mdl.state_dict()[layer_w]
                    update_state[layer_w] = c_mdl.state_dict()[layer_w]

        for key in sum_v.keys():
            update_state[key] = sum_weighted_v[key] / sum_v[key]

        self.mdl.load_state_dict(update_state)


class PW:
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

            optimizers = []
            for i, pdx in enumerate(clients_selected):
                print(f"############## client {pdx} - {i + 1} / {len(clients_selected)} ##############")
                self.clients[pdx].replace_mdl(self.server.mdl)
                optimizers.append(self.clients[pdx].train_client(self.device))

            print("############## server ##############")
            self.server.aggregate_models([self.clients[i].mdl for i in clients_selected],
                                         [len(self.clients[i].train_loader.dataset) for i in clients_selected],
                                         optimizers)

            acc, loss = evaluate_server(self.server.mdl, self.test_loaders, self.device)
            accs.append(acc)
            loss_s.append(loss)

            i_end_time = time.perf_counter()
            print_time(i_end_time, i_start_time)

            torch.save(self.server.mdl, f"pre_computed_models/{self.config['dataset']}/{self.config['model']}/{self.config['solver']}/{idx}.pkl")
            if acc > .54:
                break
            idx += 1

        t_end_time = time.perf_counter()
        print_time(t_end_time, t_start_time)

        return accs, loss_s, idx
