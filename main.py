from data.Sheks_data import get_sheks_dataloaders
from data.data import get_data, get_train_loaders
from plots.plot import save_log
from solvers.baseline import BaseLine
from solvers.fedavg import FedAvg
from solvers.pw import PW
from utils import initialize_model

config = {
    'solver': "fed_avg",  # ['fed_avg', 'pw']
    'model': "LSTM",  # ['basic', '2NN', 'LSTM']

    'dataset': "Sheks",  # ["mnist", "Sheks", "cifar-10"]
    'batch_size': 10,

    'client_type': "n-iid",  # ["iid", "n-iid"]
    'num_clients': 100,
    'blk_size': 80,

    'sample_clients': .01,   # C
    'client_iterations': 2,  # epochs E

    'lr': 0.001,
    'beta': 0.4,
    'alpha': 0.1,
    'lambda': 0.1,
}

initialize_model()


if config["dataset"] == "Sheks":
    train_loaders, test_loaders = get_sheks_dataloaders(config)
else:
    train_dataset, train_loader, test_loader = get_data(config)
    test_loaders = [test_loader]
    train_loaders = get_train_loaders(config, train_dataset)

if __name__ == "__main__":
    fed_solver = None

    if config['solver'] == "fed_avg":
        fed_solver = FedAvg
    elif config['solver'] == "pw":
        fed_solver = PW
    # elif config['solver'] == "fed_prox":
    #     fed_solver = FedProx
    # elif config['solver'] == "fed_opt":
    #     fed_solver = FedOPT
    # elif config['solver'] == "cwt":
    #     fed_solver = CWT
    # elif config['solver'] == "scaffold":
    #     fed_solver = SCAFFOLD
    # elif config['solver'] == "ditto":
    #     fed_solver = DITTO
    elif config['solver'] == "baseline":
        acc_s, loss_s = BaseLine(train_loader, test_loader, config).train()
        save_log(config['solver'], acc_s, "acc")
        save_log(config['solver'], loss_s, "loss")
        exit()
    else:
        print("wrong algorithm selected!")

    acc_s, loss_s, idx = fed_solver(train_loaders, test_loaders, config).train()
    save_log(config['solver'], acc_s, "acc")
    save_log(config['solver'], loss_s, "loss")
