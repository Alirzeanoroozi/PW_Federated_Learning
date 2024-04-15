from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import random_split, DataLoader, Subset, Dataset
import torchvision.transforms as transforms


class NextCharDataset(Dataset):
    def __init__(self, X, Y):
        self.__X = X
        self.__Y = Y

    def __len__(self):
        return self.__X.shape[0]

    def __getitem__(self, idx):
        return self.__X[idx], self.__Y[idx]


def get_data(config):
    if config['dataset'] == "mnist":
        train_dataset = MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    elif config['dataset'] == "cifar-10":
        train_dataset = CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError(f"dataset {config['dataset']} not known")

    test_loader = DataLoader(test_dataset, config['batch_size'])
    train_loader = DataLoader(train_dataset, config['batch_size'])

    return train_dataset, train_loader, test_loader


def get_train_loaders(config, dataset):
    clients_type = config['client_type']
    num_clients = config['num_clients']

    data_loaders = []
    if clients_type == "iid":
        total_train_size = len(dataset)
        examples_per_client = total_train_size // num_clients
        client_datasets = random_split(dataset, [min(i + examples_per_client, total_train_size) - i for i in
                                                 range(0, total_train_size, examples_per_client)])
        data_loaders = [DataLoader(client_datasets[i], config['batch_size'], shuffle=False) for i in range(num_clients)]
    elif clients_type == "n-iid":
        index_lists = []
        # Read indexes from the file

        for i in range(num_clients):
            with open(f"data/splits/client{i+1}.txt", 'r') as file:
                index_lists.append([int(line.strip().split("\t")[0]) for line in file.readlines()[1:]])
        
        # Create a DataLoader for each subset
        for i, indexes in enumerate(index_lists):
            subset = Subset(dataset, indexes)
            data_loader = DataLoader(subset, batch_size=config['batch_size'], shuffle=True)
            data_loaders.append(data_loader)
        
        # r = [random.randint(1, num_clients) for _ in range(num_clients - 1)]
        # new_r = [i * len(dataset) // sum(r) for i in r]
        # r = new_r + [len(dataset) - sum(new_r)]
        # client_datasets = random_split(dataset, r)

    else:
        raise ValueError(f"Clients variant {clients_type} not known")

    return data_loaders
