import json
import os
import torch
from torch.utils.data import DataLoader
from data.data import NextCharDataset


def get_sheks_dataloaders(config):
    train_loaders = []
    test_loaders = []

    if os.path.isdir("data/train_dataloaders") and os.path.isdir("data/test_dataloaders"):
        for i in range(1129):
            # print(i, "loaded")
            train_loaders.append(torch.load(f"data/train_dataloaders/{i}"))
            test_loaders.append(torch.load(f"data/test_dataloaders/{i}"))
        return train_loaders, test_loaders

    with open('/home/alireza/PycharmProjects/PW_Federated_Learning/data/shakespeare/data/all_data/all_data.json') as json_file:
        Data = json.load(json_file)

    datasets = [Data['user_data'][user]['raw'] for user in Data['users']]

    vocab = [
        '<UNK>', '\n', ' ', '!', '$', '&', "'", ',', '-', '.', ':', ';', '?', '[', ']', '(', ')', '"', '{', '}', '<', '>',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ]
    vocab_size = len(vocab)
    stoi = {ch: i + 1 for i, ch in enumerate(vocab)}
    stoi.setdefault(0)
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    if config['client_type'] == "iid":
        pass
    elif config['client_type'] == "n-iid":
        for i, dataset in enumerate(datasets):
            n = int(0.8 * len(dataset))
            train_data = dataset[:n]
            test_data = dataset[n:]

            X_train = [train_data[i: i + config['blk_size']] for i in range(len(train_data) - config['blk_size'])]
            y_train = [train_data[i + 1: i + config['blk_size'] + 1] for i in range(len(train_data) - config['blk_size'])]
            X_test = [test_data[i: i + config['blk_size']] for i in range(len(test_data) - config['blk_size'])]
            y_test = [test_data[i + 1: i + config['blk_size'] + 1] for i in range(len(test_data) - config['blk_size'])]

            X_train = torch.tensor([encode(text) for text in X_train])
            X_test = torch.tensor([encode(text) for text in X_test])
            y_train = torch.tensor([encode(text) for text in y_train])
            y_test = torch.tensor([encode(text) for text in y_test])

            train_Dataset = NextCharDataset(X_train, y_train)
            test_Dataset = NextCharDataset(X_test, y_test)

            train_loaders.append(DataLoader(train_Dataset, batch_size=config['batch_size']))
            test_loaders.append(DataLoader(test_Dataset, batch_size=config['batch_size']))

    if not os.path.isdir("data/train_dataloaders"):
        os.mkdir("data/train_dataloaders")
    if not os.path.isdir("data/test_dataloaders"):
        os.mkdir("data/test_dataloaders")

    for i, train_loader in enumerate(train_loaders):
        print(i, "saved")
        torch.save(train_loader, f"data/train_dataloaders/{i}")
    for i, test_loader in enumerate(test_loaders):
        print(i, "saved")
        torch.save(test_loader, f"data/test_dataloaders/{i}")

    return train_loaders, test_loaders
