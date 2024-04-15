import json
import os
import torch
from sklearn.model_selection import train_test_split
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

    with open('data/Sheks/all_data.json') as json_file:
        Data = json.load(json_file)

    datasets = [Data['user_data'][user] for user in Data['users']]

    vocab = [
        '<UNK>', '\n', ' ', '!', '$', '&', "'", ',', '-', '.', ':', ';', '?', '[', ']', '(', ')', '"', '{', '}', '<', '>',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ]
    stoi = {ch: i + 1 for i, ch in enumerate(vocab)}
    stoi.setdefault(0)
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    if config['client_type'] == "iid":
        pass
    elif config['client_type'] == "n-iid":
        for i, dataset in enumerate(datasets):
            if len(dataset['x']) > 0:
                print(i, "is Starting")
                X_train, X_test, y_train, y_test = train_test_split(dataset['x'], dataset['y'], train_size=0.8, shuffle=False)
                X_train = torch.tensor([encode(text) for text in X_train])
                X_test = torch.tensor([encode(text) for text in X_test])
                y_train = torch.tensor([encode(text)[0] for text in y_train])
                y_test = torch.tensor([encode(text)[0] for text in y_test])

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


def get_sheks_baseline():
    block_size = 80
    batch_size = 10
    f = open("demofile.txt", "r")
    sheks_data = f.read()

    vocab = [
        '<UNK>', '\n', ' ', '!', '$', '&', "'", ',', '-', '.', ':', ';', '?', '[', ']', '(', ')', '"', '{', '}', '<', '>',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ]
    stoi = {ch: i + 1 for i, ch in enumerate(vocab)}
    stoi.setdefault(0)
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    train, test = train_test_split(sheks_data, train_size=0.8, shuffle=False)

    ix = torch.randint(len(train) - block_size, (batch_size,))
    x = torch.stack([train[i:i+block_size] for i in ix])
    y = torch.stack([train[i+block_size+1] for i in ix])


    train_Dataset = NextCharDataset(X_train, y_train)
    test_Dataset = NextCharDataset(X_test, y_test)

    train_loaders.append(DataLoader(train_Dataset, batch_size=config['batch_size']))
    test_loaders.append(DataLoader(test_Dataset, batch_size=config['batch_size']))
