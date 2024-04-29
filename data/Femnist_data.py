import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from data.data import OwnDataset


def get_femnist_dataloaders(config):
    train_loaders = []
    test_loaders = []

    if os.path.isdir("data/pre_computed_dataloaders/Femnist/train_dataloaders") and os.path.isdir(
            "data/pre_computed_dataloaders/Femnist/test_dataloaders"):
        for i in range(3596):
            print(i, "loaded")
            train_loaders.append(torch.load(f"data/pre_computed_dataloaders/Femnist/train_dataloaders/{i}"))
            test_loaders.append(torch.load(f"data/pre_computed_dataloaders/Femnist/test_dataloaders/{i}"))
        return train_loaders, test_loaders

    if config['client_type'] == "iid":
        pass
    elif config['client_type'] == "n-iid":
        for train_json_file_name, test_json_file_name in zip(sorted(os.listdir("data/femnist/data/train")),
                                                             sorted(os.listdir("data/femnist/data/test"))):
            print(train_json_file_name)
            with open('data/femnist/data/train/' + train_json_file_name) as json_file:
                train_data = json.load(json_file)
            with open('data/femnist/data/test/' + test_json_file_name) as json_file:
                test_data = json.load(json_file)

            for u in train_data['users']:
                X_train = torch.tensor(np.array([np.array(img).reshape(28, 28) for img in train_data['user_data'][u]['x']], dtype=np.float32))
                X_test = torch.tensor(np.array([np.array(img).reshape(28, 28) for img in test_data['user_data'][u]['x']], dtype=np.float32))
                y_train = torch.tensor(train_data['user_data'][u]['y'])
                y_test = torch.tensor(test_data['user_data'][u]['y'])

                train_Dataset = OwnDataset(X_train, y_train)
                test_Dataset = OwnDataset(X_test, y_test)

                train_loaders.append(DataLoader(train_Dataset, batch_size=config['batch_size']))
                test_loaders.append(DataLoader(test_Dataset, batch_size=config['batch_size']))

    if not os.path.isdir("data/pre_computed_dataloaders/Femnist/train_dataloaders"):
        os.mkdir("data/pre_computed_dataloaders/Femnist/train_dataloaders")
    if not os.path.isdir("data/pre_computed_dataloaders/Femnist/test_dataloaders"):
        os.mkdir("data/pre_computed_dataloaders/Femnist/test_dataloaders")

    for i, train_loader in enumerate(train_loaders):
        print(i, "saved")
        torch.save(train_loader, f"data/pre_computed_dataloaders/Femnist/train_dataloaders/{i}")
    for i, test_loader in enumerate(test_loaders):
        print(i, "saved")
        torch.save(test_loader, f"data/pre_computed_dataloaders/Femnist/test_dataloaders/{i}")

    return train_loaders, test_loaders
