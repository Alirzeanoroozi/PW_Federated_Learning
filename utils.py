import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os


def initialize_model():
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_latest_model(config):
    folder_path = f"pre_computed_models/{config['dataset']}/{config['model']}/{config['solver']}"
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
        return 0, None
    elif len(os.listdir(folder_path)) == 0:
        return 0, None
    else:
        files = sorted(os.listdir(folder_path))
        return int(files[-1].split(".")[0]), torch.load(os.path.join(folder_path, files[-1]))


def print_time(end_time, start_time):
    elapsed_time = int(end_time - start_time)
    hr = elapsed_time // 3600
    mi = (elapsed_time - hr * 3600) // 60
    sec = elapsed_time - hr * 3600 - mi * 60
    print(f"training done in {hr} H {mi} M {sec} S")


def evaluate_server(model, loaders, device, config):
    model = model.to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    total_acc = 0
    total_loss = 0

    for loader in loaders:
        len_train = len(loader)

        cur_acc = 0
        cur_loss = 0

        with torch.no_grad():
            for idx, (data, target) in enumerate(loader):
                data = data.to(device)
                target = target.to(device)

                scores = model(data)

                if config['dataset'] == "Sheks":
                    B, T, C = scores.shape
                    logits = scores.reshape(B * T, C)
                    targets = target.view(B * T)

                    cur_loss += loss_fn(logits, targets) / (samples * len_train)


                    scores = scores[:, -1, :]  # becomes (B, C)
                    # apply softmax to get probabilities
                    probs = F.softmax(scores, dim=-1)  # (B, C)
                    _, predicted = torch.max(probs, dim=1)
                    target = target[:, -1]
                else:
                    loss = loss_fn(scores, target)
                    cur_loss += loss.item() / len_train

                    _, predicted = torch.max(scores, dim=1)

                correct = (predicted == target).sum()
                samples = scores.shape[0]
                cur_acc += correct / (samples * len_train)

        total_acc += cur_acc / len(loaders)
        total_loss += cur_loss / len(loaders)

    print(f"test_acc: {total_acc:.3f}, test_loss: {total_loss:.3f}")

    return total_acc, total_loss
