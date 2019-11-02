import torch


def getLossFuns():
    criterion = torch.nn.CrossEntropyLoss()
    return criterion
