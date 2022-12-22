import torch
import numpy as np

from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from base_convNN import CNN, train, test, loaders
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_FILE_NAME = "./base_cnn_model_dict.pth"

model = CNN()
model.load_state_dict(torch.load(MODEL_FILE_NAME))
model.eval()

def prune_unstructured(net, prune_type, amount=0.2):
    parameters_to_prune = []
    for _q, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))

    if prune_type == 'random':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=amount,
        )
    elif prune_type == 'l1':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    elif prune_type == 'l2':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.LnStructured,
            amount=amount,
        )
    elif prune_type == 'l_inf':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.LnStructured,
            amount=amount,
        )
    else:
        raise ValueError("Prune type not supported")


def prune_structured(net, prune_type, amount=0.2):
    parameters_to_prune = []
    for _, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))

    if prune_type == 'random':
        prune.global_structured(
            parameters_to_prune,
            pruning_method=prune.RandomStructured,
            amount=amount,
        )
    elif prune_type == 'l1':
        prune.global_structured(
            parameters_to_prune,
            pruning_method=prune.L1Structured,
            amount=amount,
        )
    elif prune_type == 'l2':
        prune.global_structured(
            parameters_to_prune,
            pruning_method=prune.LnStructured,
            amount=amount,
        )
    elif prune_type == 'l_inf':
        prune.global_structured(
            parameters_to_prune,
            pruning_method=prune.LnStructured,
            amount=amount,
        )
    else:
        raise ValueError("Prune type not supported")
