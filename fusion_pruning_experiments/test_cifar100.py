from collections import OrderedDict
import copy
from train_cifar100 import train_entry
from intrafusion_test import wrapper_intra_fusion
from fusion_utils_IF import MetaPruneType, PruneType
from pruning_modified import prune_structured, prune_structured_intra
from performance_tester import evaluate_performance_simple, get_cifar100_data_loader, train_during_pruning, train_during_pruning_cifar100, update_running_statistics
from parameters import get_parameters
from train import get_model
import torch
from fusion import MSF, IntraFusion_Clustering, fusion, fusion_bn, fusion_old, fusion_sidak_multimodel, fusion_bn_alt, intrafusion_bn
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
from models import get_pretrained_models
import json
import re
import numpy as np
import random
import pprint
import argparse
from vgg_extra import vgg19_bn




if __name__ == '__main__':
    state_dict = torch.load("models/vgg19_bn_diff_weight_init_True_cifar100_0.pth")
    print("sate_dict: ", state_dict.keys())
    print(type(state_dict))
    #model = get_model("vgg19_bn", output_dim=100)
    model = vgg19_bn()
    print("type is: ", type(model))
    print("model: ", model.state_dict().keys())
    model.load_state_dict(state_dict)
    print("type is: ", type(model))

    
    loaders = get_cifar100_data_loader()

    print(evaluate_performance_simple(
            input_model=model, loaders=loaders, gpu_id=0, prune=False))

    model, _ = wrapper_intra_fusion(model=model, model_name="vgg19", resnet=False, sparsity=0.3, prune_iter_steps=3, num_epochs=10, loaders=loaders, prune_type=PruneType.L1, meta_prune_type=MetaPruneType.IF, gpu_id=0)
    
    train_during_pruning_cifar100(model, loaders=loaders, num_epochs=150, gpu_id =0, prune=False, performed_epochs=0)


