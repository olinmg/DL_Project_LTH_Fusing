from collections import OrderedDict
import copy
from train_2 import get_pretrained_model
from intrafusion_test import wrapper_intra_fusion
from fusion_utils_IF import MetaPruneType, PruneType
from pruning_modified import prune_structured, prune_structured_intra
from performance_tester import train_during_pruning, update_running_statistics
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


def evaluate_performance_simple(input_model, loaders, gpu_id, eval=True):
    '''
    Computes the accuracy of a given model (input_model) on a given dataset (loaders["test"]).
    '''
    if gpu_id != -1:
        input_model = input_model.cuda(gpu_id)
    
    if eval:
        input_model.eval()

    accuracy_accumulated = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            if gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()
            
            test_output = input_model(images)

            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            accuracy_accumulated += accuracy 
            total += 1
    input_model.cpu()
    return accuracy_accumulated / total


def get_data_loader(shuffle=True):
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )    

    train_data = datasets.MNIST(
        root = 'data', 
        train = True, 
        transform = ToTensor()
    ) 

    # 2. defining the data loader for train and test set using the downloaded MNIST data
    loaders = {  
        'test'  : torch.utils.data.DataLoader(test_data, 
                                            batch_size=100, 
                                            shuffle=shuffle, 
                                            num_workers=1),
        "train": torch.utils.data.DataLoader(train_data, 
                                            batch_size=100, 
                                            shuffle=False, 
                                            num_workers=1)
    }
    return loaders

def get_cifar_data_loader(shuffle=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=shuffle,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)
    
    return {
        "train" : train_loader,
        "test" : val_loader
    }

def get_cifar100_data_loader():
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return {"train": train_loader, "test": val_loader}

if __name__ == '__main__':
    args = get_parameters()
    num_models = args.num_models
    it = 9

    config = dict(
        dataset="Cifar10",
        model="vgg11",
        optimizer="SGD",
        optimizer_decay_at_epochs=[150, 250],
        optimizer_decay_with_factor=10.0,
        optimizer_learning_rate=0.1,
        optimizer_momentum=0.9,
        optimizer_weight_decay=0.0001,
        batch_size=256,
        num_epochs=300,
        seed=42,
    )
    dict = {}

    file_name = "./vgg11_cifar_10.checkpoint"

    #model = get_pretrained_models(args.model_name, f"resnet18_diff_weight_init_False_cifar10_eps300_D", args.gpu_id, num_models, output_dim=10)[0]
    model,_ = get_pretrained_model(config, file_name, device_id=0)
    models = [model]

    loaders = None
    """if "vgg" not in args.model_name and "resnet" not in args.model_name:
        print("Went in here!!!")
        loaders = get_data_loader()
    else:
        print("Got cifar")
        loaders = get_cifar_data_loader()"""
    
    loaders = get_cifar_data_loader()
    results = {}

    train_epochs = 10
    sparsities = [0.4, 0.5, 0.6, 0.7]
    seeds = [1]
    meta_prune_types = [MetaPruneType.DEFAULT, MetaPruneType.IF]
    total_steps = 3
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        results[seed] = {}
        result = results[seed]
        for idx, model in enumerate(models):
            result[f"model_{idx}"] = {}
            for sparsity in sparsities:
                if sparsity in [0.4, 0.5]:
                    total_steps = 3
                elif sparsity in [0.6, 0.7, 0.8]:
                    total_steps = 4
                else:
                    total_steps = 5
                result[f"model_{idx}"][sparsity] = {}
                for meta_prune_type in meta_prune_types:
                        print("****************Sparsity: ", sparsity)

                        """fused_model_g, epoch_accuracies_interm = wrapper_intra_fusion(model=model, model_name=args.model_name, resnet=False, sparsity=sparsity, prune_iter_steps=1, num_epochs=0, loaders=loaders, prune_type=PruneType.L1, meta_prune_type=meta_prune_type, gpu_id=0)
                        accuracy_fused_g = evaluate_performance_simple(fused_model_g, loaders, 0, eval=True)
                        continue"""

                        fused_model_g, epoch_accuracies_interm = wrapper_intra_fusion(model=model, model_name=args.model_name, resnet=False, sparsity=sparsity, prune_iter_steps=total_steps, num_epochs=train_epochs, loaders=loaders, prune_type=PruneType.L1, meta_prune_type=meta_prune_type, gpu_id=0)
                        accuracy_fused_g = evaluate_performance_simple(fused_model_g, loaders, 0, eval=True)
                        print("fused: ", accuracy_fused_g)
                        for ((name_orig, module_orig), (name, module)) in list(zip(model.named_modules(), fused_model_g.named_modules())):
                            if isinstance(module_orig, (torch.nn.Conv2d, torch.nn.Linear)):
                                print(f"{module_orig.weight.shape} -> {module.weight.shape}")
                        fused_model_g, epoch_accuracies = train_during_pruning(fused_model_g, loaders=loaders, num_epochs=150-(total_steps*train_epochs), gpu_id =0, prune=False, performed_epochs=0)
                        torch.save(fused_model_g, f"./vgg11_cifar10_pruned_{meta_prune_type}_{sparsity}.pth")
                        print("Final fused is: ", epoch_accuracies[-1])
                        epoch_accuracies_interm.extend(epoch_accuracies)
                        result[f"model_{idx}"][sparsity][meta_prune_type] = epoch_accuracies_interm

                        pprint.pprint(results)
    with open("results_vgg11_cifar10_results_for_olin.json", "w") as outfile:
        json.dump(results, outfile, indent=4)



