from collections import OrderedDict
from performance_tester import train_during_pruning
from parameters import get_parameters
from train import get_model
import torch
from fusion import MSF, fusion, fusion_old
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
import models as m
import json


def get_cifar_data_loader():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True,
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


def evaluate_performance_simple(input_model, loaders, gpu_id):
    '''
    Computes the accuracy of a given model (input_model) on a given dataset (loaders["test"]).
    '''
    if gpu_id != -1:
        input_model = input_model.cuda(gpu_id)
    input_model.eval()

    accuracy_accumulated = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            if gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()
            
            try:
                test_output, _ = input_model(images)    # TODO: WHY DOES THIS RETURN TWO VALUES?!
            except:
                test_output = input_model(images)

            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            accuracy_accumulated += accuracy 
            total += 1
    input_model.cpu()
    return accuracy_accumulated / total


def get_data_loader():
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )    

    # 2. defining the data loader for train and test set using the downloaded MNIST data
    loaders = {  
        'test'  : torch.utils.data.DataLoader(test_data, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
    }
    return loaders


def get_pretrained_models(model_name, diff_weight_init, gpu_id, num_models):
    models = []

    for idx in range(num_models):
        state = torch.load(f"models/{model_name}_diff_weight_init_{diff_weight_init}_{idx}.pth")
        model = get_model(model_name)
        if "vgg" in model_name:
            new_state_dict = OrderedDict()
            for k, v in state.items():
                print(k)
                name = k
                name = name.replace(".module", "")
                print("new name is: ", name)
                new_state_dict[name] = v
            print(new_state_dict.keys())
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state)
        if gpu_id != -1:
            model = model.cuda(gpu_id)
        models.append(model)
    
    return models

def test(model, loaders, args):
    model.eval()

    accuracy_accumulated = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            if args.gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()
            test_output,_ = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            accuracy_accumulated += accuracy 
            total += 1
    return accuracy_accumulated / total


if __name__ == '__main__':
    args = get_parameters()
    num_models = args.num_models
    dict = {}
    it = 9

    models = get_pretrained_models(args.model_name, args.diff_weight_init, args.gpu_id, args.num_models)


    loaders = None
    if "vgg" not in args.model_name and "resnet" not in args.model_name:
        loaders = get_data_loader()
    else:
        loaders = get_cifar_data_loader()

    
    accuracies = []


    sparsity = 0.1
    accuracies = []
    while sparsity < 1.0:
    
        fused_model = MSF(models[0], gpu_id = args.gpu_id, metric=1, sparsity=sparsity)

        accuracies.append(evaluate_performance_simple(fused_model, loaders, 0))
        sparsity += 0.1
    
    sparsity = 0.1
    idx = 0
    while sparsity < 1.0:
    
        print(f"Sparsity: {sparsity}. Accuracy: {accuracies[idx]}")

        idx += 1
        sparsity += 0.1


    






        


