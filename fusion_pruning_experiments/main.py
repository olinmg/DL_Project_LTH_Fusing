from collections import OrderedDict
from performance_tester import train_during_pruning, update_running_statistics
from parameters import get_parameters
from train import get_model
import torch
from fusion import MSF, IntraFusion_Clustering, fusion, fusion_bn, fusion_old, fusion_sidak_multimodel
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


def get_data_loader():
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
                                            shuffle=True, 
                                            num_workers=1),
        "train": torch.utils.data.DataLoader(train_data, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1)
    }
    return loaders


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

    models = get_pretrained_models(args.model_name, "vgg11_diff_weight_init_True_cifar10", args.gpu_id, num_models, output_dim=10)

    loaders = None
    if "vgg" not in args.model_name and "resnet" not in args.model_name:
        print("Went in here!!!")
        loaders = get_data_loader()
    else:
        print("Got cifar")
        loaders = get_cifar_data_loader()

    
    accuracies = []

    
    """idx = -1
    for a in models[0].named_modules():
        idx += 1
        if idx < 2:
            continue
        print(type(a))
        print(a)
        print(type(a[1]))
        try:
            print(a[1].weight)
        except:
            print("No weights")
        print("-----")"""
        
    
    pass

    model = models[0]
    model.eval()
    fused_model= IntraFusion_Clustering(model, gpu_id = args.gpu_id, resnet = False, sparsity=0.8)

    accuracy_0 = evaluate_performance_simple(models[0], loaders, 0)
    accuracy_1 = evaluate_performance_simple(models[1], loaders, 0)
    accuracy_fused = evaluate_performance_simple(fused_model, loaders, 0, eval=True)

    print('Test Accuracy of the model 0: %.2f' % accuracy_0)
    print('Test Accuracy of the model 1: %.2f' % accuracy_1)
    print('Test Accuracy of the model fused beginning: %.2f' % accuracy_fused)

    #fused_model = update_running_statistics(fused_model, loaders, 0)
    fused_model.train()
    fused_model, _ = train_during_pruning(fused_model, loaders=loaders, num_epochs=40, gpu_id =0, prune=False, performed_epochs=0)
    accuracy_fused = evaluate_performance_simple(fused_model, loaders, 0, eval=True)
    print('Test Accuracy of the model fused after: %.2f' % accuracy_fused)





    






        


