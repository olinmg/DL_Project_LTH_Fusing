from collections import OrderedDict
from performance_tester import train_during_pruning
from parameters import get_parameters
from train import get_model
import torch
from fusion import fusion, fusion_old
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


def get_pretrained_models(model_name, diff_weight_init, gpu_id, num_models, sparsity=[1.0, 1.0]):
    models = []

    for idx in range(num_models):
        state = torch.load(f"models/vgg_alternative/{model_name}_diff_weight_init_{diff_weight_init}_{int(sparsity[idx]*10)}.pth")
        model = get_model(model_name, sparsity=sparsity[idx]/10)
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

    while True:
        models = get_pretrained_models(args.model_name, args.diff_weight_init, args.gpu_id, args.num_models, [1, 2])

        """
        model0 = m.ResNet18(linear_bias=False, use_batchnorm=False)
        for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
                enumerate(zip(model0.named_parameters(), model0.named_parameters())):
            print(f"{layer0_name} : {fc_layer0_weight.shape}")
        checkpoint = torch.load("models/resnet_models/model_0/best.checkpoint", map_location = "cpu")
        model0.load_state_dict(checkpoint['model_state_dict'])

        model1 = m.ResNet18(linear_bias=False, use_batchnorm=False)
        checkpoint = torch.load("models/resnet_models/model_1/best.checkpoint", map_location = "cpu")
        model1.load_state_dict(checkpoint['model_state_dict'])
        models=[model0, model1]"""
        loaders = None
        if "vgg" not in args.model_name and "resnet" not in args.model_name:
            loaders = get_data_loader()
        else:
            loaders = get_cifar_data_loader()
        
        accuracies_orig = []
        for model in models:
            accuracies_orig.append(evaluate_performance_simple(model, loaders, 0))


        imp = 0.0
        accuracies = []
        #fused_model = fusion(models, args, accuracies=accuracies_orig)

        fused_model = fusion(models, args)
        fused_model,_ = train_during_pruning(fused_model, loaders, 20, 0, prune=False)
        accuracy_fused = evaluate_performance_simple(fused_model, loaders, 0)
        accuracies.append((accuracy_fused, imp))

        print("---- Model 0 Parameters ------")
        for idx, (layer0_name, fc_layer0_weight) in \
                enumerate(models[0].named_parameters()):
            print(f"{layer0_name} : {fc_layer0_weight.shape}")
        
        print("---- Model 1 Parameters ------")
        for idx, (layer0_name, fc_layer0_weight) in \
                enumerate(models[1].named_parameters()):
            print(f"{layer0_name} : {fc_layer0_weight.shape}")
        
        print("---- Model Fused Parameters ------")
        for idx, (layer0_name, fc_layer0_weight) in \
                enumerate(fused_model.named_parameters()):
            print(f"{layer0_name} : {fc_layer0_weight.shape}")

        
        for accuracy in accuracies:
            print('Test Accuracy of the model fused: %.2f' % accuracy[0])
            print(accuracy[1])
            print("-----------------------")
        
        for accuracy in accuracies_orig:
            print('Test Accuracy of the model original: %.2f' % accuracy)
        
        dict[str(it/10)] = {
            "accuracy_original": accuracies_orig[0],
            "accuracy_fused": accuracy_fused
        }
        print(dict)

        break


    print(dict)
    with open("results_fusion_alternative_whatever.json", "w") as outfile:
        json.dump(dict, outfile, indent=4)




        


