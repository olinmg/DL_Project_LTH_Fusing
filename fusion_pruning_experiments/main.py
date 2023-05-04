from collections import OrderedDict
import copy
from intrafusion_test import wrapper_intra_fusion
from fusion_utils_IF import MetaPruneType, PruneType
from pruning_modified import prune_structured, prune_structured_intra
from performance_tester import train_during_pruning, update_running_statistics
from parameters import get_parameters
from train import get_model
import torch
from fusion_IF import MSF, IntraFusion_Clustering, fusion, fusion_bn, fusion_old, fusion_sidak_multimodel, fusion_bn_alt, intrafusion_bn
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


def get_cifar_data_loader(shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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
                                            shuffle=shuffle, 
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

    
    """accuracies = []

    result = {}
    sparsities = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    result["prune"] = {}
    result["IF"] = {}
    prune_type = "l1"
    for sparsity in sparsities:
        print("----------")
        t = prune_structured(net=copy.deepcopy(models[0]), loaders=None, prune_iter_epochs=0, gpu_id=args.gpu_id, example_inputs=torch.randn(1, 3, 32, 32),
                    out_features=10, prune_type=prune_type, sparsity=sparsity, train_fct=None, prune_iter_steps=1)
        result["prune"][sparsity] = evaluate_performance_simple(t, loaders, 0, eval=True)
        print(result["prune"][sparsity])

        fused_model_g = wrapper_intra_fusion(model=models[0], model_name = args.model_name, resnet=False, sparsity=sparsity, prune_iter_steps=0, num_epochs=0, loaders=None, prune_type="l1", meta_prune_type=MetaPruneType.IF, gpu_id=0)
        #fused_model_g = intrafusion_bn(models[0], full_model = models[0], meta_prune_type = MetaPruneType.IF, prune_type=prune_type, model_name=args.model_name, sparsity=sparsity, fusion_type="weight", gpu_id = args.gpu_id, resnet = True, train_loader=get_cifar_data_loader(shuffle=True)["train"])
        result["IF"][sparsity] = evaluate_performance_simple(fused_model_g, loaders, 0, eval=True)
        print(result["IF"][sparsity])
        print("--------------")
    with open(f"results_datafree_resnet18_{prune_type}.json", "w") as outfile:
        json.dump(result, outfile, indent=4)
    exit()"""


    """for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(models[0].named_parameters(), models[0].named_parameters())):
        print(f"{layer0_name} : {fc_layer0_weight.shape}")

    fused_model_g = fusion_bn(models, fusion_type="weight", gpu_id=args.gpu_id, resnet=False, train_loader=get_cifar_data_loader(shuffle=True)["train"])
    #fused_model_g = fusion(models, gpu_id=args.gpu_id, resnet=True)
    print(evaluate_performance_simple(fused_model_g, loaders, 0, eval=True))
    exit()"""
    """fused_model_g = intrafusion_bn(models[0], full_model = models[0], sparsity=0.9, fusion_type="weight", gpu_id = args.gpu_id, resnet = True, train_loader=get_cifar_data_loader(shuffle=True)["train"])
    print(evaluate_performance_simple(fused_model_g, loaders, 0, eval=True))
    exit()"""


    result = {}

    train_epochs = 10
    sparsities = [0.9]
    total_steps = 5
    for idx, model in enumerate(models):
        result[f"model_{idx}"] = {}
        for sparsity in sparsities:
            print("****************Sparsity: ", sparsity)
            """prune_steps = prune_structured_intra(net=copy.deepcopy(model), loaders=None, num_epochs=0, gpu_id=args.gpu_id, example_inputs=torch.randn(1, 3, 32, 32),
                    out_features=10, prune_type="l1", sparsity=sparsity, train_fct=None, total_steps=total_steps)
            fused_model_g = model
            for prune_step in prune_steps:
                fused_model_g = intrafusion_bn(fused_model_g, sparsity=sparsity, fusion_type="weight", full_model = model, small_model=prune_step, gpu_id = args.gpu_id, resnet = True, train_loader=get_cifar_data_loader(shuffle=True)["train"])
                fused_model_g,_ = train_during_pruning(fused_model_g, loaders=loaders, num_epochs=train_epochs, gpu_id =0, prune=False, performed_epochs=0)"""
            fused_model_g = wrapper_intra_fusion(model=model, model_name=args.model_name, resnet=False, sparsity=sparsity, prune_iter_steps=total_steps, num_epochs=train_epochs, loaders=loaders, prune_type=PruneType.L2, meta_prune_type=MetaPruneType.IF, gpu_id=0)
            accuracy_fused_g = evaluate_performance_simple(fused_model_g, loaders, 0, eval=True)
            print("fused: ", accuracy_fused_g)
            fused_model_g, epoch_accuracies = train_during_pruning(fused_model_g, loaders=loaders, num_epochs=100, gpu_id =0, prune=False, performed_epochs=0)
            print("Final fused is: ", epoch_accuracies[-1])
            result[f"model_{idx}"][sparsity] = epoch_accuracies
    
    with open("results_intrafusion_resnet18_dataaware_prune_L2_05.json", "w") as outfile:
        json.dump(result, outfile, indent=4)


    exit()
    """sparsities = [0.5, 0.6, 0.7, 0.8]
    result = {}
    for idx, model in enumerate(models):
        result[f"model_{idx}"] = {}
        for sparsity in sparsities:
            fused_model_g = model
            iterations = []
            if sparsity > 0.5:
                if sparsity == 0.6:
                    iterations = [0.2, 0.4]
                if sparsity == 0.7 or sparsity == 0.8:
                    iterations = [0.2, 0.4, 0.6]
                for i in iterations:
                    fused_model_g = intrafusion_bn(fused_model_g, sparsity=i, fusion_type="weight", full_model = model, gpu_id = args.gpu_id, resnet = True, train_loader=get_cifar_data_loader(shuffle=True)["train"])
                    fused_model_g, _ = train_during_pruning(fused_model_g, loaders=loaders, num_epochs=10, gpu_id =0, prune=False, performed_epochs=0)

            fused_model_g = intrafusion_bn(fused_model_g, sparsity=sparsity, fusion_type="weight", full_model = model, gpu_id = args.gpu_id, resnet = True, train_loader=get_cifar_data_loader(shuffle=True)["train"])
            #fused_model_g, _ = train_during_pruning(fused_model_g, loaders=loaders, num_epochs=140, gpu_id =0, prune=False, performed_epochs=0)
            accuracy_fused_g = evaluate_performance_simple(fused_model_g, loaders, 0, eval=True)
            print('Test Accuracy of the model fused beginning gradient: %.2f' % accuracy_fused_g)

            #print('Test Accuracy of the model fused beginning weight: %.2f' % accuracy_fused_w)

            fused_model_g, epoch_accuracies = train_during_pruning(fused_model_g, loaders=loaders, num_epochs=150-10*len(iterations), gpu_id =0, prune=False, performed_epochs=0)

            print('Test Accuracy of the model fused beginning gradient: %.2f' % accuracy_fused_g)
            result[f"model_{idx}"][str(sparsity)] = epoch_accuracies

    with open("results_intrafusion.json", "w") as outfile:
        json.dump(result, outfile, indent=4)
    exit()"""
    #fused_model_w, _ = train_during_pruning(fused_model_w, loaders=loaders, num_epochs=40, gpu_id =0, prune=False, performed_epochs=0)
    fused_model_g, _ = train_during_pruning(fused_model_g, loaders=loaders, num_epochs=40, gpu_id =0, prune=False, performed_epochs=0)

    #accuracy_fused_w = evaluate_performance_simple(fused_model_w, loaders, 0, eval=True)
    accuracy_fused_g = evaluate_performance_simple(fused_model_g, loaders, 0, eval=True)

    #print('Test Accuracy of the model fused beginning weight: %.2f' % accuracy_fused_w)
    print('Test Accuracy of the model fused beginning gradient: %.2f' % accuracy_fused_g)
    """fused_accs = []
    for idx in range(40):
        fused_model, _ = train_during_pruning(fused_model, loaders=loaders, num_epochs=1, gpu_id =0, prune=False, performed_epochs=0)
        fused_accs.append(evaluate_performance_simple(fused_model, loaders, 0, eval=True))


    accuracy_fused = evaluate_performance_simple(fused_model, loaders, 0, eval=True)
    print('Test Accuracy of the model fused after: %.2f' % accuracy_fused)
    print(fused_accs)"""




    






        


