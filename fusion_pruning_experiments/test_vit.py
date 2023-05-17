from model import PointNet, DGCNN
import argparse
import torch.nn as nn
import torch
from pruning_modified import prune_structured, prune_structured_intra, prune_structured_layer
import copy
import ot
import numpy as np

import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from data import ModelNet40
from models import ViT, get_pretrained_models
from torchvision import datasets
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

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

def get_ground_metric(coordinates1, coordiantes2, bias1, bias2):
    return torch.cdist(coordinates1, coordiantes2, p=1)


def prune(model, sparsity, prefix, reference, p=1, type="IF"):
    net0 = None
    net3 = None
    for name, module in model.named_modules():
        if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
            continue
        if prefix in name and "net.0" in name:
            net0 = module.weight
        elif prefix in name and "net.3" in name:
            net3 = module.weight
    
    net3 = net3.permute(1,0)

    importance = [torch.norm(net0, p=p, dim=1), torch.norm(net3, p=p, dim=1)]
    importance = torch.stack(importance, dim=0)

    importance = importance.mean(dim=0).view(-1)


    amount_pruned = int((1-sparsity)*net0.shape[0])
    print("amount pruned: ", amount_pruned)
    _, indices = torch.topk(importance, amount_pruned)

    layers = [net0, net3]

    basis_layers = []
    for vec in layers:
        basis_layers.append(torch.index_select(vec, 0, indices))
    
    M = []
    for i in range(0, len(layers)):
        M.append(get_ground_metric(layers[i], basis_layers[i], None, None))
    M = torch.stack(M)
    M = torch.mean(M, dim=0)


    target_histo = np.ones(basis_layers[0].shape[0])/basis_layers[0].shape[0]
    cardinality = layers[0].shape[0]

    #source_histo = torch.pow(importance, 1).cpu().detach().numpy()

    source_histo = np.ones(cardinality) if type == "IF" else np.zeros(cardinality)
    for indice in indices:
        source_histo[indice] = cardinality/indices.size()[0]
    #source_histo = torch.pow(importance, 1).cpu().detach().numpy()
    source_histo /= np.sum(source_histo)

    cpuM = M.data.cpu().numpy() # POT does not accept array on GPU

    T = ot.emd(source_histo, target_histo, cpuM)        

    T_var = torch.from_numpy(T).float()

    """if type == "IF":
        T_var *= torch.pow(importance, 3)[:,None]"""
    T_var = T_var / T_var.sum(dim=0)

    for i, layer in enumerate(layers):
        layers[i] = torch.matmul(T_var.t(), layer)
    

    for name, module in reference.named_modules():
        if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
            continue
        if prefix in name and "net.0" in name:
            module.weight = nn.Parameter(layers[0])
        elif prefix in name and "net.3" in name:
            module.weight = nn.Parameter(layers[1].permute(1,0))
    return reference

def check(model):
    prefix = "0.0.fn.to_qkv"
    qkv = None
    for name, module in model.named_modules():
        if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
            continue
        if prefix in name:
            qkv = module.weight
    
    
    query = qkv[:512]
    key = qkv[512:1024]

    query_0 = query[:, 0]
    key_0 = key[:, 0]

    query[:, 0] = query[:, 1]
    key[:, 0] = key[:, 1]
    query[:, 1] = query_0
    key[:, 1] = key_0

    qkv[:512] = query
    qkv[512:1024] = key

    for name, module in model.named_modules():
        if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
            continue
        if prefix in name:
            module.weight = qkv
    
def checking():
    x = torch.randn([128, 65, 512])
    linear = nn.Linear(512, 1536)
    print(linear.weight.shape)

    qkv = linear(x)
    print("qkv then: ", qkv.shape)

    print(qkv.shape)
    print(qkv.chunk(3, dim = -1)[0])
    print(torch.equal(qkv[:,:,:512], qkv.chunk(3, dim = -1)[0]))
    qkv = qkv.chunk(3, dim = -1)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = 8), qkv)
        #print("q and k have shape: ", q.shape)
        #exit()

    print("k shape before:", k.shape)
    dots = torch.matmul(q, k.transpose(-1, -2))
    print("dots are: ", dots.shape)

    query_0 = q[:, 0].clone()
    key_0 = k[:, 0].clone()

    print("k shape down here: ", k.shape)

    q[:,0] = q[:, 1]
    k[:, 0] = k[:, 1]
    q[:, 1] = query_0
    k[:, 1] = key_0

    print("k shape: ", k.shape)
    print("k shape trans: ", k.transpose(-1, -2).shape)
    dots_1 = torch.matmul(q, k.transpose(-1, -2))


    print(dots == dots_1)
    print(torch.equal(dots, dots_1))
    exit()

    """query = linear.weight[:512]
    key = linear.weight[512:1024]

    query_0 = query[0].clone()
    key_0 = key[0].clone()

    query[0] = query[1]
    key[0] = key[1]
    query[1] = query_0
    key[1] = key_0

    linear.weight[:512] = query
    linear.weight[512:1024] = key

    qkv = linear(x)
    print("qkv now: ", qkv.shape)
    qkv = qkv.chunk(3, dim = -1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = 8), qkv)"""
    dots_1 = torch.matmul(q, k.transpose(-1, -2))

    print("dots now: ", dots_1)
    print(dots == dots_1)

    exit()
    

"""with torch.no_grad():
    checking()"""

loaders = get_cifar_data_loader()

"""model = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )"""

model = get_pretrained_models("vit", f"vit_diff_weight_init_True_cifar10", 0, 1, output_dim=10)[0]

for name, module in model.named_modules():
    if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
        continue
    print(f"name: {name} : {module}")

model.cpu()
print("Normal: ", evaluate_performance_simple(model, loaders, -1, True))

with torch.no_grad():
    check(model)

model.cpu()
print("After permute: ", evaluate_performance_simple(model, loaders, -1, True))

"""sparsity=0.2
t = prune_structured(net=copy.deepcopy(model), loaders=None, prune_iter_epochs=0, gpu_id=-1, example_inputs=torch.randn(1, 3, 32, 32).to("cpu"),
                    out_features=10, prune_type="l2", sparsity=sparsity, train_fct=None, prune_iter_steps=1)

model.cpu()
t = prune(model, sparsity=sparsity, prefix="0.1", reference=t, p=2, type="prune")
for name, module in t.named_modules():
    if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
        continue
    print(f"name: {name} : {module}")

print("Normally pruned ", evaluate_performance_simple(t, loaders, -1, True))

model.cpu()
model = prune(model, sparsity=sparsity, prefix="0.1", reference=t, p=2)
for name, module in model.named_modules():
    if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
        continue
    print(f"name: {name} : {module}")
print("Normally pruned ", evaluate_performance_simple(model, loaders, -1, True))"""


