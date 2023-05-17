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


def get_ground_metric(coordinates1, coordiantes2, bias1, bias2):
    return torch.cdist(coordinates1, coordiantes2, p=1)

def prune(model):
    conv1 = None
    conv2 = None
    conv5 = None

    conv1_orig_shape = None
    conv2_orig_shape = None
    conv5_orig_shape = None

    for (name, module) in model.named_modules():

        if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
            continue
        if name == "conv1.0":
            conv1 = module.weight
        elif name == "conv2.0":
            conv2 = module.weight
        elif name == "conv5.0":
            conv5 = module.weight
        print(f" name: {name} {module.weight.shape}")

    conv1_orig_shape = conv1.shape
    conv2_orig_shape = conv2.shape
    conv5_orig_shape = conv5.shape
    conv2_orig = conv2.clone().view(conv2.shape[0], -1)
    conv1 = conv1.view(conv1.shape[0], -1)
    conv2 = conv2.view(conv2.shape[0], -1)
    conv2 = conv2.permute((1, 0))
    conv2 = conv2.reshape(conv1.shape[0], -1)

    conv5_orig = conv5.clone()
    conv5 = conv5.view(conv5.shape[0], -1).permute(1, 0)[:conv1.shape[0],:]

    print(conv1.shape)
    print(conv2.shape)
    print(conv5.shape)

    layers = [conv1, conv2, conv5]
    importance = []

    for layer in layers:
        importance.append(torch.norm(layer, dim=1, p=1))
    importance = torch.stack(importance, dim=0)

    importance = importance.mean(dim=0)

    """imp_argsort = torch.argsort(importance)
    indices = imp_argsort[:32]"""
    _, indices = torch.topk(importance, 32)
    print("indices: ", indices)
    indices, _ = torch.sort(indices)
    #print("indices: ", indices)

    #indices, _= torch.sort(indices)
    #print("indices are: ", indices)

    basis_layers = []
    for vec in layers:
        basis_layers.append(torch.index_select(vec, 0, indices))

    print("IMPORTANCE: ", importance)

    M = []
    for i in range(0, len(layers)):
        M.append(get_ground_metric(layers[i], basis_layers[i], None, None))
    M = torch.stack(M)
    M = torch.mean(M, dim=0)

    target_histo = np.ones(basis_layers[0].shape[0])/basis_layers[0].shape[0]
    cardinality = layers[0].shape[0]
    #source_histo = np.ones(layers[0].shape[0])/layers[0].shape[0]
    source_histo = np.zeros(cardinality)
    for indice in indices:
        source_histo[indice] = cardinality/indices.size()[0]

    source_histo /= np.sum(source_histo)

    cpuM = M.data.cpu().numpy() # POT does not accept array on GPU

    T = ot.emd(source_histo, target_histo, cpuM)        

    T_var = torch.from_numpy(T).float()
    
    #T_var *= torch.pow(importance, 1)[:,None]

    """eps=1e-7
    marginals = torch.ones(T_var.shape[1]) / T_var.shape[0]
    marginals = torch.diag(1.0/(marginals + eps))  # take inverse

    T_var = torch.matmul(T_var, marginals)"""
    T_var = T_var / T_var.sum(dim=0)

    for i, layer in enumerate(layers):
        if i == 1:
            layers[i] = torch.cat((torch.matmul(T_var.t(), conv2_orig[:, :64].t()), torch.matmul(T_var.t(), conv2_orig[:, 64:].t())), dim=0).t()
        else:
            layers[i] = torch.matmul(T_var.t(), layer)

    #layers = basis_layers
    print("layer0 shape", layers[0].shape)
    conv1 = layers[0].view(32, 6, 1, 1)

    """conv2 = torch.cat((layers[1][:, :64], layers[1][:, 64:]))
    conv2 = conv2.t()
    conv2 = conv2.reshape(64, 64, 1, 1)"""
    """conv2 = layers[1].view(64, -1)
    print("conv2 interm shape_: ", conv2.shape)
    conv2 = conv2.t()
    print("conv2 shape: ", conv2.shape)"""

    conv2 = layers[1].reshape(64, 64, 1, 1)

    conv5 = conv5_orig.view(conv5_orig.shape[0], -1)[:, 64:]
    print("conv5 raw: ", conv5.shape)
    print("layer 2: ", layers[2].shape)
    
    conv5 = torch.cat((layers[2].permute(1,0), conv5), dim=1)

    return conv1, conv2, conv5

def prune_pointnet(model, names, sparsity, p=1):
    conv1 = None
    conv2 = None 
    conv1_orig = None
    conv2_orig = None
    for (name, module) in model.named_modules():

        if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
            continue
        if name == names[0]:
            conv1_orig = module.weight.clone()
            conv1 = module.weight.view(module.weight.shape[0], -1)
        elif name == names[1]:
            conv2_orig = module.weight.clone()
            conv2 = module.weight.view(module.weight.shape[0], -1)
        print(f" name: {name} {module.weight.shape}")
    
    importance = [torch.norm(conv1, p=p, dim=1), torch.norm(conv2, p=p, dim=0)]
    importance = torch.stack(importance, dim=0)

    importance = importance.mean(dim=0).view(-1)

    """imp_argsort = torch.argsort(importance)
    indices = imp_argsort[:32]"""

    print("importance shape: ", importance.shape)
    amount_pruned = int((1-sparsity)*conv1.shape[0])
    print("amount pruned: ", amount_pruned)
    _, indices = torch.topk(importance, amount_pruned)

    layers = [conv1, conv2.permute(1,0)]

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
    #source_histo = np.ones(layers[0].shape[0])/layers[0].shape[0]
    #source_histo = torch.pow(importance, 1).cpu().detach().numpy()
    source_histo = np.ones(cardinality)
    for indice in indices:
        source_histo[indice] = cardinality/indices.size()[0]
    #source_histo *= torch.pow(importance, 1).cpu().detach().numpy()
    source_histo /= np.sum(source_histo)

    cpuM = M.data.cpu().numpy() # POT does not accept array on GPU

    T = ot.emd(source_histo, target_histo, cpuM)        

    T_var = torch.from_numpy(T).float()

    T_var *= torch.pow(importance, 3)[:,None]
    T_var = T_var / T_var.sum(dim=0)

    for i, layer in enumerate(layers):
        layers[i] = torch.matmul(T_var.t(), layer)
    
    conv1 = layers[0].view(layers[0].shape[0], conv1_orig.shape[1], 1)
    conv2 = layers[1].permute(1, 0).view(conv2_orig.shape[0], conv1.shape[0], 1)

    return conv1, conv2


def test(args, model):
    print("I am in test")
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    print("Done with test loader**********************************************")
    device = torch.device("cpu")#torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = model.to(device)

    model = nn.DataParallel(model)

    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()

        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    print(outstr)



if __name__ == "__main__":

    """x = torch.randn(64, 128)
    y0 = x.t()
    y = y0.reshape(64, 128)
    print(x == y.view(128, 64).t())
    exit()"""
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    model = PointNet(args)#.to("cpu")

    #model = nn.DataParallel(model)
    state_dict = torch.load("./models/model.t7")
    new_state_dict = {}
    for key, value in state_dict.items():
        print(key)
        print(key.replace("module.", ""))
        new_state_dict[key.replace("module.", "")] = value

    model.load_state_dict(new_state_dict)
    model.to("cpu")
    print("model device: ", next(model.parameters()).is_cuda)

    for name, module in model.named_modules():
        print(f"{name} : {module}")
    print("--------Done with first one---------")
    for name, module in model.named_modules():

        if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
            continue
        print(f"name: {name} : {module}")

    #conv1, conv2, conv5 = prune(model)  
    
    print("Full model accuracy")
    test(args, model)
    model.cpu()
    sparsity = 0.2
    names = ["conv4", "conv5"]
    t = prune_structured(net=copy.deepcopy(model), loaders=None, prune_iter_epochs=0, gpu_id=-1, example_inputs=torch.randn(16, 3, 1024).to("cpu"),
                    out_features=10, prune_type="l1", sparsity=sparsity, train_fct=None, prune_iter_steps=1)
    
    conv1, conv2 = prune_pointnet(model, names, sparsity=sparsity)
    
    for name, module in t.named_modules():

        if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
            continue
        print(f"name: {name} : {module}")
    print("--------AFTER--------")
    test(args, t)

    for (name, module) in t.named_modules():

        if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
            continue
        if name == names[0]:
            print("GOT IN HERE")
            module.weight = nn.Parameter(conv1)
        elif name == names[1]:
            module.weight = nn.Parameter(conv2)
        print(f" name: {name} {module.weight.shape}")
    test(args, t)

    """test(args, t)"""
    """print("--------Done with first one---------")
    for (name0, module0), (name, module) in zip(model.named_modules(), t.named_modules()):

        if not(isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)):
            continue
        print(f" name: {name} {module0.weight.shape} : {module.weight.shape}")
        if module.weight.shape == module0.weight.shape:
            print(module.weight == module0.weight)
        else:
            print(False)
        if name == "conv1.0":
            #print(module.weight[:, 0, :, :])
            print("Conv1")
            conv1_ = conv1[:, 0, :, :].view(-1)
            print("conv1: meine: ", conv1_)
            print("conv1 other: ", module.weight[:, 0, :, :].view(-1))

            for i, val in enumerate(conv1_):
                if (i % 32 == 0):
                    print("----------------------------------------------------------")
                print("val: ", val)
                if val not in module.weight[:, 0, :, :].view(-1):
                    print("NOT FOUND")
                else:
                    print("Found")
            module.weight == nn.Parameter(conv1)
        if name == "conv2.0":
            print("CONV2")
            conv2_ = conv2[0, :, :, :].view(-1)
            print("conv1: meine: ", conv2_)
            print("conv1 other: ", module.weight[0, :, :, :].view(-1))

            for i, val in enumerate(conv2_):
                if (i % 32 == 0):
                    print("----------------------------------------------------------")
                print("val: ", val)
                if val not in module.weight[0, :, :, :].view(-1):
                    print("NOT FOUND")
                else:
                    print("Found")

            module.weight = nn.Parameter(conv2)
        if name == "conv5.0":
            print("CONV5")
            conv2_ = conv5[0, 3].view(-1)
            print("conv1: meine: ", conv2_)
            print("conv1 other: ", module.weight[0, :32].view(-1))
            print("and trad: ", module.weight[:32])

            for i, val in enumerate(conv2_):

                if val not in module.weight[0, :32].view(-1):
                    print("NOT FOUND")
                else:
                    print("Found")

            print("conv5 weiht: ", conv5.shape)
            print("module weight: ", module.weight.shape)
            print("Get down here!!!!")
            module.weight = nn.Parameter(conv5.view(module.weight.shape))
            #print(module.weight[0, :, :, :])"""