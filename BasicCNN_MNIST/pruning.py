import torch
import numpy as np

from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch_pruning as tp

from base_convNN import CNN, train, test, loaders
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_FILE_NAME = "./base_cnn_model_dict.pth"

model = CNN()
model.load_state_dict(torch.load(MODEL_FILE_NAME))
model.eval()


### TO-DO:
### RETURN MODEL
### RETRAIN AFTER PRUNING IN EACH STEP

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


#Example inputs: any inputs of the correct shape
#Out features: the number of output features of the model
#Train fct: a function that takes the model and the data loaders as input and trains the model
def prune_structured(net, example_inputs, out_features, prune_type, total_steps=3, train_fct = None):


    ori_size = tp.utils.count_params(net)
    imp = None

    if prune_type == 'random':
        imp = tp.importance.RandomImportance()
    elif prune_type ==  'sensitivity':
        imp = tp.importance.SensitivityImportance()
    elif prune_type ==  'l1':
        imp = tp.importance.MagnitudeImportance(1)
    elif prune_type ==  'l2':
        imp = tp.importance.MagnitudeImportance(2)
    elif prune_type ==  'l_inf':
        imp = tp.importance.MagnitudeImportance(np.inf)
    elif prune_type ==  'hessian':
        imp = tp.importance.HessianImportance()
    elif prune_type ==  'bnscale':
        imp = tp.importance.BNScaleImportance()
    elif prune_type ==  'structural':
        imp = tp.importance.StructuralImportance()
    elif prune_type ==  'lamp':
        imp = tp.importance.LAMPImportance()
    else:
        raise ValueError("Prune type not supported")


    ignored_layers = []

    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == out_features:
            ignored_layers.append(m)

    total_steps = int(total_steps)
    pruner = tp.pruner.LocalMagnitudePruner( 
        model,
        example_inputs,
        importance=imp,
        total_steps=total_steps, # number of iterations
        ch_sparsity=0.5, # channel sparsity
        ignored_layers=ignored_layers, # ignored_layers will not be pruned
    )
    for i in range(total_steps): # iterative pruning
        print(i)
        pruner.step()
        print(
            "  Params: %.2f M => %.2f M"
            % (ori_size / 1e6, tp.utils.count_params(model) / 1e6)
        )

        #Potentially retrain the model
        #The train function is a function that takes the model and does the training on it.
        #It probably calls other training functions

        if train_fct is not None:
            train_fct(model)

    #The model is returned, but the pruning is done in situ...
    return model



def test2():

    example_inputs = torch.randn(1,1, 28,28)
    print(model, tp.utils.count_params(model))
    prune_structured(model, example_inputs, 10, 'random', 4)
    print(model, tp.utils.count_params(model))



