import torch
import numpy as np

from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch_pruning as tp


### TO-DO:
### RETURN MODEL
### RETRAIN AFTER PRUNING IN EACH STEP

def prune_unstructured(net, prune_type, amount=0.2):
    parameters_to_prune = []
    for _q, module in net.named_modules() :
            # TODO: Attention! Here all the layer-types that are contained in the networks should be contained
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
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
def prune_structured(net, loaders, num_epochs, example_inputs, out_features, prune_type, gpu_id, sparsity=0.5, total_steps=3,train_fct=None):

    print(f"Structured pruning with type {prune_type} and channel sparsity {sparsity}")
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
        imp = tp.importance.StrcuturalImportance
    elif prune_type ==  'lamp':
        imp = tp.importance.LAMPImportance()
    else:
        raise ValueError("Prune type not supported")


    ignored_layers = []
    model = net #Correct????
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == out_features:
            ignored_layers.append(m)

    total_steps = int(total_steps)
    
    if next(model.parameters()).is_cuda:
        model.to("cpu")
    
    pruner = tp.pruner.LocalMagnitudePruner( 
        model,
        example_inputs,
        importance=imp,
        total_steps=total_steps, # number of iterations
        ch_sparsity=sparsity, # channel sparsity
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
            model, val_acc_per_epoch = train_fct(model, loaders, num_epochs, gpu_id)

    #The model is returned, but the pruning is done in situ...
    return model



from torch.autograd import Variable
from torch import optim

def train_during_pruning(model, loaders, num_epochs, gpu_id):
    '''
    Has to be a function that loads a dataset. 

    A given model and an amount of epochs of training will be given.
    '''

    loss_func = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    model.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    val_acc_per_epoch = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            if gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y

            predictions = model(b_x)
            loss = loss_func(predictions, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
        this_epoch_acc = evaluate_performance_simple(input_model=model, loaders=loaders, gpu_id=gpu_id)
        val_acc_per_epoch.append(this_epoch_acc)
    return model, val_acc_per_epoch



def test2():

    example_inputs = torch.randn(1,1, 28,28)
    print(model, tp.utils.count_params(model))
    prune_structured(model, example_inputs, 10, 'random', 4)
    print(model, tp.utils.count_params(model))



