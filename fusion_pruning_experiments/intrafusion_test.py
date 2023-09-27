from fusion_IF import intrafusion_bn
from torch_pruning_new.optimal_transport import OptimalTransport
from performance_tester import train_during_pruning, train_during_pruning_cifar100
from pruning_modified import prune_structured_intra
from fusion_utils_IF import MetaPruneType, PruneType
import copy
import torch
import torch_pruning_new as tp_n

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


def wrapper_intra_fusion(model, model_name: str, resnet: bool, sparsity: float, prune_iter_steps: int, num_epochs: int, loaders, prune_type: PruneType, meta_prune_type: MetaPruneType, gpu_id: int):
    """
    :param model: The model to be pruned
    :param model_name: The name of the model
    :param resnet: resnet = True means that the model is a resnet
    :param sparsity: The desired sparsity. sparsity = 0.9 means that 90% of the nodes within the layers are removed
    :param prune_iter_steps: The amount of intermediate pruning steps it takes to arrive at the desired sparsity
    :param num_epochs: The amount of epochs it retrains for the intermediate pruning steps (see prune_iter_steps)
    :param loaders: The loaders containing the training and test data
    :param prune_type: The neuron importance metric. Options: "l1" and "l2"
    :param meta_prune_type: If meta_prune_type = MetaPruneType.DEFAULT then it will prune the model in the normal way. If meta_prune_type = MetaPruneType.IF it will do intrafusion
    :return: the pruned model
    """ 

    ignored_layers = []

    pruner_model = copy.deepcopy(model)
    pruner_model.cpu()

    for m in pruner_model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)
    

    pruner = tp_n.pruner.MagnitudePruner(
        pruner_model,
        torch.randn(1, 3, 32, 32),
        importance=tp_n.importance.MagnitudeImportance(1),
        iterative_steps=prune_iter_steps,  # number of iterations
        ch_sparsity=sparsity,  # channel sparsity
        ignored_layers=ignored_layers,  # ignored_layers will not be pruned
        optimal_transport=None if meta_prune_type == MetaPruneType.DEFAULT else OptimalTransport(gpu_id=gpu_id, target="most_important"),
        backward_pruning=False,
        dimensionality_preserving=False
    )

    if prune_iter_steps == 0:
        return intrafusion_bn(model, full_model = model, meta_prune_type = meta_prune_type, prune_type=prune_type, model_name=model_name, sparsity=sparsity, fusion_type="weight", gpu_id = gpu_id, resnet = resnet, train_loader=loaders)
    else:
        epoch_accuracies = []
        prune_steps = prune_structured_intra(net=copy.deepcopy(pruner_model), loaders=None, num_epochs=0, gpu_id=gpu_id, example_inputs=torch.randn(1, 3, 32, 32),
                    out_features=10, prune_type=prune_type, sparsity=sparsity, train_fct=None, total_steps=prune_iter_steps)

        for prune_step in prune_steps:
            print("Right before step: ", evaluate_performance_simple(pruner_model, loaders, 0, eval=True))
            pruner.step()
            print("Right after step: ", evaluate_performance_simple(pruner_model, loaders, 0, eval=True))
            _, epoch_accuracy= train_during_pruning(pruner_model, loaders=loaders, num_epochs=num_epochs, gpu_id = gpu_id, prune=False, performed_epochs=0)
            print("After training: ", evaluate_performance_simple(pruner_model, loaders, 0, eval=True))
            epoch_accuracies.extend(epoch_accuracy)
            for ((name_orig, module_orig), (name, module)) in list(zip(model.named_modules(), pruner_model.named_modules())):
                        if isinstance(module_orig, (torch.nn.Conv2d, torch.nn.Linear)):
                            print(f"{module_orig.weight.shape} -> {module.weight.shape}")
            continue
            print("Right before step: ", evaluate_performance_simple(pruner_model, loaders, 0, eval=True))
            pruner_model = intrafusion_bn(pruner_model, model_name = model_name, meta_prune_type = meta_prune_type, prune_type = prune_type, sparsity=sparsity, fusion_type="weight", full_model = model, small_model=prune_step, gpu_id = gpu_id, resnet = resnet, train_loader=None)
            print("Right after step: ", evaluate_performance_simple(pruner_model, loaders, 0, eval=True))
            _,epoch_accuracy= train_during_pruning(pruner_model, loaders=loaders, num_epochs=num_epochs, gpu_id = gpu_id, prune=False, performed_epochs=0)
            epoch_accuracies.extend(epoch_accuracy)
        return pruner_model, epoch_accuracies


"""
For Olin:
Make sure of the following:
meta_prune_type = MetaPruneType.IF
prune_iter_steps = 4 (you can also try 5, seems to work better for sparsity = 0.9)
num_epochs = 10
It's also very important that you set resnet=True if you're using a resnet and that you pass in the correct model_name (i.e. don't pass in vgg11 if it's a resnet18)
Since you probably want to compare Intra-Fusion with default pruning, please don't use the torch-pruning library directly! Instead, just use wrapper_intra_fusion BUT pass in: meta_prune_type = MetaPruneType.DEFAULT. Then it will do default pruning.
The rest should be self-explanatory, otherwise message me :)

------------------------------------------------------------------------

For Friedrich::
Make sure of the following:
meta_prune_type = MetaPruneType.IF
model_name = "vgg11" (Since you're doing Intra-Fusion for a vgg11)
resnet = False
prune_iter_steps = 0
num_epochs = 0
loaders = None (You don't need it because you don't retrain)
Let me know if something is unclear :)
"""
