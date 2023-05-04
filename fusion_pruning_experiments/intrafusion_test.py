from fusion_IF import intrafusion_bn
from performance_tester import train_during_pruning
from pruning_modified import prune_structured_intra
from fusion_utils_IF import MetaPruneType, PruneType
import copy
import torch



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
    if prune_iter_steps == 0:
        return intrafusion_bn(model, full_model = model, meta_prune_type = meta_prune_type, prune_type=prune_type, model_name=model_name, sparsity=sparsity, fusion_type="weight", gpu_id = gpu_id, resnet = resnet, train_loader=None)
    else:
        prune_steps = prune_structured_intra(net=copy.deepcopy(model), loaders=None, num_epochs=0, gpu_id=gpu_id, example_inputs=torch.randn(1, 3, 32, 32),
                    out_features=10, prune_type=prune_type, sparsity=sparsity, train_fct=None, total_steps=prune_iter_steps)
        fused_model_g = model
        for prune_step in prune_steps:
            fused_model_g = intrafusion_bn(fused_model_g, model_name = model_name, sparsity=sparsity, fusion_type="weight", full_model = model, small_model=prune_step, gpu_id = gpu_id, resnet = resnet, train_loader=None)
            fused_model_g,_ = train_during_pruning(fused_model_g, loaders=loaders, num_epochs=num_epochs, gpu_id = gpu_id, prune=False, performed_epochs=0)
        return fused_model_g


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
