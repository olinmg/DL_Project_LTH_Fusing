import copy
from models import get_pretrained_models
from pruning_modified import prune_structured_new
from torch_pruning_new.optimal_transport import OptimalTransport
import torch_pruning_new as tp
import torch

def find_ignored_layers(model_original, out_features):
    ignored_layers = []
    for m in model_original.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == out_features:
            ignored_layers.append(m)
    
    return ignored_layers


if __name__ == '__main__':

    model_name = "vgg11_bn"
    example_inputs = torch.randn(1, 3, 32, 32)
    out_features = 10
    file_name = "vgg11_bn_diff_weight_init_False_cifar10_eps300_A"
    gpu_id = -1


    model_original = get_pretrained_models(model_name, file_name, gpu_id, 1, output_dim=out_features)[0]

    DG = tp.DependencyGraph().build_dependency(model_original, example_inputs=example_inputs)

    ignored_layers = find_ignored_layers(model_original=model_original, out_features=out_features)
    num_groups = 0
    for group in DG.get_all_groups_in_order(ignored_layers=ignored_layers):
        num_groups += 1
    
    
    ot = OptimalTransport(gpu_id=gpu_id)
    meta_pruning_types = [None, ot]
    sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    prune_types = ["l1"]

    groups = [i for i in range(num_groups)]

    for prune_type in prune_types:
        for group_idx in groups:
            for sparsity in sparsities:
                for meta_prune in meta_pruning_types:
                    pruned_model = copy.deepcopy(model_original)
                    prune_structured_new(
                        pruned_model,
                        None,
                        None,
                        example_inputs,
                        out_features,
                        prune_type,
                        gpu_id,
                        sparsity=sparsity,
                        prune_iter_steps=1,
                        optimal_transport=meta_prune,
                        backward_pruning=True,
                        group_idxs=[group_idx],
                        train_fct=None)
                    
                    for ((name_orig, module_orig), (name, module)) in list(zip(model_original.named_modules(), pruned_model.named_modules())):
                        if isinstance(module_orig, (torch.nn.Conv2d, torch.nn.Linear)):
                            print(f"{module_orig.weight.shape} -> {module.weight.shape}")


                    
                    










