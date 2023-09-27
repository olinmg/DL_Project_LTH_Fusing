import torch
import numpy as np
import matplotlib.pyplot as plt
from train import get_cifar_data_loader, get_cifar100_data_loader
# from models import get_pretrained_models
from pruning_modified import prune_structured_new
from torch_pruning_new.optimal_transport import OptimalTransport
import torch_pruning_new as tp
import copy
from collections import namedtuple
import json
from train_2 import get_pretrained_model

# Define the main function
def main(selected_model):
    # Utils
    def find_ignored_layers(model_original, out_features):
        ignored_layers = []
        for m in model_original.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == out_features:
                ignored_layers.append(m)

        return ignored_layers

    # Parameters (specify via params later)
    original_model_basis_name = "vgg11_bn" if "vgg" in selected_model else "resnet18"
    gpu_id = -1
    num_models = 1
    dataset_name = "cifar100" if "100" in selected_model else "cifar10"
    prune_iter_epochs = 1
    example_inputs = torch.randn(1, 3, 32, 32)
    sparsities = [idx/10 for idx in range(1, 8)]
    prune_types = ["l1", "l2"]
    create_plots = True

    # Parameters for histogram pruning
    distance_metric = lambda x, y: (torch.sum((x - y)**2))**0.5

    # Load the dataset
    Args = namedtuple('Args', ['diff_weight_init'])
    args = Args(diff_weight_init=True)
    if dataset_name == "cifar10":
        loaders = get_cifar_data_loader(1, args)
        output_dim = 10
    elif dataset_name == "cifar100":
        loaders = get_cifar100_data_loader(1, args)
        output_dim = 100

    # Get three types of models: original, pruned, and Intra-Fusion
    config = dict(
        dataset="Cifar100" if dataset_name == "cifar100" else "Cifar10",
        model=original_model_basis_name,
        optimizer="SGD",
        optimizer_decay_at_epochs=[150, 250],
        optimizer_decay_with_factor=10.0,
        optimizer_learning_rate=0.1,
        optimizer_momentum=0.9,
        optimizer_weight_decay=0.0001,
        batch_size=256,
        num_epochs=300,
        seed=42,
    )

    if dataset_name == "cifar10":
        if "resnet" in selected_model:
            model_original, _ = get_pretrained_model(config, "models/resnet18_cifar10_0.checkpoint")
        elif "vgg" in selected_model:
            model_original, _ = get_pretrained_model(config, "models/vg11_bn_cifar10_300eps.checkpoint")
    elif dataset_name == "cifar100":
        if "resnet" in selected_model:
            model_original, _ = get_pretrained_model(config, "models/resnet18_cifar100_300eps.checkpoint")
        elif "vgg" in selected_model:
            model_original, _ = get_pretrained_model(config, "models/vg11_bn_cifar100_300eps.checkpoint")

    original_models = [model_original]

    print("Got the models")

    results = {}

    for idx in range(num_models):
        model_original = original_models[idx]

        DG = tp.DependencyGraph().build_dependency(model_original, example_inputs=example_inputs)

        ignored_layers = find_ignored_layers(model_original=model_original, out_features=output_dim)
        num_groups = 0
        for group in DG.get_all_groups_in_order(ignored_layers=ignored_layers):
            num_groups += 1

        ot = OptimalTransport(gpu_id=gpu_id)
        groups = [i for i in range(num_groups)]

        # Prune whole network
        groups = [None]

        for prune_type in prune_types:
            for group_idx in groups:
                print(group_idx)
                for sparsity in sparsities:

                    pruned_model = copy.deepcopy(model_original)
                    prune_structured_new(
                        pruned_model,
                        None,
                        None,
                        example_inputs,
                        output_dim,
                        prune_type,
                        gpu_id,
                        sparsity=sparsity,
                        prune_iter_steps=1,
                        optimal_transport=None,
                        backward_pruning=True,
                        group_idxs=None,
                        train_fct=None)

                    if_model = copy.deepcopy(model_original)
                    prune_structured_new(
                        if_model,
                        None,
                        None,
                        example_inputs,
                        output_dim,
                        prune_type,
                        gpu_id,
                        sparsity=sparsity,
                        prune_iter_steps=1,
                        optimal_transport=ot,
                        backward_pruning=True,
                        group_idxs=None,
                        train_fct=None)

                    # Compare the outputs of the models
                    outputs_original, output_pruned, output_if = [], [], []

                    with torch.no_grad():

                        for i, (inputs, _) in enumerate(loaders["test"]):

                            outputs_original.append(model_original(inputs))
                            output_pruned.append(pruned_model(inputs))
                            output_if.append(if_model(inputs))

                    # Compute the distances of the models
                    distances_pruned, distances_if = [], []
                    for i in range(len(outputs_original)):
                        for idx in range(len(outputs_original[i])):
                        
                            distances_pruned.append(distance_metric(outputs_original[i][idx], output_pruned[i][idx]).item())
                            distances_if.append(distance_metric(outputs_original[i][idx], output_if[i][idx]).item())

                    key_tuple = f"num{idx}, prune_type:{prune_type}, group_idx:{group_idx}, sparsity:{sparsity}, origname:{original_model_basis_name}"
                    results[key_tuple] = (distances_pruned, distances_if)

                    if create_plots:
                        # Plot the histograms in one single plot
                        bin_edges = np.arange(min(distances_pruned), max(distances_pruned) + 0.5, 0.5)
                        plt.figure()
                        plt.hist(distances_pruned, bins=bin_edges, alpha=0.5, label='Pruned')
                        plt.hist(distances_if, bins=bin_edges, alpha=0.5, label='Intra-Fusion')
                        plt.legend(loc='upper right')
                        plt.title(f"Distance Histogram for {original_model_basis_name}")
                        plt.xlabel("Distance to outputs of the original model")
                        plt.ylabel("Frequency")
                        plt.savefig(f"plots_friedrich/distance_histogram_{original_model_basis_name}_Num{idx}_PruneType{prune_type}_GroupIdx{group_idx}_Sparsity{sparsity}_ds{dataset_name}.png")
                    
                    # Release memory by setting variables to None
                    pruned_model = None
                    if_model = None


    file_path = f"plots_friedrich/output_preservation_{selected_model}.json"

    # Write the dictionary to the JSON file
    with open(file_path, "w") as json_file:
        json.dump(results, json_file)

# Add this if block to check if the script is executed as the main program
if __name__ == "__main__":
    import argparse

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run the script with a selected model.")

    # Add an argument for the selected_model
    parser.add_argument("--selected_model", type=str, default="resnet_cifar10", help="Selected model")

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the selected_model as an argument
    main(args.selected_model)
