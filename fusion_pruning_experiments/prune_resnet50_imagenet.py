import copy
import json
import math
import os
import shutil

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

from fusion_utils import FusionType
from model_caching import (
    ensure_folder_existence,
    file_already_exists,
    get_model_trainHistory,
    model_already_exists,
    save_experiment_results,
    save_model,
    save_model_trainHistory,
)

# import main #from main import get_data_loader, test
from models import get_model, get_pretrained_model_by_name, get_pretrained_models
from parameters import get_parameters
from performance_tester_utils import (
    accuracy,
    evaluate_performance_imagenet,
    evaluate_performance_simple,
    float_format,
    fusion_test_manager,
    get_cifar10_data_loader,
    get_cifar100_data_loader,
    get_imagenet_data_loader,
    get_mnist_data_loader,
    original_test_manager,
    pruning_test_manager,
    train_during_pruning,
    train_during_pruning_resnet50,
    update_running_statistics,
    wrapper_first_fusion,
    wrapper_structured_pruning,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpu_id = -1
model_name = "cnn_diff_weight_init_False_mnist_1"
model_path = f"./models/models_cnn/{model_name}"
dataset_name = "mnist"

resnet50_model = get_pretrained_model_by_name(model_path, gpu_id=0)

if dataset_name == "mnist":
    loaders = get_mnist_data_loader()
    output_dim = 10
elif dataset_name == "cifar10":
    loaders = get_cifar10_data_loader()
    output_dim = 10
elif dataset_name == "cifar100":
    loaders = get_cifar100_data_loader()
    output_dim = 100
elif dataset_name == "imagenet":
    loaders = get_imagenet_data_loader()
    output_dim = 1000

prune_params = {
    "prune_type": "l1",
    "sparsity": 0.6,
    "example_input": torch.randn(1, 3, 224, 224),
    "out_features": output_dim,
    "use_iter_prune": True,
    "prune_iter_steps": 4,
    "prune_iter_epochs": 10,
    "loaders": loaders,
    "gpu_id": gpu_id,
    "model_name": "resnet50",
    "use_intrafusion_for_pruning": False,
}

params = {
    "pruning_function": wrapper_structured_pruning,
    "fusion_function": wrapper_first_fusion(
        fusion_type=FusionType.ACTIVATION,
        train_loader=loaders["train"],
        gpu_id=gpu_id,
        num_samples=200,
    ),
    "eval_function": evaluate_performance_imagenet,
    "loaders": loaders,
    "gpu_id": gpu_id,
}

model_accuracy_development = {}

# prune the model - possibly iteratively
pruned_model_lis, pruned_model_accuracies, _ = pruning_test_manager(
    input_model_list=[resnet50_model], prune_params=prune_params, **params
)
pruned_model = pruned_model_lis[0]
# store the model after iterative pruning
iterprune_text = f"{prune_params['prune_iter_steps']}iter{prune_params['prune_iter_epochs']}"
torch.save(pruned_model, filename=f"{model_path}_{iterprune_text}.pth.tar")
model_accuracy_development["pruning_accuracies"] = pruned_model_accuracies
with open(f"pruning_accuracies_{model_name}.json", "w") as outfile:
    json.dump(model_accuracy_development, outfile, indent=4)

# additional retraining of the model
retrain_epochs = 0
retrained_pruned_model, val_acc_per_epoch = train_during_pruning_resnet50(
    pruned_model, loaders, retrain_epochs, gpu_id, prune=False, performed_epochs=0
)
torch.save(
    retrained_pruned_model, filename=f"{model_path}_{iterprune_text}_T{retrain_epochs}.pth.tar"
)
model_accuracy_development["retraining_accuracies"] = val_acc_per_epoch
with open(f"retraining_accuracies_{model_name}.json", "w") as outfile:
    json.dump(model_accuracy_development, outfile, indent=4)
model_accuracy_development["all_accuracies"] = pruned_model_accuracies.extend(val_acc_per_epoch)
with open(f"overall_pruning_retraining_accuracies_{model_name}.json", "w") as outfile:
    json.dump(model_accuracy_development, outfile, indent=4)
