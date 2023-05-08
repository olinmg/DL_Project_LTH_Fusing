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
import torch.nn as nn
import torchvision.models as model_archs
from torch.optim.lr_scheduler import StepLR

from fusion import fusion_bn
from fusion_IF import intrafusion_bn
from fusion_utils_IF import MetaPruneType, PruneType
from pruning_modified import prune_structured_intra
from train_resnet50 import train_resnet50, validate


def iterative_pruning(model, iter_num_epochs, prune_iter_steps, prune_type, sparsity):
    meta_prune_type = MetaPruneType.DEFAULT  # using regular pruning
    accuarcies_between_prunesteps = []
    last_model_path = 0
    for iter_step in range(prune_iter_steps):
        # just to figure out the prune step sizes
        prune_steps = prune_structured_intra(
            net=copy.deepcopy(model),
            loaders=None,
            num_epochs=0,
            gpu_id=gpu_id,
            example_inputs=example_input,  # torch.randn(1, 3, 32, 32),
            out_features=out_features,
            prune_type=prune_type,
            sparsity=sparsity,
            train_fct=None,
            total_steps=prune_iter_steps,
        )

        fused_model_g = model
        for prune_step in prune_steps:
            # 1. do the pruning of the network
            fused_model_g = intrafusion_bn(
                fused_model_g,
                meta_prune_type=meta_prune_type,
                out_features=out_features,
                example_inputs=example_input,
                prune_type=prune_type,
                model_name=model_name,
                sparsity=sparsity,
                fusion_type="weight",
                full_model=model,
                small_model=prune_step,
                gpu_id=gpu_id,
                resnet=True,
                train_loader=None,
            )

            after_prune_acc = validate(
                model=fused_model_g, val_loader=loaders["test"], gpu_id=gpu_id
            )
            accuarcies_between_prunesteps.append(after_prune_acc)
            # 2. store the prune model
            optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

            state = {
                "epoch": 1,
                "arch": "resnet50",
                "state_dict": model.state_dict(),
                "best_acc1": 0,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            store_model_path = f"{model_file}_{iter_step}iter"
            torch.save(state, store_model_path)

            # 3. retrain the stored pruned model (using train_resnet50/main.py)
            last_model_path = f"{model_file}_{iter_step}iter{iter_num_epochs}"
            train_resnet50(
                num_epochs_to_train=iter_num_epochs,
                dataset_path="/local/home/stuff/imagenet",
                checkpoint_path=store_model_path,
                result_model_path_=last_model_path,
            )

            # 4. load the retrained model
            model = model_archs.__dict__["resnet50"]()
            model = torch.nn.DataParallel(model)
            checkpoint = torch.load(last_model_path)
            model.load_state_dict(checkpoint["state_dict"])
            after_retrain_acc = validate(model=model, val_loader=loaders["test"], gpu_id=gpu_id)
            # after_retrain_acc = evaluate_performance_imagenet(model, loaders["test"], gpu_id)
            accuarcies_between_prunesteps.append(after_retrain_acc)
            model = model.module.to("cpu")

    return model, accuarcies_between_prunesteps, last_model_path


gpu_id = 0
retrain_epochs = 1
model_name = "resnet18"
model_file = "resnet18_withBN_diff_weight_init_False_cifar10_eps300_1"
model_path = f"./models/models_resnet18/{model_file}"
dataset_name = "cifar10"

print("Loading resnet50 model ...")
loaded_model = get_pretrained_model_by_name(model_path, gpu_id=0)

print("Loading imagenet dataset ...")
loaders = get_imagenet_data_loader()
out_features = 1000
example_input = torch.randn(1, 3, 224, 224)
eval_func = evaluate_performance_imagenet

prune_params = {
    "prune_type": "l1",
    "sparsity": 0.8,
    "example_input": example_input,
    "out_features": out_features,
    "use_iter_prune": True,
    "prune_iter_steps": 4,
    "prune_iter_epochs": 1,
    "loaders": loaders,
    "gpu_id": gpu_id,
    "model_name": model_name,
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
    "eval_function": eval_func,
    "loaders": loaders,
    "gpu_id": gpu_id,
}

model_accuracy_development = {}

from performance_tester_utils import evaluate_performance_imagenet

# 0. original model accuracy
print("Starting to evaluate the original model performance ...")
original_acc = validate(loaders["test"], loaded_model, gpu_id)
model_accuracy_development["original_accuracy"] = original_acc

# 1. prune the model - possibly iteratively
print("Starting the (iterative) pruning ...")
pruned_model, accuarcies_between_prunesteps, last_model_path = iterative_pruning(
    model=loaded_model,
    num_epochs=prune_params.get("prune_iter_epochs"),
    prune_iter_steps=prune_params.get("prune_iter_steps"),
    prune_type=prune_params.get("prune_type"),
    sparsity=prune_params.get("sparsity"),
)
val_perf = accuarcies_between_prunesteps[-1]
model_accuracy_development["iterative_pruning"] = accuarcies_between_prunesteps

with open(
    f"./results_of_pruning_experiment/retraining_accuracies_{model_file}.json", "w"
) as outfile:
    json.dump(model_accuracy_development, outfile, indent=4)

print(f"Model pruning is done. Final accuracy: {val_perf}")


# 2. additional retraining of the model
final_model_path = f"{model_file}_{prune_params.get('prune_iter_steps')}iter{prune_params.get('prune_iter_epochs')}_T{retrain_epochs}"
print(f"Starting additional training for {retrain_epochs} epochs ...")
train_resnet50(
    num_epochs_to_train=retrain_epochs,
    dataset_path="/local/home/stuff/imagenet",
    checkpoint_path=last_model_path,
    result_model_path_=final_model_path,
)
final_model = model_archs.__dict__["resnet50"]()
final_model = torch.nn.DataParallel(final_model)
checkpoint = torch.load(final_model_path)
final_model.load_state_dict(checkpoint["state_dict"])
after_retrain_acc = validate(loaders["test"], final_model, gpu_id)
# after_retrain_acc = evaluate_performance_imagenet(final_model, loaders["test"], gpu_id)

model_accuracy_development["retraining"] = after_retrain_acc
with open(f"./results_of_pruning_experiment/all_accuracies_{model_file}.json", "w") as outfile:
    json.dump(model_accuracy_development, outfile, indent=4)

print(f"Done. Final accuracy is: {after_retrain_acc}")
