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

import torch_pruning as tp
from fusion import fusion_bn
from fusion_IF import intrafusion_bn
from fusion_utils_IF import MetaPruneType, PruneType
from pruning_modified import prune_structured, prune_structured_intra
from train_resnet50 import train_resnet50, validate

gpu_id = 0
sparsity_in = 0.6
retrain_epochs = 20
prune_iter_epochs_in = 10
prune_iter_steps_in = 4
dataset_path = "/local/home/stuff/imagenet"  # "/local/home/gaf/coolvenv/testarea_imagenetIntegration/fake_imagenet"  # "/local/home/stuff/imagenet"
model_name = "resnet50"
model_file = "resnet50_imagenet_eps90_datasplit_0"
model_path = f"./models/models_resnet50/{model_file}"
dataset_name = "imagenet"


def prune_structured_resnet50(
    net,
    loaders,
    prune_iter_epochs,
    example_inputs,
    out_features,
    prune_type,
    gpu_id,
    sparsity=0.5,
    prune_iter_steps=4,
):
    print(f"Structured pruning with type {prune_type} and channel sparsity {abs(1-sparsity)}")
    imp = None

    if prune_type == "random":
        imp = tp.importance.RandomImportance()
    elif prune_type == "sensitivity":
        imp = tp.importance.SensitivityImportance()
    elif prune_type == "l1":
        imp = tp.importance.MagnitudeImportance(1)
    elif prune_type == "l2":
        imp = tp.importance.MagnitudeImportance(2)
    elif prune_type == "l_inf":
        imp = tp.importance.MagnitudeImportance(np.inf)
    elif prune_type == "hessian":
        imp = tp.importance.HessianImportance()
    elif prune_type == "bnscale":
        imp = tp.importance.BNScaleImportance()
    elif prune_type == "structural":
        imp = tp.importance.StrcuturalImportance
    elif prune_type == "lamp":
        imp = tp.importance.LAMPImportance()
    else:
        raise ValueError("Prune type not supported")

    ignored_layers = []
    model = net  # Correct????
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == out_features:
            ignored_layers.append(m)

    prune_iter_steps = int(prune_iter_steps)

    if next(model.parameters()).is_cuda:
        model.to("cpu")

    accuarcies_between_prunesteps = []
    step_size = sparsity / prune_iter_steps
    goal_sparsities = [(1 - step_size * (step + 1)) for step in range(prune_iter_steps)]
    prune_steps = []
    for i in range(len(goal_sparsities)):
        if i == 0:
            prune_steps.append(1 - goal_sparsities[i])
        else:
            prune_steps.append(1 - (goal_sparsities[i] / goal_sparsities[i - 1]))
    print(prune_steps)
    for i in range(prune_iter_steps):  # iterative pruning
        print(f"\n{i}: goal sparsity {prune_steps[i]}")
        ori_size = tp.utils.count_params(model)
        pruner = tp.pruner.LocalMagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            total_steps=1,  # number of iterations
            ch_sparsity=prune_steps[i],  # channel sparsity
            ignored_layers=ignored_layers,  # ignored_layers will not be pruned
        )
        pruner.step()
        print("  Params: %.2f M => %.2f M" % (ori_size / 1e6, tp.utils.count_params(model) / 1e6))
        model.cuda()
        after_prune_acc = 0
        after_prune_acc = validate(model=model, val_loader=loaders["test"], gpu_id=gpu_id)
        accuarcies_between_prunesteps.append(after_prune_acc)

        print(f"Doing iterative retraining for {prune_iter_epochs} epochs")
        optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        model = torch.nn.DataParallel(model).cuda()
        state = {
            "epoch": 0,
            "arch": "resnet50",
            "state_dict": model.state_dict(),
            "best_acc1": -1,  # after_prune_acc,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        store_model_path = f"{model_file}_s{int((1-goal_sparsities[i])*100)}.pth.tar"
        torch.save(state, f"{store_model_path.split('.')[0]}.pth.tar")
        torch.save(model, f"{store_model_path.split('.')[0]}.pth")

        last_model_path = f"{model_file}_s{int((1-goal_sparsities[i])*100)}iter{prune_iter_epochs}"
        print("\nHanding over to train_resnet50()")
        after_retrain_acc = train_resnet50(
            num_epochs_to_train=prune_iter_epochs,
            dataset_path=dataset_path,
            checkpoint_path=store_model_path,
            result_model_path_=last_model_path,
            ext_gpu=gpu_id,
        )
        print(
            f"Loding result of retraining into pruner with path: {last_model_path}_best_model.pth"
        )
        model = torch.load(f"{last_model_path}_best_model.pth")
        after_retrain_acc = validate(model=model, val_loader=loaders["test"], gpu_id=gpu_id)
        accuarcies_between_prunesteps.append(after_retrain_acc)
        model = model.module.to("cpu")
        print("\n ---------------------------------------------")
    return model, accuarcies_between_prunesteps, f"{last_model_path}_best_model.pth.tar"


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

        pruned_model_g = model
        for prune_step in prune_steps:
            # 1. do the pruning of the network
            pruned_model_g = intrafusion_bn(
                pruned_model_g,
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
            """
            ## VS
            pruned_model_g = prune_structured_resnet50(
                net=pruned_model_g,
                loaders=loaders,
                prune_iter_epochs=0,
                prune_iter_steps=0,
                gpu_id=gpu_id,
                example_inputs=example_input,
                out_features=out_features,
                prune_type=prune_type,
                sparsity=sparsity,
                train_fct=None,
            )
            """
            # after_prune_acc = validate(
            #     model=fused_model_g, val_loader=loaders["test"], gpu_id=gpu_id
            # )
            # accuarcies_between_prunesteps.append(after_prune_acc)
            # 2. store the prune model
            optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
            pruned_model_g = torch.nn.DataParallel(pruned_model_g).cuda()
            state = {
                "epoch": 1,
                "arch": "resnet50",
                "state_dict": pruned_model_g.state_dict(),
                "best_acc1": 0,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            store_model_path = f"{model_file}_{iter_step}iter"
            torch.save(state, f"{store_model_path}.pth.tar")
            torch.save(pruned_model_g, f"{store_model_path}.pth")
            # 3. retrain the stored pruned model (using train_resnet50/main.py)
            last_model_path = f"{model_file}_{iter_step}iter{iter_num_epochs}"
            print("Handing over to train_resnet50()")
            after_retrain_acc = train_resnet50(
                num_epochs_to_train=iter_num_epochs,
                dataset_path=dataset_path,
                checkpoint_path=store_model_path,
                result_model_path_=last_model_path,
                ext_gpu_id=gpu_id,
            )

            # 4. load the retrained model
            # model = model_archs.__dict__["resnet50"]()
            # checkpoint = torch.load(f"{last_model_path}_checkpoint.pth")
            model = torch.load(f"{last_model_path}_best_model.pth")
            model = torch.nn.DataParallel(model)
            # model.load_state_dict(checkpoint["state_dict"])
            # after_retrain_acc = evaluate_performance_imagenet(model, loaders["test"], gpu_id)
            accuarcies_between_prunesteps.append(after_retrain_acc)
            model = model.module.to("cpu")

    return model, accuarcies_between_prunesteps, last_model_path


print(f"Loading resnet50 model: {model_path}")

loaded_model = model_archs.__dict__["resnet50"]()
loaded_model = torch.nn.DataParallel(loaded_model)
checkpoint = torch.load(f"{model_path}.pth.tar")
loaded_model.load_state_dict(checkpoint["state_dict"])
loaded_model = loaded_model.module.to("cpu")
loaded_model = loaded_model.cuda(gpu_id)

print("Loading imagenet dataset ...")
loaders = get_imagenet_data_loader()
out_features = 1000
example_input = torch.randn(1, 3, 224, 224)

prune_params = {
    "prune_type": "l1",
    "sparsity": sparsity_in,
    "example_input": example_input,
    "out_features": out_features,
    "use_iter_prune": True,
    "prune_iter_steps": prune_iter_steps_in,
    "prune_iter_epochs": prune_iter_epochs_in,
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
    "loaders": loaders,
    "gpu_id": gpu_id,
}

model_accuracy_development = {}

# 0. original model accuracy
print("Starting to evaluate the original model performance ...")
original_acc = validate(loaders["test"], loaded_model, gpu_id)
model_accuracy_development["original_accuracy"] = original_acc

# 1. prune the model - possibly iteratively
print("Starting the (iterative) pruning ...")
"""
pruned_model, accuarcies_between_prunesteps, last_model_path = iterative_pruning(
    model=loaded_model,
    iter_num_epochs=prune_params.get("prune_iter_epochs"),
    prune_iter_steps=prune_params.get("prune_iter_steps"),
    prune_type=prune_params.get("prune_type"),
    sparsity=prune_params.get("sparsity"),
)"""
pruned_model, accuarcies_between_prunesteps, last_model_path = prune_structured_resnet50(
    net=loaded_model,
    loaders=loaders,
    prune_iter_epochs=prune_params.get("prune_iter_epochs"),
    example_inputs=example_input,
    out_features=out_features,
    prune_type=prune_params.get("prune_type"),
    gpu_id=gpu_id,
    sparsity=prune_params.get("sparsity"),
    prune_iter_steps=prune_params.get("prune_iter_steps"),
)
val_perf = accuarcies_between_prunesteps[-1]
model_accuracy_development["iterative_pruning"] = accuarcies_between_prunesteps
print(model_accuracy_development)
with open(
    f"./results_of_pruning_experiment/retraining_accuracies_{model_file}.json", "w"
) as outfile:
    json.dump(model_accuracy_development, outfile, indent=4)

print(f"Model pruning is done. Final accuracy: {val_perf}")


# 2. additional retraining of the model
# final_model_path = f"{model_file}_{prune_params.get('prune_iter_steps')}iter{prune_params.get('prune_iter_epochs')}_T{retrain_epochs}"
final_model_path = f"{last_model_path.split('.')[0]}_T{retrain_epochs}"
print(f"Starting additional training for {retrain_epochs} epochs ...")
after_retrain_acc = train_resnet50(
    num_epochs_to_train=retrain_epochs,
    dataset_path=dataset_path,
    checkpoint_path=last_model_path,
    result_model_path_=final_model_path,
    ext_gpu=gpu_id,
)
print(f"The final pruned and retrained model is stored in: {final_model_path}_best_model.pth")

model_accuracy_development["retraining"] = after_retrain_acc
with open(f"./results_of_pruning_experiment/all_accuracies_{model_file}.json", "w") as outfile:
    json.dump(model_accuracy_development, outfile, indent=4)

print(f"Done. Final accuracy is: {after_retrain_acc}")
