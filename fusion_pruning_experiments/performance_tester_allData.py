import copy
import datetime
import json
import logging
import math

import torch
import torchvision.transforms as transforms
from model_caching import (
    ensure_folder_existence,
    file_already_exists,
    get_model_trainHistory,
    model_already_exists,
    save_experiment_results,
    save_model,
    save_model_trainHistory,
)
from parameters import get_parameters
from performance_tester_dirty import (
    evaluate_performance_simple,
    get_cifar10_data_loader,
    get_cifar100_data_loader,
    get_mnist_data_loader,
    get_result_skeleton,
    original_test_manager,
    pruning_test_manager,
    train_during_pruning,
    wrapper_first_fusion,
    wrapper_structured_pruning,
)
from torchvision import datasets
from torchvision.transforms import ToTensor

# import main #from main import get_data_loader, test
from models import get_pretrained_model_by_name, get_pretrained_models


def float_format(number):
    return float("{:.3f}".format(number))


if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        level="INFO",
        filename=f"./fusion_pruning_experiments/logger_files/{date}_logger.txt",
    )

    logging.info(f"Loading experiment and dataset...")

    with open("./fusion_pruning_experiments/experiment_parameters.json", "r") as f:
        experiment_params = json.load(f)
    num_epochs = experiment_params["num_epochs"]
    use_caching = experiment_params["use_caching"]
    gpu_id = experiment_params["gpu_id"]
    prune_iter_steps = experiment_params["prune_iter_steps"]
    prune_iter_epochs = experiment_params["prune_iter_epochs"]
    use_iter_prune = experiment_params["use_iter_prune"]
    name = experiment_params["models"][0]["name"]
    use_iterative_pruning = True if (prune_iter_epochs > 0 and prune_iter_steps > 1) else False
    if use_iter_prune and prune_iter_epochs > 0:
        logging.info(
            f"Working with iterative pruning: {prune_iter_steps} steps with each {prune_iter_epochs} epochs retraining."
        )
    else:
        logging.info(f"Working with direct pruning (NOT iterative pruning).")
    iterprune_text = (
        f"{experiment_params['prune_iter_steps']}iter{experiment_params['prune_iter_epochs']}"
        if use_iterative_pruning
        else ""
    )

    # Load needed dataset
    loaders = None
    output_dim = None
    if experiment_params["dataset"] == "mnist":
        loaders = get_mnist_data_loader()
        output_dim = 10
    elif experiment_params["dataset"] == "cifar10":
        loaders = get_cifar10_data_loader()
        output_dim = 10
    elif experiment_params["dataset"] == "cifar100":
        loaders = get_cifar100_data_loader()
        output_dim = 100
    else:
        raise Exception("Provided dataset does not exist.")

    # Setup experiment parameters
    pruning_function = wrapper_structured_pruning
    eval_function = evaluate_performance_simple
    fusion_function = wrapper_first_fusion

    params = {
        "pruning_function": pruning_function,
        "fusion_function": fusion_function,
        "eval_function": eval_function,
        "loaders": loaders,
        "gpu_id": experiment_params["gpu_id"],
    }

    # Load the model that should be pruned and retraiend
    model_path = f"./fusion_pruning_experiments/models/{experiment_params['models'][0]['basis_name']}_fullData"
    logging.info(f"Loading model: {model_path}.")
    print(f"Loading model: {model_path}.")
    models_original = [get_pretrained_model_by_name(model_path, gpu_id=experiment_params["gpu_id"])]

    # Results dictionary
    result = {"experiment_parameters": experiment_params}

    # Compute original model accuracy
    original_model_accuracies = original_test_manager(input_model_list=models_original, **params)
    result["accuracy_original"] = original_model_accuracies[0]

    # iterate over sparsities and do prune + retraining
    sparsity_list = experiment_params["sparsity"]
    for sp in sparsity_list:
        logging.info(f"Starting with sparsity: {sp}.")
        result[sp] = {}

        MODELS_CACHING_PATH = f"./models/models_{name}/cached_models"
        ensure_folder_existence(MODELS_CACHING_PATH)
        input_model_name = f"{MODELS_CACHING_PATH}/{model_path.rsplit('/')[-1]}"

        prune_params = {
            "prune_type": experiment_params["prune_type"][0],
            "sparsity": sp,
            "num_epochs": experiment_params["num_epochs"],
            "example_input": torch.randn(1, 1, 28, 28)
            if "cnn" in name
            else torch.randn(1, 3, 32, 32),
            "use_iter_prune": experiment_params["use_iter_prune"],
            "prune_iter_steps": experiment_params["prune_iter_steps"],
            "prune_iter_epochs": experiment_params["prune_iter_epochs"],
            "out_features": output_dim,
            "loaders": loaders,
            "gpu_id": experiment_params["gpu_id"],
        }

        logging.info("(P) Pruning the original models...")

        ##### NEW
        assert num_epochs > 0
        pruned_model_accuracy = None
        epoch_accuracy = None
        if num_epochs > 0:
            pruned_model_path = (
                f"{input_model_name}_s{int(sp*100)}_P{iterprune_text}_T{int(num_epochs*2)}"
            )
            if model_already_exists(pruned_model_path, loaders, gpu_id, use_caching):
                logging.info(f"\t\tFound the model {pruned_model_path}.pth in cache.")
                pat_model = get_pretrained_model_by_name(pruned_model_path, gpu_id)
                epoch_accuracy = get_model_trainHistory(pruned_model_path)
            else:
                pruned_models, pruned_model_accuracies, _ = pruning_test_manager(
                    input_model_list=models_original, prune_params=prune_params, **params
                )
                pat_model, epoch_accuracy = train_during_pruning(
                    copy.deepcopy(pruned_models[0]),
                    loaders=loaders,
                    num_epochs=experiment_params["num_epochs"] * 2,
                    gpu_id=experiment_params["gpu_id"],
                    prune=False,
                )
                if use_caching:
                    save_model(pruned_model_path, pat_model)
                    save_model_trainHistory(pruned_model_path, epoch_accuracy)
            pruned_model_accuracy = epoch_accuracy[-1]
            epoch_accuracy = epoch_accuracy[:-1]
        ##### END OF NEW
        """
        if experiment_params["num_epochs"] > 0:
            logging.info("(PaT) Retraining the pruned models...")
            _, epoch_accuracy = train_during_pruning(
                copy.deepcopy(pruned_models[0]),
                loaders=loaders,
                num_epochs=experiment_params["num_epochs"],
                gpu_id=experiment_params["gpu_id"],
                prune=False,
            )
            pruned_model_accuracy = epoch_accuracy[-1]
            epoch_accuracy = epoch_accuracy[:-1]
        """

        n_epochs = experiment_params["num_epochs"]
        result[sp]["accuracy_pruned"] = pruned_model_accuracy
        result[sp]["accuracy_pruned_epochs"] = epoch_accuracy
        logging.info(f"Done with sparsity: {sp}.")

    with open(
        f"./results_and_plots_o/fullDict_wholeData_{name}/results_sAll_re{experiment_params['num_epochs']*2}_wholeDataModel.json",
        "w",
    ) as outfile:
        json.dump(result, outfile, indent=4)
