import copy
import json
import math
import os

import torch
import torchvision.transforms as transforms
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
    get_result_skeleton,
    original_test_manager,
    pruning_test_manager,
    train_during_pruning,
    update_running_statistics,
    wrapper_first_fusion,
    wrapper_structured_pruning,
)
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import datetime
import logging

from fusion import MSF

if __name__ == "__main__":
    # introducing logger
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        level="INFO",
        filename=f"./logger_files/{date}_logger.txt",
        force=True,
    )

    # loading experiments and dataset
    logging.info(f"Loading experiment and dataset...")
    with open("./experiment_parameters.json", "r") as f:
        experiment_params = json.load(f)
    num_epochs = experiment_params["num_epochs"]
    use_caching = experiment_params["use_caching"]
    gpu_id = experiment_params["gpu_id"]
    prune_iter_steps = experiment_params["prune_iter_steps"]
    use_iter_prune = experiment_params["use_iter_prune"]
    use_intrafusion_for_pruning = experiment_params["use_intrafusion_for_pruning"]
    use_iterative_pruning = (
        True
        if (
            experiment_params["prune_iter_epochs"] > 0
            and experiment_params["prune_iter_steps"] > 1
            and use_iter_prune
        )
        else False
    )
    prune_iter_epochs = experiment_params["prune_iter_epochs"]
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
    if use_intrafusion_for_pruning:
        iterprune_text = "intra" + iterprune_text

    result_final = get_result_skeleton(experiment_params)
    loaders = None
    output_dim = None
    dataset_name = experiment_params["dataset"]
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
    else:
        raise Exception("Provided dataset does not exist.")

    # defining fusion/pruning/eval function to be used
    fusion_function = wrapper_first_fusion(
        fusion_type=experiment_params["fusion_type"],
        train_loader=loaders["train"],
        gpu_id=gpu_id,
        num_samples=experiment_params["num_samples"]
        if experiment_params["fusion_type"] != FusionType.WEIGHT
        else None,
    )
    pruning_function = wrapper_structured_pruning
    eval_function = (
        evaluate_performance_imagenet
        if experiment_params["dataset"] == "imagenet"
        else evaluate_performance_simple
    )

    # collecting the experiment settings into a dict
    params = {}
    params["pruning_function"] = pruning_function
    params["fusion_function"] = fusion_function
    params["eval_function"] = eval_function
    params["loaders"] = loaders
    params["gpu_id"] = gpu_id

    # setting skeleton for json that collects experiment results
    new_result = {}
    for sparsity in result_final["experiment_parameters"]["sparsity"]:
        new_result["sparstiy"] = {"paf": None, "pruned": None, "pruned_fused": None, "paf": None}

    # loading the models that are going to be worked with
    logging.info("Loading basis models:")
    original_model_name, diff_weight_init = (
        experiment_params["models"][0]["name"],
        experiment_params["diff_weight_init"],
    )
    original_model_basis_name = experiment_params["models"][0]["basis_name"]
    models_original = get_pretrained_models(
        original_model_name,
        original_model_basis_name,
        gpu_id,
        experiment_params["num_models"],
        output_dim=output_dim,
    )

    for idx in range(experiment_params["num_models"]):
        logging.info(f"\tLoaded the model: ./models/{original_model_basis_name}_{idx}.pth")

    # measuring the performance of the original models
    logging.info("Basis Model Accuracies:")
    original_model_accuracies = original_test_manager(input_model_list=models_original, **params)
    logging.info(f"\t{original_model_accuracies}")

    for idx_result, result in enumerate(result_final["results"]):
        logging.info("")
        logging.info(f"Starting with sparsity: {result['sparsity']}.")
        for model_dict in experiment_params["models"]:
            name, diff_weight_init = model_dict["name"], experiment_params["diff_weight_init"]

            # TODO: set example_input to work for resnet50!
            prune_params = {
                "prune_type": result["prune_type"],
                "sparsity": result["sparsity"],
                "example_input": torch.randn(1, 1, 28, 28)
                if "cnn" in name
                else (
                    torch.randn(1, 3, 224, 224) if "resnet50" in name else torch.randn(1, 3, 32, 32)
                ),
                "out_features": output_dim,
                "use_iter_prune": use_iter_prune,
                "prune_iter_steps": prune_iter_steps,
                "prune_iter_epochs": prune_iter_epochs,
                "loaders": loaders,
                "gpu_id": gpu_id,
                "model_name": name,
                "use_intrafusion_for_pruning": use_intrafusion_for_pruning,
            }

            # Write original performance into results json
            for i in range(len(original_model_accuracies)):
                result[name][f"model_{i}"]["accuracy_original"] = float_format(
                    original_model_accuracies[i]
                )

            # describe the needed model with a name
            MODELS_CACHING_PATH = f"./models/models_{name}/cached_models"
            ensure_folder_existence(MODELS_CACHING_PATH)
            input_model_names = [
                f"{MODELS_CACHING_PATH}/{model_dict['basis_name'].rsplit('/')[-1]}_{idx}"
                for idx in range(experiment_params["num_models"])
            ]
            if num_epochs > 0:
                logging.info(f"(PaT) Pruning and retraining ({num_epochs}) the original models...")
                pruned_model_paths = [
                    f"{in_mo}_s{int(result['sparsity']*100)}_P{iterprune_text}_T{int(num_epochs)}"
                    for in_mo in input_model_names
                ]
            else:
                logging.info(f"(P) Pruning the original models...")
                pruned_model_paths = [
                    f"{in_mo}_s{int(result['sparsity']*100)}_P{iterprune_text}"
                    for in_mo in input_model_names
                ]

            # P/PaT: get the Pruned (P) or PrunedAndTrained (PaT) version of the input-models
            pruned_models = []
            pruned_model_accuracies = []
            pruned_model_train_accuracies = []
            assert len(pruned_model_paths) == len(models_original)
            for k, this_model_path in enumerate(pruned_model_paths):
                logging.info(f"\tModel {k}:")
                if model_already_exists(this_model_path, loaders, gpu_id, use_caching):
                    logging.info(f"\t\tFound the model {this_model_path}.pth in cache.")
                    this_pruned_model = get_pretrained_model_by_name(this_model_path, gpu_id)
                    this_load_trainhist = get_model_trainHistory(this_model_path)
                    this_pruned_model_accuracy = this_load_trainhist[-1]
                    this_pruned_model_train_accuracies = this_load_trainhist[:-1]
                else:
                    this_original_model = [models_original[k]]
                    this_pruned_model_lis, this_pruned_model_accuracies, _ = pruning_test_manager(
                        input_model_list=this_original_model, prune_params=prune_params, **params
                    )
                    this_pruned_model_accuracies = this_pruned_model_accuracies[0]
                    this_pruned_model, epoch_accuracy = train_during_pruning(
                        copy.deepcopy(this_pruned_model_lis[0]),
                        loaders=loaders,
                        num_epochs=num_epochs,
                        gpu_id=gpu_id,
                        prune=False,
                        model_name=name,
                    )
                    # this_pruned_model_accuracies = this_pruned_model_accuracies
                    # this_pruned_model_accuracies.extend(epoch_accuracy)
                    this_pruned_model_accuracies = epoch_accuracy
                    this_pruned_model = this_pruned_model_lis[0]
                    if use_caching:
                        save_model(this_model_path, this_pruned_model)
                        save_model_trainHistory(this_model_path, this_pruned_model_accuracies)
                    this_pruned_model_accuracy = this_pruned_model_accuracies[-1]
                    this_pruned_model_train_accuracies = this_pruned_model_accuracies[:-1]

                # train for another num_epochs epochs to create the prune benchmark performance
                pruned_model_further_trained_path = (
                    f"{this_model_path.rsplit('_', 1)[0]}_T{int(2*num_epochs)}"
                )
                pruned_epoch_accuracy = this_pruned_model_train_accuracies
                # compute the benchmark pruned model with additional retrain epochs
                if num_epochs > 0 and experiment_params["compute_prune_baseline"]:
                    if model_already_exists(
                        pruned_model_further_trained_path, loaders, gpu_id, use_caching
                    ):
                        # pruned_model_further_trained = get_model(pruned_model_further_trained_path)
                        logging.info(
                            f"\t\tFound the model {pruned_model_further_trained_path}.pth in cache."
                        )
                        epoch_acc_further = get_model_trainHistory(
                            pruned_model_further_trained_path
                        )
                    else:
                        (
                            pruned_model_further_trained,
                            epoch_accuracy_further_tr,
                        ) = train_during_pruning(
                            copy.deepcopy(this_pruned_model),
                            loaders=loaders,
                            num_epochs=num_epochs,
                            gpu_id=gpu_id,
                            prune=False,
                            performed_epochs=num_epochs,
                            model_name=name,
                        )
                        epoch_acc_further = this_pruned_model_train_accuracies
                        epoch_acc_further.extend(epoch_accuracy_further_tr)
                        if use_caching:
                            save_model(
                                pruned_model_further_trained_path, pruned_model_further_trained
                            )
                            # ATTENTION: the trainhistory will only show the best performance during the further retraining at [-1]
                            save_model_trainHistory(
                                pruned_model_further_trained_path, epoch_acc_further
                            )
                    pruned_epoch_accuracy = epoch_acc_further
                # store the performance development of the retraining of the pruned model in result
                for idx, accuracy in enumerate(pruned_epoch_accuracy):
                    result[name][f"model_{k}"]["accuracy_pruned"][idx] = float_format(accuracy)
                # here we should maybe also add the this_load_trainhist to result[PaF]
                # so it does not only contain the fusion retrain accuracies, but its whole performance development in one dict entry
                pruned_models.append(this_pruned_model)
                pruned_model_accuracies.append(this_pruned_model_accuracy)
                pruned_model_train_accuracies.append(this_pruned_model_train_accuracies)

            fusion_params = get_parameters()
            fusion_weights = experiment_params["fusion_weights"][0]
            fusion_params.model_name = name

            # PaF: pruning the half-data models individually and fusing the results
            if experiment_params["PaF"]:
                if experiment_params["num_epochs"] > 0:
                    logging.info(
                        f"(PaTaFaT) Fusing and retraining ({num_epochs}) the pruned models..."
                    )
                    paf_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(result['sparsity']*100)}_P{iterprune_text}_aT{int(num_epochs)}aFaT{int(num_epochs)}"
                else:
                    logging.info(f"(PaF) Fusing the pruned models...")
                    paf_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(result['sparsity']*100)}_P{iterprune_text}_aF"

                if model_already_exists(paf_model_path, loaders, gpu_id, use_caching):
                    logging.info(f"\t\tFound the model {paf_model_path}.pth in cache.")
                    paf_model = get_pretrained_model_by_name(this_model_path, gpu_id)
                    paf_accuracy = get_model_trainHistory(this_model_path)
                else:
                    paf_model, paf_model_accuracy, _ = fusion_test_manager(
                        input_model_list=pruned_models,
                        **params,
                        num_epochs=0,
                        name=name,
                    )
                    m, paf_accuracy = train_during_pruning(
                        paf_model,
                        loaders=loaders,
                        num_epochs=num_epochs,
                        gpu_id=gpu_id,
                        prune=False,
                        model_name=name,
                    )
                    if use_caching:
                        save_model(paf_model_path, paf_model)
                        save_model_trainHistory(paf_model_path, paf_accuracy)
                for idx, accuracy in enumerate(paf_accuracy):
                    result[name]["accuracy_PaF"][idx] = float_format(accuracy)

            # FaP: fusing the half-data models into one and then pruning the result
            if experiment_params["FaP"]:
                # FaP - Step 1: fusing the half-data models
                if num_epochs > 0:
                    logging.info(f"(FaTaPaT) Fusing the original models and retraining...")
                    f_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(result['sparsity']*100)}_FaT{int(num_epochs)}"
                else:
                    logging.info(f"(FaP) Fusing the original models...")
                    f_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(result['sparsity']*100)}_F"

                if model_already_exists(f_model_path, loaders, gpu_id, use_caching):
                    logging.info(f"\t\tFound the model {f_model_path}.pth in cache.")
                    fused_model = get_pretrained_model_by_name(f_model_path, gpu_id)
                    fused_model_accuracy_re = get_model_trainHistory(f_model_path)
                else:
                    fused_model, fused_model_accuracy, _ = fusion_test_manager(
                        input_model_list=models_original,
                        **params,
                        accuracies=original_model_accuracies,
                        num_epochs=0,
                        name=name,
                    )
                    fused_model, fused_model_accuracy_re = train_during_pruning(
                        fused_model,
                        loaders=loaders,
                        num_epochs=num_epochs,
                        gpu_id=gpu_id,
                        prune=False,
                        model_name=name,
                    )
                    if use_caching:
                        save_model(f_model_path, fused_model)
                        save_model_trainHistory(f_model_path, fused_model_accuracy_re)

                # FaP - Step 2: pruning the result of the fusion
                if num_epochs > 0:
                    result[name]["accuracy_fused"] = float_format(fused_model_accuracy_re[-1])
                    fap_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(result['sparsity']*100)}_FaT{int(num_epochs)}_aP{iterprune_text}_aT{int(num_epochs)}"
                else:
                    # result[name]["accuracy_fused"] = float_format(fused_model_accuracy)
                    fap_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(result['sparsity']*100)}_F_aP{iterprune_text}"

                if model_already_exists(fap_model_path, loaders, gpu_id, use_caching):
                    logging.info(f"\t\tFound the model {fap_model_path}.pth in cache.")
                    fap_model = get_pretrained_model_by_name(fap_model_path, gpu_id)
                    fap_model_accuracies = get_model_trainHistory(fap_model_path)
                else:
                    fap_models, fap_model_accuracies, _ = pruning_test_manager(
                        input_model_list=[fused_model], prune_params=prune_params, **params
                    )
                    fap_model, epoch_accuracy = train_during_pruning(
                        fap_models[0],
                        loaders=loaders,
                        num_epochs=experiment_params["num_epochs"],
                        gpu_id=gpu_id,
                        prune=False,
                        model_name=name,
                    )

                    # fap_model_accuracies = fap_model_accuracies[0]
                    # fap_model_accuracies.extend(epoch_accuracy)
                    fap_model_accuracies = epoch_accuracy
                    if use_caching:
                        save_model(fap_model_path, fap_model)
                        save_model_trainHistory(fap_model_path, fap_model_accuracies)
                for idx, accuracy in enumerate(fap_model_accuracies):
                    result[name]["accuracy_FaP"][idx] = float_format(accuracy)

        result_final["results"][idx_result] = result

        # storing the experiment results
        model_name = experiment_params["models"][0]["name"]
        fusion_add_numsamples = (
            str(experiment_params["num_samples"])
            if experiment_params["fusion_type"] != FusionType.WEIGHT
            else ""
        )
        result_folder_name = f"./results_and_plots_o/fullDict_results_{experiment_params['fusion_type']}{fusion_add_numsamples}_{model_name}"
        ensure_folder_existence(result_folder_name)
        save_experiment_results(
            f"./results_and_plots_o/fullDict_results_{experiment_params['fusion_type']}{fusion_add_numsamples}_{model_name}/results_{iterprune_text}s{int(result['sparsity']*100)}_re{experiment_params['num_epochs']}",
            result,
        )
        logging.info(f"Done with sparsity: {result['sparsity']}.")

    logging.info("")
    logging.info("All experiments completed.")
    save_experiment_results(
        f"./results_and_plots_o/fullDict_results_{experiment_params['fusion_type']}{fusion_add_numsamples}_{model_name}/results_{iterprune_text}sAll_re{experiment_params['num_epochs']}",
        result_final,
    )
