import copy
import json
import math

import torch
import torchvision.transforms as transforms
from fusion_utils import FusionType

# import main #from main import get_data_loader, test
from models import get_model, get_pretrained_models
from parameters import get_parameters
from performance_tester import (
    evaluate_performance_simple,
    float_format,
    fusion_test_manager,
    get_cifar10_data_loader,
    get_cifar100_data_loader,
    get_mnist_data_loader,
    get_result_skeleton,
    original_test_manager,
    pruning_test_manager,
    train_during_pruning,
    update_running_statistics,
    wrapper_first_fusion,
    wrapper_structured_pruning,
)
from pruning_modified import prune_unstructured
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import datetime
import logging

if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        level="INFO",
        filename=f"./logger_files/{date}_logger.txt",
        force=True,
    )

    logging.info(f"Loading experiment and dataset...")

    with open("./experiment_parameters.json", "r") as f:
        experiment_params = json.load(f)

    result_final = get_result_skeleton(experiment_params)

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

    fusion_function = wrapper_first_fusion(
        fusion_type=experiment_params["fusion_type"],
        train_loader=loaders["train"],
        gpu_id=experiment_params["gpu_id"],
        num_samples=experiment_params["num_samples"]
        if experiment_params["fusion_type"] != FusionType.WEIGHT
        else None,
    )
    pruning_function = (
        wrapper_structured_pruning  # still need to implement the structured pruning function
    )
    eval_function = evaluate_performance_simple

    new_result = {}
    for sparsity in result_final["experiment_parameters"]["sparsity"]:
        new_result["sparstiy"] = {"paf": None, "pruned": None, "pruned_fused": None, "paf": None}

    print(json.dumps(result_final, indent=4))

    for idx_result, result in enumerate(result_final["results"]):
        for model_dict in experiment_params["models"]:
            logging.info(f"Starting with sparsity: {result['sparsity']}.")

            print("new_result: ", new_result)
            name, diff_weight_init = model_dict["name"], experiment_params["diff_weight_init"]

            print(f"models/models_{name}/{name}_diff_weight_init_{diff_weight_init}_{0}.pth")
            models_original = get_pretrained_models(
                name,
                model_dict["basis_name"],
                experiment_params["gpu_id"],
                experiment_params["num_models"],
                output_dim=output_dim,
            )

            print(type(models_original[0]))

            params = {}
            params["pruning_function"] = pruning_function
            params["fusion_function"] = fusion_function
            params["eval_function"] = eval_function
            params["loaders"] = loaders
            params["gpu_id"] = experiment_params["gpu_id"]

            original_model_accuracies = original_test_manager(
                input_model_list=models_original, **params
            )
            print("original_model_accuracies ")
            print(original_model_accuracies)
            for i in range(len(original_model_accuracies)):
                result[name][f"model_{i}"]["accuracy_original"] = float_format(
                    original_model_accuracies[i]
                )

            prune_params = {
                "prune_type": result["prune_type"],
                "sparsity": result["sparsity"],
                "num_epochs": experiment_params["num_epochs"],
                "example_input": torch.randn(1, 1, 28, 28)
                if "cnn" in name
                else torch.randn(1, 3, 32, 32),
                "out_features": output_dim,
                "loaders": loaders,
                "gpu_id": experiment_params["gpu_id"],
            }

            if experiment_params["num_epochs"] > 0:
                logging.info("(PaT) Pruning and retraining the original models...")
            else:
                logging.info(f"(P) Pruning the original models...")
            pruned_models, pruned_model_accuracies, _ = pruning_test_manager(
                input_model_list=models_original, prune_params=prune_params, **params
            )
            for i in range(len(pruned_model_accuracies)):
                # torch.save(pruned_models[i].state_dict(), "models/{}_pruned_{}_.pth".format(name, i))
                pruned_models[i], epoch_accuracy = train_during_pruning(
                    copy.deepcopy(pruned_models[i]),
                    loaders=loaders,
                    num_epochs=experiment_params["num_epochs"],
                    gpu_id=experiment_params["gpu_id"],
                    prune=False,
                )

                s = int(result["sparsity"] * 100)
                n_epochs = experiment_params["num_epochs"]
                torch.save(
                    pruned_models[i].state_dict(),
                    f"./models/models_{name}/{name}_pruned_{s}_{n_epochs}",
                )
                pruned_model_accuracies[i] = epoch_accuracy[-1]

                _, epoch_accuracy_1 = train_during_pruning(
                    copy.deepcopy(pruned_models[i]),
                    loaders=loaders,
                    num_epochs=experiment_params["num_epochs"],
                    gpu_id=experiment_params["gpu_id"],
                    prune=False,
                    performed_epochs=experiment_params["num_epochs"],
                )
                epoch_accuracy = epoch_accuracy[:-1]
                epoch_accuracy.extend(epoch_accuracy_1)
                for idx, accuracy in enumerate(epoch_accuracy):
                    result[name][f"model_{i}"]["accuracy_pruned"][idx] = float_format(accuracy)

            fusion_params = get_parameters()
            fusion_params.model_name = name

            if experiment_params["FaP"]:
                if experiment_params["num_epochs"] > 0:
                    logging.info(f"(FaT) Fusing the original models...")
                else:
                    logging.info(f"(F) Fusing the original models...")
                fused_model, fused_model_accuracy, _ = fusion_test_manager(
                    input_model_list=models_original,
                    **params,
                    accuracies=original_model_accuracies,
                    num_epochs=experiment_params["num_epochs"],
                    name=name,
                )
                fused_model, fused_model_accuracy_re = train_during_pruning(
                    fused_model,
                    loaders=loaders,
                    num_epochs=experiment_params["num_epochs"],
                    gpu_id=experiment_params["gpu_id"],
                    prune=False,
                )
                if experiment_params["num_epochs"] > 0:
                    result[name]["accuracy_fused"] = float_format(fused_model_accuracy_re[-1])
                else:
                    result[name]["accuracy_fused"] = float_format(fused_model_accuracy)

                # experimental(new_result, result["sparsity"], models_original, original_model_accuracies, pruned_models, pruned_model_accuracies, get_parameters(), loaders, experiment_params["gpu_id"], name, params, fused_model)
                # break

            if experiment_params["SSF"]:
                for i in range(len(pruned_models)):
                    (
                        pruned_and_fused_model,
                        pruned_and_fused_model_accuracy,
                        _,
                    ) = fusion_test_manager(
                        input_model_list=[pruned_models[i], models_original[i]],
                        **params,
                        num_epochs=experiment_params["num_epochs"],
                        name=name,
                    )
                    m, epoch_accuracy = train_during_pruning(
                        copy.deepcopy(pruned_and_fused_model),
                        loaders=loaders,
                        num_epochs=experiment_params["num_epochs"],
                        gpu_id=experiment_params["gpu_id"],
                        prune=False,
                    )
                    for idx, accuracy in enumerate(epoch_accuracy):
                        result[name][f"model_{i}"]["accuracy_SSF"][idx] = float_format(accuracy)

            if experiment_params["PaF"]:
                if experiment_params["num_epochs"] > 0:
                    logging.info(f"(PaTaFaT) Fusing the pruned models...")
                else:
                    logging.info(f"(PaF) Fusing the pruned models...")
                paf_model, paf_model_accuracy, _ = fusion_test_manager(
                    input_model_list=pruned_models,
                    **params,
                    num_epochs=experiment_params["num_epochs"],
                    name=name,
                )
                m, epoch_accuracy = train_during_pruning(
                    paf_model,
                    loaders=loaders,
                    num_epochs=experiment_params["num_epochs"],
                    gpu_id=experiment_params["gpu_id"],
                    prune=False,
                )
                for idx, accuracy in enumerate(epoch_accuracy):
                    result[name]["accuracy_PaF"][idx] = float_format(accuracy)

            # PaF_all does the following: fuses following networks: pruned_model[0], original_model[0], original_model[1], ..., original_model[-1]
            # PaF_all achieves higher accuracy than PaF, but when we finetune PaF achieves higher accuracy
            if experiment_params["PaF_all"]:
                paf_all_model, paf_all_model_accuracy, _ = fusion_test_manager(
                    input_model_list=[
                        *models_original,
                        pruned_models[0]
                        if pruned_model_accuracies[0] > pruned_model_accuracies[1]
                        else pruned_models[1],
                    ],
                    **params,
                    num_epochs=experiment_params["num_epochs"],
                    name=name,
                )
                m, epoch_accuracy = train_during_pruning(
                    paf_all_model,
                    loaders=loaders,
                    num_epochs=experiment_params["num_epochs"],
                    gpu_id=experiment_params["gpu_id"],
                    prune=False,
                )
                for idx, accuracy in enumerate(epoch_accuracy):
                    result[name]["accuracy_PaF_all"][idx] = float_format(accuracy)

            if experiment_params["FaP"]:
                if experiment_params["num_epochs"] > 0:
                    logging.info(f"(FaTaPaT) Pruning the fused model...")
                else:
                    logging.info(f"(FaP) Pruning the fused model...")
                fap_models, fap_model_accuracies, _ = pruning_test_manager(
                    input_model_list=[fused_model], prune_params=prune_params, **params
                )
                m, epoch_accuracy = train_during_pruning(
                    fap_models[0],
                    loaders=loaders,
                    num_epochs=experiment_params["num_epochs"],
                    gpu_id=experiment_params["gpu_id"],
                    prune=False,
                )
                for idx, accuracy in enumerate(epoch_accuracy):
                    result[name]["accuracy_FaP"][idx] = float_format(accuracy)

            if experiment_params["IntraFusion"]:
                for i in range(len(pruned_models)):
                    intra_fusion_model = MSF(
                        models_original[i], gpu_id=-1, resnet=False, sparsity=result["sparsity"]
                    )
                    m, epoch_accuracy = train_during_pruning(
                        intra_fusion_model,
                        loaders=loaders,
                        num_epochs=experiment_params["num_epochs"] * 2,
                        gpu_id=experiment_params["gpu_id"],
                        prune=False,
                    )
                    # intra_fusion_model, _,_ = fusion_test_manager(input_model_list=[intra_fusion_model, models_original[i]], **params, num_epochs = experiment_params["num_epochs"], name=name)
                    # m,epoch_accuracy = train_during_pruning(intra_fusion_model, loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"], prune=False)
                    for idx, accuracy in enumerate(epoch_accuracy):
                        result[name][f"model_{i}"]["accuracy_IntraFusion"][idx] = float_format(
                            accuracy
                        )

            # Following code creates entries for our multi-sparsity fusion approach
            if experiment_params["MSF"]:
                for i in range(len(pruned_models)):
                    models_sparsities = []
                    models_sparsities_accuracies = []
                    sparsity_iter = result["sparsity"]
                    while sparsity_iter >= 0.1:
                        prune_params = {
                            "prune_type": result["prune_type"],
                            "sparsity": sparsity_iter,
                            "num_epochs": 0,
                            "example_input": torch.randn(1, 1, 28, 28)
                            if "cnn" in name
                            else torch.randn(1, 3, 32, 32),
                            "out_features": 10,
                            "loaders": loaders,
                            "gpu_id": experiment_params["gpu_id"],
                        }

                        pruned_models_new, pruned_models_new_accuracies, _ = pruning_test_manager(
                            input_model_list=[models_original[i]],
                            prune_params=prune_params,
                            **params,
                        )
                        models_sparsities.append(pruned_models_new[0])
                        models_sparsities_accuracies.append(pruned_models_new_accuracies[0])
                        sparsity_iter -= 0.1

                    models_sparsities.append(models_original[i])
                    model_sparsity, model_sparsity_accuracy, _ = fusion_test_manager(
                        input_model_list=models_sparsities,
                        **params,
                        num_epochs=experiment_params["num_epochs"],
                        name=name,
                    )
                    m, epoch_accuracy = train_during_pruning(
                        copy.deepcopy(model_sparsity),
                        loaders=loaders,
                        num_epochs=experiment_params["num_epochs"],
                        gpu_id=experiment_params["gpu_id"],
                        prune=False,
                    )
                    for idx, accuracy in enumerate(epoch_accuracy):
                        result[name][f"model_{i}"]["MSF"][idx] = float_format(accuracy)

            logging.info(f"Done with sparsity: {result['sparsity']}.")
        print(json.dumps(result_final, indent=4))
        result_final["results"][idx_result] = result

        with open(
            f"./results_and_plots_o/fullDict_results_resnet18/results_s{int(result['sparsity']*100)}_re{experiment_params['num_epochs']}.json",
            "w",
        ) as outfile:
            json.dump(result, outfile, indent=4)

    with open(
        f"./results_and_plots_o/fullDict_results_resnet18_withBN/results_sAll_re{experiment_params['num_epochs']}.json",
        "w",
    ) as outfile:
        json.dump(result_final, outfile, indent=4)
