import copy
import json
import logging

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

from models import get_model, get_pretrained_model_by_name, get_pretrained_models
from parameters import get_parameters
from performance_tester import (
    evaluate_performance_simple,
    fusion_test_manager,
    get_cifar10_data_loader,
    get_cifar100_data_loader,
    get_mnist_data_loader,
    original_test_manager,
    pruning_test_manager,
    train_during_pruning,
    wrapper_first_fusion,
    wrapper_structured_pruning,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def model_already_exists(model_path):
#    return bool(os.path.exists(f"{model_path}.pth"))


def file_already_exists(path):
    return bool(os.path.exists(path))


def save_model(model, path):
    save_to_path = f"{path}.pth"
    count = 0
    different_path_name = False
    while file_already_exists(save_to_path):
        save_to_path = f"{save_to_path.split('.')[0]}_{count}.pth"
        different_path_name = True
        count = count + 1
    torch.save(model, save_to_path)
    logging.info(f"\tStoring model to {save_to_path}")
    if different_path_name:
        logging.info(
            f"Warning: result file already existed. Saved to {save_to_path} instead. Did not check if overwrote another file with this name."
        )


def get_model_trainHistory(model_path):
    with open(f"{model_path}.json", "r") as f:
        train_hist = json.load(f)
    # logging.info(f"\t- Performance in last epochs was: {train_hist['train_epoch_perf'][-1]}")
    return train_hist["train_epoch_perf"]


def save_model_trainHistory(model_path, history):
    logging.info(f"\tStoring epoch performance during training to {model_path}.json")
    history_dict = {"train_epoch_perf": history}
    # logging.info(f"\t- Performance in last epoch was: {history[-1]}")
    with open(f"{model_path}.json", "w") as outfile:
        json.dump(history_dict, outfile, indent=4)


def get_correct_dataloader(dataset_name):
    if dataset_name == "mnist":
        return get_mnist_data_loader(), 10
    elif dataset_name == "cifar10":
        return get_cifar10_data_loader(), 10
    elif dataset_name == "cifar100":
        return get_cifar100_data_loader(), 100
    else:
        raise Exception("Provided dataset does not exist.")


def save_experiment_results(save_to_path, experiment_results_dict):
    count = 0
    different_path_name = False
    while file_already_exists(f"{save_to_path}.json"):
        save_to_path = f"{save_to_path}_{count}"
        different_path_name = True
        count = count + 1
    with open(f"{save_to_path}.json", "w") as outfile:
        json.dump(experiment_results_dict, outfile, indent=4)
    if different_path_name:
        logging.info(f"Warning: result file already existed. Saved to {save_to_path} instead.")


def model_already_exists(wanted_file_path, loaders, gpu_id):
    # Looking for a model that has a smaller number at the end of the name. Then retraining it.

    # checking if the file already exists
    if file_already_exists(f"{wanted_file_path}.pth"):
        return True

    # if the file doesnt exist and its not about "retraining" we return False
    # this means we cant take anything from the prefomputations.
    if not wanted_file_path[-1].isnumeric():
        return False

    # extract the amount of necessary epochs from the ending of wanted_file_path
    i = 1
    while wanted_file_path[-i].isnumeric():
        i += 1
    target_epochs = int(wanted_file_path[-i + 1 :])
    rest_of_path = wanted_file_path[: -i + 1]

    for k in reversed(range(1, target_epochs)):
        if file_already_exists(f"{rest_of_path}{k}.pth"):
            existing_model_path = f"{rest_of_path}{k}"
            logging.info(
                f"\tFound a model of the same type with less pretrained epochs: {existing_model_path}. Now retraining it for {target_epochs-k} epochs."
            )
            less_trained_model = get_pretrained_model_by_name(existing_model_path, gpu_id=gpu_id)
            # retrain the model for additional epochs
            retrained_model, retrained_model_accuracy = train_during_pruning(
                copy.deepcopy(less_trained_model),
                loaders=loaders,
                num_epochs=target_epochs - k,
                gpu_id=gpu_id,
                prune=False,  # unnecessary
            )
            retrained_model_accuracy = retrained_model_accuracy[:-1]
            # save the new model
            save_model(retrained_model, wanted_file_path)
            # extend the old models train accuracies
            less_trained_model_epochs = get_model_trainHistory(existing_model_path)
            less_trained_model_epochs.extend(retrained_model_accuracy)
            # less_trained_model_epochs.append(max(less_trained_model_epochs))
            save_model_trainHistory(wanted_file_path, less_trained_model_epochs)
            return True
    return False


def test_multiple_settings(RESULT_FILE_PATH):
    """
    Given a list of sparsity values (e.g. [0.9, 0.8]) and fusion weight combinations (e.g. [[0.5, 0.5], [0.6, 0.4]])
    this function "coordinates" the execution of experiments for all the possible combinations of setups.
    This is done by calling the function "test_one_setting()" on the inidividual combinations of the two parameters.

    Which parameters should be used and what to computer (e.g. PaF, FaP, PaTaF, ...) is defined in the experiment_parameters.json.

    ATTENTION: conditions on experiment_parameters.json
        1. fusion_weights: has to be set to a list of lists of weights. E.g. [[0.5, 0.5], [0.8, 0.2], [0.3, 0.7]] or for single one [[0.5, 0.5]]
        2. num_epochs: integer that defines how many epochs a possible retraining should be
        3. while multiple sparsities and fusion_weights can be handled, only ONE prune_type can be given.
        4. the following fields havt to exist and be set to boolean values:
            "FaP", "PaF", "PaF_all", "PaFaP", "PaFaT", "PaT", "PaTaF", "PaTaFaT", "FaPaT", "FaT", "FaTaP", "FaTaPaT"

    Results:
        The following .json files will be created in the following scenarios:
            - only one fusion_weight choice is given and possibly multiple sparsities:
                One single .json file that contains the different sparsities at fixed weights.
                The files name will end on f"_weight{int(fusion_weights[0]*100)}.json".

            - all other cases (multiple fusion weights with multiple sparsities, multiple fusion weights with a since sparsity):
                One .json file for each given sparsity. In each file the sparsity is fixed and the fusion_weights are varied.
                The files name will end on f"_sparsity{int(sparsity*100)}.json".
    """

    # 1. load the relevant experiment parameters
    with open("./experiment_parameters.json", "r") as f:
        experiment_params = json.load(f)

    # 2. figure out some basic experiment parameters and get the names of the models the experiment is performed on
    name = experiment_params["models"][0]["name"]
    MODELS_PATH = f"./models/models_{name}"

    basis_name = experiment_params["models"][0]["basis_name"]
    num_models = experiment_params["num_models"]
    gpu_id = experiment_params["gpu_id"]
    num_epochs = experiment_params["num_epochs"]
    dataset = experiment_params["dataset"]
    input_model_names = [f"{MODELS_PATH}/{basis_name}_{idx}" for idx in range(num_models)]

    # 3. running the experiments on different sparsity and fusion_weight combinations
    fusion_weights_list = experiment_params["fusion_weights"]
    sparsity_list = experiment_params["sparsity"]

    # 4. write the names of the models to the logger file
    logging.info("Starting process with models:")
    logging.info(f"{input_model_names[0]}")
    logging.info(f"{input_model_names[1]}\n")

    # 5. load the relevant datasets
    loaders, _ = get_correct_dataloader(dataset)

    # 6. load the pretrained models
    original_models = [
        get_pretrained_model_by_name(model_file_path, gpu_id)
        for model_file_path in input_model_names
    ]

    # 7. set parameters of fusion and pruning process
    fusion_function = wrapper_first_fusion
    pruning_function = wrapper_structured_pruning
    eval_function = evaluate_performance_simple
    params = {
        "pruning_function": pruning_function,
        "fusion_function": fusion_function,
        "eval_function": eval_function,
        "loaders": loaders,
        "gpu_id": gpu_id,
    }

    ### START COMPUTATIONS THAT ARE COMMON OVER VARIATIONS OF SPARSITY AND WEIGHTS ###
    # 8. Performance of: original models
    logging.info("Computing original models performances...")
    original_model_accuracies = original_test_manager(input_model_list=original_models, **params)
    performance_measurements = {"original_model_accuracies": original_model_accuracies}
    logging.info(f"\t- Performance: {original_model_accuracies}")

    # 9. Performance of: original models + extra training
    if num_epochs > 0:
        orig_model_retrained_accuracies_list = []
        logging.info(f"Training original models for additional {num_epochs} epochs...")
        for i, this_model in enumerate(original_models):
            orig_retrained_model_path = f"{input_model_names[i]}_re{num_epochs}"
            if model_already_exists(orig_retrained_model_path, loaders, gpu_id):
                logging.info(f"\tFound the model {orig_retrained_model_path}.pth in cache.")
                retrained_model = get_pretrained_model_by_name(orig_retrained_model_path, gpu_id)
                retrained_model_accuracy = original_test_manager([retrained_model], **params)
            else:
                retrained_model, retrained_model_accuracy = train_during_pruning(
                    copy.deepcopy(this_model),
                    loaders=loaders,
                    num_epochs=num_epochs,
                    gpu_id=gpu_id,
                    prune=False,  # unnecessary
                )
                retrained_model_accuracy = retrained_model_accuracy[:-1]
                save_model(retrained_model, orig_retrained_model_path)
                save_model_trainHistory(orig_retrained_model_path, retrained_model_accuracy)
            orig_model_retrained_accuracies_list.append(retrained_model_accuracy)
        performance_measurements[
            "original_model_retrained_accuracies_list"
        ] = orig_model_retrained_accuracies_list
        logging.info(f"\t- Performance: {[x[-1] for x in orig_model_retrained_accuracies_list]}\n")

    ### DONE WITH COMPUTATIONS THAT ARE COMMON OVER VARIATIONS OF SPARSITY AND WEIGHTS ###

    # 10. ITERATE THE FUSION_WEIGHTS AND SPARSITIES
    for sparsity in sparsity_list:
        for fusion_weights in fusion_weights_list:
            start_time = time.time()
            perf_meas_dict = test_one_setting(
                sparsity,
                fusion_weights,
                original_models,
                original_model_accuracies,
                experiment_params,
                performance_measurements,
                params,
                input_model_names,
            )
            end_time = time.time()
            execution_time = end_time - start_time

            perf_meas_dict["sparsity"] = sparsity
            perf_meas_dict["fusion_weights"] = fusion_weights
            perf_meas_dict["execution_time"] = execution_time
            perf_meas_dict["experiment_parameters.json"] = experiment_params

            RESULT_FILE_PATH_MOD = f"{MODELS_PATH}_experiments/{basis_name}_{dataset}_re{num_epochs}_w{int(fusion_weights[0]*100)}_s{int(sparsity*100)}"
            save_experiment_results(
                save_to_path=f"{RESULT_FILE_PATH_MOD}",
                experiment_results_dict=perf_meas_dict,
            )


def test_one_setting(
    sparsity,
    fusion_weights,
    original_models,
    original_model_accuracies,
    experiment_params,
    performance_measurements,
    params,
    input_model_names,
):
    """
    Given a single sparsity and fusion_weights setting, this function executes all experiments (e.g. PaF, FaP, ...) - besides MSF and SSF - that are set to true in the experiment_parameters.json.
    The resulting performance measures (including epoch-wise test performance for possible retraining) are returned in the dictionary "performance_measurements".

    The corresponding trained models are also available (see e.g. FaP_model, PaF_model, ...), but are not returned. This can be easily changed.
    """
    # Can only handle one sparsity type (e.g. "l1") at a time
    logging.info(f"Starting with sparsity: {sparsity}, fusion_weights: {fusion_weights}.")

    name, diff_weight_init, gpu_id, num_epochs = (
        experiment_params["models"][0]["name"],
        experiment_params["diff_weight_init"],
        experiment_params["gpu_id"],
        experiment_params["num_epochs"],
    )
    loaders = params["loaders"]

    out_features_ = 10
    if experiment_params["dataset"] == "cifar100":
        out_features_ = 100
    elif experiment_params["dataset"] in ["cifar10", "mnist"]:
        out_features_ = 10
    else:
        raise ValueError("Did not specify valid dataset")

    # FUSION AND PRUNING SPECIFICATIONS
    fusion_params = get_parameters()
    fusion_params.model_name = name
    # Can only handle one sparsity type at a time (takes the first if multiple in list)
    prune_type = (
        experiment_params["prune_type"]
        if isinstance(experiment_params["prune_type"], str)
        else experiment_params["prune_type"][0]
    )
    prune_params = {
        "prune_type": prune_type,
        "sparsity": sparsity,
        "num_epochs": num_epochs,
        "example_input": torch.randn(1, 1, 28, 28)
        if not ("vgg" in name)
        else torch.randn(1, 3, 32, 32),
        "out_features": out_features_,
        "loaders": loaders,
        "gpu_id": experiment_params["gpu_id"],
    }

    ### Combinations that start with pruning
    # P
    if (
        experiment_params["PaF"]
        or experiment_params["PaFaP"]
        or experiment_params["PaFaT"]
        or experiment_params["PaT"]
        or experiment_params["PaTaF"]
        or experiment_params["PaTaFaT"]
    ):
        logging.info(f"Pruning the original models...")
        pruned_model_0_path = f"{input_model_names[0]}_s{int(sparsity*100)}_P"
        pruned_model_1_path = f"{input_model_names[1]}_s{int(sparsity*100)}_P"
        if model_already_exists(pruned_model_0_path, loaders, gpu_id) and model_already_exists(
            pruned_model_1_path, loaders, gpu_id
        ):
            logging.info(f"\tFound the model {pruned_model_0_path}.pth in cache.")
            logging.info(f"\tFound the model {pruned_model_1_path}.pth in cache.")
            pruned_model_0 = get_pretrained_model_by_name(pruned_model_0_path, gpu_id)
            pruned_model_1 = get_pretrained_model_by_name(pruned_model_1_path, gpu_id)
            pruned_models = [pruned_model_0, pruned_model_1]
            original_pruned_models_accuracies = original_test_manager(pruned_models, **params)

        else:
            pruned_models, original_pruned_models_accuracies, _ = pruning_test_manager(
                input_model_list=copy.deepcopy(original_models), prune_params=prune_params, **params
            )
            save_model(pruned_models[0], pruned_model_0_path)
            save_model(pruned_models[1], pruned_model_1_path)
            pruned_model_0 = pruned_models[0]
            pruned_model_1 = pruned_models[1]
        performance_measurements[
            "original_pruned_models_accuracies"
        ] = original_pruned_models_accuracies
        logging.info(f"\t- Performance: {original_pruned_models_accuracies}")
        pruned_model_0.to("cpu")
        pruned_model_1.to("cpu")

    # PaF
    if experiment_params["PaF"] or experiment_params["PaFaP"] or experiment_params["PaFaT"]:
        logging.info(f"Computing PaF...")
        PaF_model_path = (
            f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(sparsity*100)}_PaF"
        )
        if model_already_exists(PaF_model_path, loaders, gpu_id):
            logging.info(f"\tFound the model {PaF_model_path}.pth in cache.")
            PaF_model = get_pretrained_model_by_name(PaF_model_path, gpu_id)
            PaF_model_accuracy = original_test_manager([PaF_model], **params)
        else:
            PaF_model, PaF_model_accuracy, _ = fusion_test_manager(
                input_model_list=copy.deepcopy(pruned_models),
                **params,
                num_epochs=num_epochs,  # unnecessary
                importance=fusion_weights,
            )
            save_model(PaF_model, PaF_model_path)
        performance_measurements["PaF_model_accuracy"] = PaF_model_accuracy
        logging.info(f"\t- Performance: {PaF_model_accuracy}")
        PaF_model.to("cpu")

    # PaFaP
    if experiment_params["PaFaP"]:
        logging.info(f"Computing PaFaP...")
        PaFaP_model_path = (
            f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(sparsity*100)}_PaFaP"
        )
        if model_already_exists(PaFaP_model_path, loaders, gpu_id):
            logging.info(f"\tFound the model {PaFaP_model_path}.pth in cache.")
            PaFaP_model = get_pretrained_model_by_name(PaFaP_model_path, gpu_id)
            PaFaP_model_accuracy = original_test_manager([PaFaP_model], **params)
        else:
            PaFaP_model, PaFaP_model_accuracy, _ = pruning_test_manager(
                input_model_list=copy.deepcopy([PaF_model]),
                prune_params=prune_params,
                **params,
            )
            PaFaP_model = PaFaP_model[0]
            save_model(PaFaP_model, PaFaP_model_path)
        performance_measurements["PaFaP_model_accuracy"] = PaFaP_model_accuracy[0]
        logging.info(f"\t- Performance: {PaFaP_model_accuracy[0]}")
        PaFaP_model.to("cpu")

    # PaFaT
    if num_epochs > 0 and experiment_params["PaFaT"]:
        logging.info(f"Computing PaFaT...")
        PaFaT_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(sparsity*100)}_PaFaT{num_epochs}"
        if model_already_exists(PaFaT_model_path, loaders, gpu_id):
            logging.info(f"\tFound the model {PaFaT_model_path}.pth in cache.")
            PaFaT_model = get_pretrained_model_by_name(PaFaT_model_path, gpu_id)
            PaFaT_model_epoch_accuracy = get_model_trainHistory(PaFaT_model_path)
        else:
            PaFaT_model, PaFaT_model_epoch_accuracy = train_during_pruning(
                copy.deepcopy(PaF_model),
                loaders=loaders,
                num_epochs=num_epochs,
                gpu_id=experiment_params["gpu_id"],
                prune=False,
            )
            PaFaT_model_epoch_accuracy = PaFaT_model_epoch_accuracy[:-1]
            save_model(PaFaT_model, PaFaT_model_path)
            save_model_trainHistory(PaFaT_model_path, PaFaT_model_epoch_accuracy)
        performance_measurements["PaFaT_model_epoch_accuracy"] = PaFaT_model_epoch_accuracy
        logging.info(f"\t- Performance: {PaFaT_model_epoch_accuracy[-1]}")
        PaFaT_model.to("cpu")

    # PaT
    if num_epochs > 0 and (
        experiment_params["PaT"] or experiment_params["PaTaF"] or experiment_params["PaTaFaT"]
    ):
        logging.info(f"Computing PaT...")
        PaT_model_0_path = f"{input_model_names[0]}_s{int(sparsity*100)}_PaT{int(num_epochs/2)}"
        PaT_model_1_path = f"{input_model_names[1]}_s{int(sparsity*100)}_PaT{int(num_epochs/2)}"
        PaT_model_list = []
        PaT_models_epoch_accuracies_list = []

        if model_already_exists(PaT_model_0_path, loaders, gpu_id) and model_already_exists(
            PaT_model_1_path, loaders, gpu_id
        ):
            logging.info(f"\tFound the model {PaT_model_0_path}.pth in cache.")
            logging.info(f"\tFound the model {PaT_model_1_path}.pth in cache.")
            PaT_model_0 = get_pretrained_model_by_name(PaT_model_0_path, gpu_id)
            PaT_model_1 = get_pretrained_model_by_name(PaT_model_1_path, gpu_id)
            PaT_model_list = [PaT_model_0, PaT_model_1]
            PaT_models_epoch_accuracies_list.extend(
                (
                    get_model_trainHistory(PaT_model_0_path),
                    get_model_trainHistory(PaT_model_1_path),
                )
            )
        else:
            lis = []
            for i in [0, 1]:
                PaT_model, PaT_model_epoch_accuracies = train_during_pruning(
                    copy.deepcopy(pruned_models[i]),
                    loaders=loaders,
                    num_epochs=int(experiment_params["num_epochs"] / 2),
                    gpu_id=experiment_params["gpu_id"],
                    prune=False,
                )
                lis.append(PaT_model)
                PaT_model_epoch_accuracies = PaT_model_epoch_accuracies[:-1]
                PaT_model_list.append(PaT_model)
                PaT_models_epoch_accuracies_list.append(PaT_model_epoch_accuracies)
                save_model(
                    PaT_model, f"{input_model_names[i]}_s{int(sparsity*100)}_PaT{int(num_epochs/2)}"
                )
                save_model_trainHistory(
                    f"{input_model_names[i]}_s{int(sparsity*100)}_PaT{int(num_epochs/2)}",
                    PaT_model_epoch_accuracies,
                )
            PaT_model_0 = lis[0]
            PaT_model_1 = lis[1]
        performance_measurements[
            "PaT_models_epoch_accuracies_list"
        ] = PaT_models_epoch_accuracies_list
        logging.info(f"\t- Performance: {[x[-1] for x in PaT_models_epoch_accuracies_list]}")
        PaT_model_0.to("cpu")
        PaT_model_1.to("cpu")

    # PaTaF
    if num_epochs > 0 and (experiment_params["PaTaF"] or experiment_params["PaTaFaT"]):
        logging.info(f"Computing PaTaF...")
        PaTaF_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(sparsity*100)}_PaT{int(num_epochs/2)}aF"
        if model_already_exists(PaTaF_model_path, loaders, gpu_id):
            logging.info(f"\tFound the model {PaTaF_model_path}.pth in cache.")
            PaTaF_model = get_pretrained_model_by_name(PaTaF_model_path, gpu_id)
            PaTaF_model_accuracy = original_test_manager([PaTaF_model], **params)
        else:
            PaTaF_model, PaTaF_model_accuracy, _ = fusion_test_manager(
                input_model_list=copy.deepcopy(PaT_model_list),
                **params,
                num_epochs=experiment_params["num_epochs"],
                importance=fusion_weights,
            )
            save_model(PaTaF_model, PaTaF_model_path)
        performance_measurements["PaTaF_model_accuracy"] = PaTaF_model_accuracy
        logging.info(f"\t- Performance: {PaTaF_model_accuracy}")
        PaTaF_model.to("cpu")

    # PaTaFaT
    if num_epochs > 0 and experiment_params["PaTaFaT"]:
        logging.info(f"Computing PaTaFaT...")
        PaTaFaT_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(sparsity*100)}_PaT{int(num_epochs/2)}aFaT{int(num_epochs/2)}"
        if model_already_exists(PaTaFaT_model_path, loaders, gpu_id):
            logging.info(f"\tFound the model {PaTaFaT_model_path}.pth in cache.")
            PaTaFaT_model = get_pretrained_model_by_name(PaTaFaT_model_path, gpu_id)
            PaTaFaT_model_accuracy = get_model_trainHistory(PaTaFaT_model_path)
        else:
            PaTaFaT_model, PaTaFaT_model_accuracy = train_during_pruning(
                copy.deepcopy(PaTaF_model),
                loaders=loaders,
                num_epochs=int(experiment_params["num_epochs"] / 2),
                gpu_id=experiment_params["gpu_id"],
                prune=False,
            )
            PaTaFaT_model_accuracy = PaTaFaT_model_accuracy[:-1]
            save_model(PaTaFaT_model, PaTaFaT_model_path)
            save_model_trainHistory(PaTaFaT_model_path, PaTaFaT_model_accuracy)
        performance_measurements["PaTaFaT_model_accuracy"] = PaTaFaT_model_accuracy
        logging.info(f"\t- Performance: {PaTaFaT_model_accuracy[-1]}")
        PaTaFaT_model.to("cpu")

    ### Combinations that start with fusion
    # F
    if (
        experiment_params["FaP"]
        or experiment_params["FaPaT"]
        or experiment_params["FaT"]
        or experiment_params["FaTaP"]
        or experiment_params["FaTaPaT"]
    ):
        logging.info(f"Fusing the original models...")
        F_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_F"
        if model_already_exists(F_model_path, loaders, gpu_id):
            logging.info(f"\tFound the model {F_model_path}.pth in cache.")
            original_fused_model = get_pretrained_model_by_name(F_model_path, gpu_id)
            original_fused_model_accuracy = original_test_manager([original_fused_model], **params)
        else:
            original_fused_model, original_fused_model_accuracy, _ = fusion_test_manager(
                input_model_list=copy.deepcopy(original_models),
                **params,
                accuracies=original_model_accuracies,
                num_epochs=experiment_params["num_epochs"],
                importance=fusion_weights,
            )
            save_model(original_fused_model, F_model_path)
        performance_measurements["original_fused_model_accuracy"] = original_fused_model_accuracy
        logging.info(f"\t- Performance: {original_fused_model_accuracy}")
        original_fused_model.to("cpu")

    # FaP
    if experiment_params["FaP"] or experiment_params["FaPaT"]:
        logging.info(f"Computing FaP...")
        FaP_model_path = (
            f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(sparsity*100)}_FaP"
        )
        if model_already_exists(FaP_model_path, loaders, gpu_id):
            logging.info(f"\tFound the model {FaP_model_path}.pth in cache.")
            FaP_model = get_pretrained_model_by_name(FaP_model_path, gpu_id)
            FaP_model_accuracy = original_test_manager([FaP_model], **params)
        else:
            FaP_model, FaP_model_accuracy, _ = pruning_test_manager(
                input_model_list=copy.deepcopy([original_fused_model]),
                prune_params=prune_params,
                **params,
            )
            FaP_model = FaP_model[0]
            save_model(FaP_model, FaP_model_path)
        performance_measurements["FaP_model_accuracy"] = FaP_model_accuracy
        logging.info(f"\t- Performance: {FaP_model_accuracy}")
        FaP_model.to("cpu")

    # FaPaT
    if num_epochs > 0 and experiment_params["FaPaT"]:
        logging.info(f"Computing FaPaT...")
        FaPaT_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(sparsity*100)}_FaPaT{num_epochs}"
        if model_already_exists(FaPaT_model_path, loaders, gpu_id):
            logging.info(f"\tFound the model {FaPaT_model_path}.pth in cache.")
            FaPaT_model = get_pretrained_model_by_name(FaPaT_model_path, gpu_id)
            FaPaT_model_epoch_accuracy = get_model_trainHistory(FaPaT_model_path)
        else:
            FaPaT_model, FaPaT_model_epoch_accuracy = train_during_pruning(
                copy.deepcopy(FaP_model),
                loaders=loaders,
                num_epochs=experiment_params["num_epochs"],
                gpu_id=experiment_params["gpu_id"],
                prune=False,
            )
            FaPaT_model_epoch_accuracy = FaPaT_model_epoch_accuracy[:-1]
            save_model(FaPaT_model, FaPaT_model_path)
            save_model_trainHistory(FaPaT_model_path, FaPaT_model_epoch_accuracy)
        performance_measurements["FaPaT_model_epoch_accuracy"] = FaPaT_model_epoch_accuracy
        logging.info(f"\t- Performance: {FaPaT_model_epoch_accuracy[-1]}")
        FaPaT_model.to("cpu")

    # FaT
    if num_epochs > 0 and (
        experiment_params["FaT"] or experiment_params["FaTaP"] or experiment_params["FaTaPaT"]
    ):
        logging.info(f"Computing FaT...")
        FaT_model_path = (
            f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_FaT{int(num_epochs/2)}"
        )
        if model_already_exists(FaT_model_path, loaders, gpu_id):
            logging.info(f"\tFound the model {FaT_model_path}.pth in cache.")
            FaT_model = get_pretrained_model_by_name(FaT_model_path, gpu_id)
            FaT_epoch_accuracies = get_model_trainHistory(FaT_model_path)
        else:
            FaT_model, FaT_epoch_accuracies = train_during_pruning(
                copy.deepcopy(original_fused_model),
                loaders=loaders,
                num_epochs=int(experiment_params["num_epochs"] / 2),
                gpu_id=experiment_params["gpu_id"],
                prune=False,
            )
            FaT_epoch_accuracies = FaT_epoch_accuracies[:-1]
            save_model(FaT_model, FaT_model_path)
            save_model_trainHistory(FaT_model_path, FaT_epoch_accuracies)
        performance_measurements["FaT_epoch_accuracies"] = FaT_epoch_accuracies
        logging.info(f"\t- Performance: {FaT_epoch_accuracies[-1]}")
        FaT_model.to("cpu")

    # FaTaP
    if num_epochs > 0 and (experiment_params["FaTaP"] or experiment_params["FaTaPaT"]):
        logging.info(f"Computing FaTaP...")
        FaTaP_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(sparsity*100)}_FaT{int(num_epochs/2)}aP"
        if model_already_exists(FaTaP_model_path, loaders, gpu_id):
            logging.info(f"\tFound the model {FaTaP_model_path}.pth in cache.")
            FaTaP_model = get_pretrained_model_by_name(FaTaP_model_path, gpu_id)
            FaTaP_model_accuracy = original_test_manager([FaTaP_model], **params)
        else:
            FaTaP_model, FaTaP_model_accuracy, _ = pruning_test_manager(
                input_model_list=copy.deepcopy([FaT_model]),
                prune_params=prune_params,
                **params,
            )
            FaTaP_model = FaTaP_model[0]
            save_model(FaTaP_model, FaTaP_model_path)
        performance_measurements["FaTaP_model_accuracy"] = FaTaP_model_accuracy
        logging.info(f"\t- Performance: {FaTaP_model_accuracy}")
        FaTaP_model.to("cpu")

    # FaTaPaT
    if num_epochs > 0 and experiment_params["FaTaPaT"]:
        logging.info(f"Computing FaTaPaT...")
        FaTaPaT_model_path = f"{input_model_names[0][:-2]}_w{int(fusion_weights[0]*100)}_s{int(sparsity*100)}_FaT{int(num_epochs/2)}aPaT{int(num_epochs/2)}"
        if model_already_exists(FaTaPaT_model_path, loaders, gpu_id):
            logging.info(f"\tFound the model {FaTaPaT_model_path}.pth in cache.")
            FaTaPaT_model = get_pretrained_model_by_name(FaTaPaT_model_path, gpu_id)
            FaTaPaT_model_accuracy = get_model_trainHistory(FaTaPaT_model_path)
        else:
            FaTaPaT_model, FaTaPaT_model_accuracy = train_during_pruning(
                copy.deepcopy(FaTaP_model),
                loaders=loaders,
                num_epochs=int(experiment_params["num_epochs"] / 2),
                gpu_id=experiment_params["gpu_id"],
                prune=False,
            )
            FaTaPaT_model_accuracy = FaTaPaT_model_accuracy[:-1]
            save_model(FaTaPaT_model, FaTaPaT_model_path)
            save_model_trainHistory(FaTaPaT_model_path, FaTaPaT_model_accuracy)
        performance_measurements["FaTaPaT_model_accuracy"] = FaTaPaT_model_accuracy
        logging.info(f"\t- Performance: {FaTaPaT_model_accuracy[-1]}")
        FaTaPaT_model.to("cpu")

    logging.info(f"Done with sparsity: {sparsity}, fusion_weights: {fusion_weights}.\n")

    return performance_measurements


import datetime
import os
import random
import time
from time import sleep

import numpy as np

if __name__ == "__main__":
    """
    See description of test_multiple_settings() for details on whats happening in this script.

    The format of output are .json files that are stored in "results_and_plots_o" in the subfolder specified by the model name.
    The location can be changed.
    Models are loaded from the below given files (without the .pth ending).
    """

    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(3)
    random.seed(3)
    os.environ["PYTHONHASHSEED"] = str(3)

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        level="INFO",
        filename=f"./logger_files/{date}_logger.txt",
        force=True,
    )

    # Loading experiment parameters from corresponding file
    with open("./experiment_parameters.json", "r") as f:
        experiment_params = json.load(f)
    model_name = experiment_params["models"][0]["name"]
    network_difference = "DiffInit" if experiment_params["diff_weight_init"] else "DiffData"

    # Define where the resulting .json file should be stored
    result_folder_name = f"./results_and_plots_o/fullDict_results_{model_name}"
    result_file_name = (
        f"TEST_this_experiments_result_nets{network_difference}"  # without .json ending!
    )
    result_file_path = f"{result_folder_name}/{result_file_name}"
    if not os.path.exists(result_folder_name):
        os.makedirs(result_folder_name)  # create folder if doesnt exist before

    # Path to the models that should be used (without .pth)
    input_model_names = ["", ""]

    test_multiple_settings(RESULT_FILE_PATH=result_file_path)
