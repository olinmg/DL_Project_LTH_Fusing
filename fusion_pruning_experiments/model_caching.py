import copy
import json
import logging
import os

import torch

from models import get_pretrained_model_by_name
from performance_tester import (
    get_cifar10_data_loader,
    get_cifar100_data_loader,
    get_mnist_data_loader,
    train_during_pruning,
)


def ensure_folder_existence(this_folder_path):
    # creates a folder at the given path, if it does not yet exist
    if not file_already_exists(this_folder_path):
        os.makedirs(this_folder_path)


def file_already_exists(path):
    return bool(os.path.exists(path))


def save_model(path, model):
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


def model_already_exists(wanted_file_path, loaders, gpu_id, use_caching):
    # Looking for a model that has a smaller number at the end of the name. Then retraining it.

    # if caching is turned of, this function will simply return false
    if not use_caching:
        return False

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
                f"\t\tFound a model of the same type with less retrained epochs: {existing_model_path}. Now retraining it for {target_epochs-k} epochs."
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
            save_model(wanted_file_path, retrained_model)
            # extend the old models train accuracies
            less_trained_model_epochs = get_model_trainHistory(existing_model_path)
            less_trained_model_epochs.extend(retrained_model_accuracy)
            # less_trained_model_epochs.append(max(less_trained_model_epochs))
            save_model_trainHistory(wanted_file_path, less_trained_model_epochs)
            return True
    return False
