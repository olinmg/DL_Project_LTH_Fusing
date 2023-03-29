import copy
import json
import math

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

# import main #from main import get_data_loader, test
from models import get_model, get_pretrained_models
from parameters import get_parameters
from pruning_modified import prune_unstructured

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist_data_loader():
    "Be aware about if the data should be shuffled or not!"

    mnist_train = datasets.MNIST("data", train=True, transform=ToTensor(), download=True)
    mnist_test = datasets.MNIST("data", train=False, transform=ToTensor(), download=True)

    loaders = {
        "train": torch.utils.data.DataLoader(
            mnist_train, batch_size=128, shuffle=True, num_workers=4
        ),
        "test": torch.utils.data.DataLoader(
            mnist_test, batch_size=128, shuffle=True, num_workers=4
        ),
    }
    return loaders


def get_cifar10_data_loader():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return {"train": train_loader, "test": val_loader}


def get_cifar100_data_loader():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return {"train": train_loader, "test": val_loader}


def evaluate_performance_simple(input_model, loaders, gpu_id, prune=True):
    """
    Computes the accuracy of a given model (input_model) on a given dataset (loaders["test"]).
    """
    if gpu_id != -1:
        input_model = input_model.cuda(gpu_id)
    input_model.eval()

    accuracy_accumulated = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders["test"]:
            if gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()

            test_output = input_model(images)

            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            accuracy_accumulated += accuracy
            total += 1
    if prune:
        input_model.cpu()
    return accuracy_accumulated / total


def update_running_statistics(input_model, loaders, gpu_id, batches=10):
    if gpu_id != -1:
        input_model = input_model.cuda(gpu_id)
    input_model.train()

    accuracy_accumulated = 0
    total = 0
    batches_count = 0
    with torch.no_grad():
        for images, labels in loaders["train"]:
            if batches_count == batches:
                break
            if gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()

            test_output = input_model(images)

            batches_count += 1

    return input_model


def original_test_manager(
    input_model_list, loaders, eval_function, pruning_function, fusion_function, gpu_id
):
    """
    Evaluate the performance of a list of networks. Typically the original/unchanged networks.
    """
    # return [0 for i in input_model_list]

    original_model_accuracies = []
    print("The accuracies of the original models are:")
    for i, input_model in enumerate(input_model_list):
        acc_this_model = eval_function(input_model=input_model, loaders=loaders, gpu_id=gpu_id)
        original_model_accuracies.append(acc_this_model)
        print(f"Model {i}:\t{acc_this_model}")

    return original_model_accuracies


def pruning_test_manager(
    input_model_list,
    loaders,
    pruning_function,
    fusion_function,
    eval_function,
    gpu_id,
    prune_params,
):
    # return input_model_list, [0 for i in input_model_list], ""
    """
    Does fusion on all models included in input_model_list and evaluates the performance of the resulting models.
    """

    pruned_models = []
    pruned_models_accuracies = []
    for i, input_model in enumerate(input_model_list):
        input_model_copy = copy.deepcopy(input_model)
        # Prune the individual networks (in place)
        _, description_pruning = pruning_function(
            input_model=input_model_copy, prune_params=prune_params
        )
        # input_model_copy,_ = train_during_pruning(model=input_model_copy, loaders=loaders, num_epochs=5, gpu_id = gpu_id, prune=False)
        pruned_models.append(input_model_copy)
        # Evaluate the performance on the given data (loaders)
        acc_model_pruned = eval_function(
            input_model=pruned_models[i], loaders=loaders, gpu_id=gpu_id
        )
        pruned_models_accuracies.append(acc_model_pruned)
        print(f"Model {i} pruned:\t{acc_model_pruned}")

    return pruned_models, pruned_models_accuracies, description_pruning


# importance para not given
# ATT: added importance para!
def fusion_test_manager(
    input_model_list,
    loaders,
    pruning_function,
    fusion_function,
    eval_function,
    gpu_id,
    num_epochs,
    accuracies=None,
    importance=None,
    name="",
):
    # return input_model_list[0], 0, ""
    """
    Does fusion of the models in input_model_list and evaluates the performance of the resulting model.
    """

    print("name is: ", name)

    fused_model, description_fusion = fusion_function(
        input_model_list, gpu_id=gpu_id, accuracies=accuracies, importance=importance, name=name
    )
    # fused_model,_ = train_during_pruning(model=fused_model, loaders=loaders, num_epochs=num_epochs, gpu_id = gpu_id, prune=False)
    acc_model_fused = eval_function(input_model=fused_model, loaders=loaders, gpu_id=gpu_id)
    print(f"Fused model:\t{acc_model_fused}")

    return fused_model, acc_model_fused, description_fusion


"""
from pruning_modified import prune_unstructured
def wrapper_unstructured_pruning(input_model, para_dict):
    '''
    This is an example for how a pruning function should be build. It takes a single model and a dictionary of parameters (para_dict).

    Checks if all arguments requiered for the wanted pruning function exist and then uses the pruning function accordingly.
    '''

    # check if necessary arguments are contained in kwargs
    assert "amount" in para_dict.keys()
    assert "prune_type" in para_dict.keys()

    amount = para_dict.get("amount")
    prune_type = para_dict.get("prune_type")

    # following does the actual pruning in place
    prune_unstructured(net=input_model, amount=amount, prune_type=prune_type)
    description = {"name": "Unstructured Pruning", "amount": amount, "prune_type":prune_type}
    return input_model, description
"""

import torch.nn as nn

############################ WORK IN PROGRESS - STRUCTURED PRUNING ##########################
from torch import optim
from torch.autograd import Variable


def train_during_pruning(model, loaders, num_epochs, gpu_id, prune=True, performed_epochs=0):
    # return model, [0 for i in range(num_epochs)]
    """
    Has to be a function that loads a dataset.

    A given model and an amount of epochs of training will be given.
    """

    if gpu_id != -1:
        model = model.cuda(gpu_id)

    loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr = 0.01)
    lr = 0.005
    model.train()

    # Train the model
    total_step = len(loaders["train"])
    best_model = model
    best_accuracy = -1

    val_acc_per_epoch = []
    this_epoch_acc = evaluate_performance_simple(
        input_model=model, loaders=loaders, gpu_id=gpu_id, prune=False
    )
    val_acc_per_epoch.append(this_epoch_acc)
    is_nan = False
    for epoch in range(num_epochs):
        optimizer = optim.SGD(
            model.parameters(), lr=lr * (0.5 ** ((epoch + performed_epochs) // 30)), momentum=0.9
        )
        for i, (images, labels) in enumerate(loaders["train"]):
            if gpu_id != -1 and not next(model.parameters()).is_cuda:
                model = model.cuda(gpu_id)
            if gpu_id != -1:
                images, labels = images.cuda(gpu_id), labels.cuda(gpu_id)
            # gives batch data, normalize x when iterate train_loader

            predictions = model(images)
            loss = loss_func(predictions, labels)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

            if math.isnan(loss.item()):
                is_nan = True
                break

        if is_nan:
            print("Is NAN")
            break
        this_epoch_acc = evaluate_performance_simple(
            input_model=model, loaders=loaders, gpu_id=gpu_id, prune=False
        )
        if this_epoch_acc > best_accuracy:
            best_model = copy.deepcopy(model)
            best_accuracy = this_epoch_acc
        val_acc_per_epoch.append(this_epoch_acc)

    model.cpu()
    best_model.cpu()
    val_acc_per_epoch.append(best_accuracy)
    return best_model, val_acc_per_epoch


from pruning_modified import prune_structured


def wrapper_structured_pruning(input_model, prune_params):
    """
    A function that makes the structured pruning function available.

    Special: it needs to provide a "retrain function" as parameter to the structured pruning function.
    """

    assert "loaders" in prune_params.keys()
    assert "num_epochs" in prune_params.keys()
    assert "example_input" in prune_params.keys()
    assert "out_features" in prune_params.keys()
    assert "prune_type" in prune_params.keys()

    loaders = prune_params.get("loaders")
    num_epochs = prune_params.get("num_epochs")
    example_input = prune_params.get("example_input")
    out_features = prune_params.get("out_features")
    prune_type = prune_params.get("prune_type")
    sparsity = prune_params.get("sparsity")
    gpu_id = prune_params.get("gpu_id")
    print("gpu_id here is: ", gpu_id)

    if "total_steps" in prune_params.keys():
        total_steps = prune_params.get("total_steps")
        pruned_model = prune_structured(
            net=input_model,
            loaders=loaders,
            num_epochs=num_epochs,
            gpu_id=gpu_id,
            example_inputs=example_input,
            out_features=out_features,
            prune_type=prune_type,
            sparsity=sparsity,
            total_steps=total_steps,
            train_fct=None,
        )
    else:
        pruned_model = prune_structured(
            net=input_model,
            loaders=loaders,
            num_epochs=num_epochs,
            gpu_id=gpu_id,
            example_inputs=example_input,
            out_features=out_features,
            prune_type=prune_type,
            sparsity=sparsity,
            train_fct=None,
        )

    description = {"name": "Structured Pruning", "num_epochs": num_epochs, "prune_type": prune_type}

    return pruned_model, description


from fusion import fusion, fusion_bn, fusion_old, fusion_old2, fusion_sidak_multimodel


def wrapper_first_fusion(list_of_models, gpu_id=-1, accuracies=None, importance=None, name=""):
    """
    Uses the first simple fusion approach created by Alex in fusion.py.
    So far this can only handle two (simple -> CNN and MLP) networks in list_of_models.
    """

    fused_model = fusion_bn(
        networks=list_of_models,
        gpu_id=gpu_id,
        accuracies=accuracies,
        importance=importance,
        resnet="resnet" in name,
    )

    description = {"name": name}

    return fused_model, description


def get_result_skeleton(parameters):
    result_final = {
        "experiment_parameters": parameters,
    }

    sparsity_list = (
        experiment_params["sparsity"]
        if isinstance(experiment_params["sparsity"], list)
        else [experiment_params["sparsity"]]
    )
    prune_type_list = (
        experiment_params["prune_type"]
        if isinstance(experiment_params["prune_type"], list)
        else [experiment_params["prune_type"]]
    )

    result_final["results"] = []
    for sparsity in sparsity_list:
        for prune_type in prune_type_list:
            result = {"sparsity": sparsity, "prune_type": prune_type}
            for model in parameters["models"]:
                dict = {
                    "accuracy_PaF": {},
                    "accuracy_FaP": {},
                    "accuracy_PaF_all": {},
                    "accuracy_fused": None,
                }
                for epoch in range(parameters["num_epochs"]):
                    dict["accuracy_PaF"][epoch] = None
                    dict["accuracy_PaF"][epoch] = None
                    dict["accuracy_PaF_all"][epoch] = None

                for j in range(0, parameters["num_models"]):
                    dict_n = {}
                    dict_n[f"accuracy_original"] = {}
                    dict_n[f"accuracy_pruned"] = {}
                    dict_n[f"accuracy_pruned_and_fused"] = {}
                    dict_n[f"accuracy_pruned_and_fused_multiple_sparsities"] = {}
                    for epoch in range(parameters["num_epochs"]):
                        dict_n[f"accuracy_original"][epoch] = None
                        dict_n[f"accuracy_pruned"][epoch] = None
                        dict_n[f"accuracy_pruned_and_fused"][epoch] = None
                        dict_n[f"accuracy_pruned_and_fused_multiple_sparsities"][epoch] = None
                    dict[f"model_{j}"] = dict_n
                result[model["name"]] = dict
            result_final["results"].append(result)
    return result_final


def float_format(number):
    return float("{:.3f}".format(number))


if __name__ == "__main__":
    with open("./experiment_parameters.json", "r") as f:
        experiment_params = json.load(f)

    result_final = get_result_skeleton(experiment_params)

    fusion_function = wrapper_first_fusion
    pruning_function = (
        wrapper_structured_pruning  # still need to implement the structured pruning function
    )
    eval_function = evaluate_performance_simple

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

    new_result = {}
    for sparsity in result_final["experiment_parameters"]["sparsity"]:
        new_result["sparstiy"] = {"paf": None, "pruned": None, "pruned_fused": None, "paf": None}

    print(json.dumps(result_final, indent=4))

    for idx_result, result in enumerate(result_final["results"]):
        for model_dict in experiment_params["models"]:
            print("new_result: ", new_result)
            name, diff_weight_init = model_dict["name"], experiment_params["diff_weight_init"]

            print(f"models/{name}_diff_weight_init_{diff_weight_init}_{0}.pth")
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
                torch.save(pruned_models[i].state_dict(), f"./models/{name}_pruned_{s}_{n_epochs}")
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
                fused_model, fused_model_accuracy, _ = fusion_test_manager(
                    input_model_list=models_original,
                    **params,
                    accuracies=original_model_accuracies,
                    num_epochs=experiment_params["num_epochs"],
                    name=name,
                )
                # experimental(new_result, result["sparsity"], models_original, original_model_accuracies, pruned_models, pruned_model_accuracies, get_parameters(), loaders, experiment_params["gpu_id"], name, params, fused_model)
                # break
                result[name]["accuracy_fused"] = float_format(fused_model_accuracy)

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
                        result[name][f"model_{i}"]["accuracy_pruned_and_fused"][idx] = float_format(
                            accuracy
                        )

            if experiment_params["PaF"]:
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
                        result[name][f"model_{i}"]["accuracy_pruned_and_fused_multiple_sparsities"][
                            idx
                        ] = float_format(accuracy)
        print(json.dumps(result_final, indent=4))
        result_final["results"][idx_result] = result

    with open("results.json", "w") as outfile:
        json.dump(result_final, outfile, indent=4)
