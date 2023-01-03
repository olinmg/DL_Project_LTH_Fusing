import torch
from models import get_pretrained_models
from parameters import get_parameters
from pruning_modified import prune_unstructured
#import main #from main import get_data_loader, test
from models import get_model
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import copy
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_mnist_data_loader():

    mnist_train = datasets.MNIST("data", train=True, transform = ToTensor(), download=True)
    mnist_test = datasets.MNIST("data", train=False, transform = ToTensor(), download=True)

    loaders = {  
        'train'  : torch.utils.data.DataLoader(mnist_train, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
        'test'  : torch.utils.data.DataLoader(mnist_test, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
    }
    return loaders
    

def get_cifar_data_loader():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)
    
    return {
        "train" : train_loader,
        "test" : val_loader
    }


def evaluate_performance_simple(input_model, loaders, gpu_id, prune=True):
    '''
    Computes the accuracy of a given model (input_model) on a given dataset (loaders["test"]).
    '''
    if gpu_id != -1:
        input_model = input_model.cuda(gpu_id)
    input_model.eval()

    accuracy_accumulated = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
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


def original_test_manager(input_model_list, loaders, eval_function, pruning_function, fusion_function, gpu_id):
    '''
    Evaluate the performance of a list of networks. Typically the original/unchanged networks.
    '''

    original_model_accuracies = []
    print("The accuracies of the original models are:")
    for i, input_model in enumerate(input_model_list):
        acc_this_model = eval_function(input_model=input_model, loaders=loaders, gpu_id=gpu_id)
        original_model_accuracies.append(acc_this_model)
        print(f"Model {i}:\t{acc_this_model}")
    
    return original_model_accuracies


"""
def functionList_test_manager(input_model, loaders, args, function_list, eval_function, para_dict):
    '''
    Takes a list of functions that should be used -> [prune_unstructured, simple_fusion]
    Takes a dictionary of arguments that will be used by the functions in the above list.

    What this function does:
    Deploys the functions in the given order with the given parameters. The resulting model will be tested on a given dataset using "evaluate_performance()".
    '''
    if len(function_list) == 0:
        print("No function in given function_list!")
        return -1

    resulting_model = input_model
    for this_function in function_list:
        resulting_model = this_function(input_model=resulting_model, para_dict=para_dict)

    accuracy = eval_function(input_model=input_model, loaders=loaders, args=args)
    return accuracy
"""

def pruning_test_manager(input_model_list, loaders, pruning_function, fusion_function, eval_function, gpu_id, prune_params):
    '''
    Does fusion on all models included in input_model_list and evaluates the performance of the resulting models.
    '''

    pruned_models = []
    pruned_models_accuracies = []
    for i, input_model in enumerate(input_model_list):
        input_model_copy = copy.deepcopy(input_model)
        # Prune the individual networks (in place)
        _, description_pruning = pruning_function(input_model=input_model_copy, prune_params=prune_params)
        #input_model_copy,_ = train_during_pruning(model=input_model_copy, loaders=loaders, num_epochs=5, gpu_id = gpu_id, prune=False)
        pruned_models.append(input_model_copy)
        # Evaluate the performance on the given data (loaders)
        acc_model_pruned = eval_function(input_model=pruned_models[i], loaders=loaders, gpu_id=gpu_id)
        pruned_models_accuracies.append(acc_model_pruned)
        print(f"Model {i} pruned:\t{acc_model_pruned}")

    return pruned_models, pruned_models_accuracies, description_pruning


def fusion_test_manager(input_model_list, loaders, pruning_function, fusion_function, eval_function, gpu_id, args, num_epochs, accuracies=None,):
    '''
    Does fusion of the models in input_model_list and evaluates the performance of the resulting model.
    '''
    
    fused_model, description_fusion = fusion_function(input_model_list, args, accuracies=accuracies)
    #fused_model,_ = train_during_pruning(model=fused_model, loaders=loaders, num_epochs=num_epochs, gpu_id = gpu_id, prune=False)
    acc_model_fused = eval_function(input_model=fused_model, loaders=loaders, gpu_id = gpu_id)
    print(f"Fused model:\t{acc_model_fused}")

    return fused_model, acc_model_fused, description_fusion

"""
def FaP_test_manager(input_model_list, loaders, args, pruning_function, fusion_function, eval_function, para_dict):
    '''
    Takes an input_model_list, a dataset, a function to do pruning, a function to do fusion and a parameter dictionary.
    First fuses the given models and the prunes the resulting network.
    The accuracy is evaluated in the intermediate steps.
    '''

    # 1. Do the fusion of the given models
    fused_model, description_fusion  = fusion_function(input_model_list, args, para_dict)
    acc_model_fused = eval_function(input_model=fused_model, loaders=loaders, args=args, para_dict=para_dict)
    print(f"Fused model:\t{acc_model_fused}")

    # 2. Do the pruning of the fused model
    pruned_model = copy.deepcopy(fused_model)
    _, description_pruning = pruning_function(input_model=pruned_model, para_dict=para_dict)
    acc_model_FaP = eval_function(input_model=pruned_model, loaders=loaders, args=args, para_dict=para_dict)
    print(f"FaP model:\t{acc_model_FaP}")

    return fused_model, pruned_model, acc_model_fused, acc_model_FaP, description_fusion, description_pruning
"""
"""
def PaF_test_manager(input_model_list, loaders, args, pruning_function, fusion_function, eval_function, para_dict):
    '''
    Takes an input_model_list, a dataset, a function to do pruning, a function to do fusion and a parameter dictionary.
    First fuses the given models and the prunes the resulting network.
    The accuracy is evaluated in the intermediate steps.
    '''

    # 1. Do the individual pruning of the given models
    pruned_models = []
    pruned_models_accuracies = []
    for i, input_model in enumerate(input_model_list):
        input_model_copy = copy.deepcopy(input_model)
        _, description_pruning = pruning_function(input_model=input_model_copy, para_dict=para_dict) #prunes in place
        pruned_models.append(input_model_copy)
        acc_model_pruned = eval_function(input_model=input_model_copy, loaders=loaders, args=args, para_dict=para_dict)
        pruned_models_accuracies.append(acc_model_pruned)
        print(f"Model {i} pruned:\t{acc_model_pruned}")

    # 2. Do a fusion of the pruned networks
    fused_model, description_fusion = fusion_function(pruned_models, args, para_dict, pruned=True)
    acc_model_fused = eval_function(input_model=fused_model, loaders=loaders, args=args, para_dict=para_dict)
    print(f"PaF model:\t{acc_model_fused}")

    return pruned_models, fused_model, pruned_models_accuracies, acc_model_fused, description_pruning, description_fusion
"""

def description_to_label(dict):
    '''
    Takes a dictionary that describes a pruning or fusion process. It turns the dictionary into a label that contains all the keys.
    Expects the dict to contain the key "name".
    '''

    assert "name" in dict.keys()

    keys = list(dict.keys())
    keys.remove("name")
    label = dict.get("name") + ":"

    for key in keys:
        label += key + "=" + str(dict.get(key)) + ","

    description = label[:-1]    # remove last comma
    
    return description

"""
################################### THIS IS WHAT NEEDS TO BE IMPLEMENTED TO MAKE USE OF THE PERFORMANCE EVALUATION FUNCTIONS ####################################
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

############################ WORK IN PROGRESS - STRUCTURED PRUNING ##########################
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
def train_during_pruning(model, loaders, num_epochs, gpu_id, prune=True):
    '''
    Has to be a function that loads a dataset. 

    A given model and an amount of epochs of training will be given.
    '''

    if gpu_id != -1:
        model = model.cuda(gpu_id)

    loss_func = nn.CrossEntropyLoss()   
    #optimizer = optim.Adam(model.parameters(), lr = 0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.05,
                                momentum=0.9)
    model.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    val_acc_per_epoch = []
    this_epoch_acc = evaluate_performance_simple(input_model=model, loaders=loaders, gpu_id=gpu_id, prune=False)
    val_acc_per_epoch.append(this_epoch_acc)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
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
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
        this_epoch_acc = evaluate_performance_simple(input_model=model, loaders=loaders, gpu_id=gpu_id, prune=False)
        val_acc_per_epoch.append(this_epoch_acc)
    
    if prune:
        model.cpu()
    return model, val_acc_per_epoch


from pruning_modified import prune_structured
def wrapper_structured_pruning(input_model, prune_params):
    '''
    A function that makes the structured pruning function available.

    Special: it needs to provide a "retrain function" as parameter to the structured pruning function. 
    '''

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
        pruned_model = prune_structured(net=input_model, loaders=loaders, num_epochs=num_epochs, gpu_id=gpu_id, example_inputs=example_input,
                    out_features=out_features, prune_type=prune_type, sparsity=sparsity, total_steps=total_steps, train_fct=None)
    else:
        pruned_model = prune_structured(net=input_model, loaders=loaders, num_epochs=num_epochs, gpu_id=gpu_id, example_inputs=example_input,
                    out_features=out_features, prune_type=prune_type, sparsity=sparsity, train_fct=None)

    description = {"name": "Structured Pruning", "num_epochs": num_epochs, "prune_type":prune_type}

    return pruned_model, description
########################################################################################



def wrapper_fake_fusion(list_of_models, args, para_dict):
    '''
    This is an example for how a fusion function should be build. It takes a list of models and a dictionary of parameters (para_dict).

    Checks if all arguments requiered for the wanted fusion function exist and then uses the fusion function accordingly.
    
    ! JUST A DUMMY FUNCTION !
    '''
    return list_of_models[0], {"name":"Fake fusion, no paras."}


from fusion import fusion, fusion_old, fusion_old2
def wrapper_first_fusion(list_of_models, args, accuracies=None,):
    '''
    Uses the first simple fusion approach created by Alex in fusion.py.
    So far this can only handle two (simple -> CNN and MLP) networks in list_of_models.
    '''

    #assert "eps" in para_dict.keys()   
    name = "First Fusion"

    fused_model = fusion(networks=list_of_models, args=args, accuracies=accuracies)

    description = {"name": name}

    return fused_model, description

"""
def create_csv_entry_from_experiment(experiment, experiment_paras_original):
    '''
    Exectues the requested pruning/fusion/PaF/FaP and creates a formatted text that can be written to a csv later.

    Essentially implements a wrapper for all the experiment functions.

    CSV Format (Column names):
    experiment_name; nn_description; data_description; pruning_description, fusion_description; original_network_performances; pruned_networks_performance; fused_network_performance; PaF_performance; FaP_performance
    '''

    assert "nn_description" in experiment_paras_original.keys()
    assert "data_description" in experiment_paras_original.keys()

    nn_description = experiment_paras_original.get("nn_description")
    data_description = experiment_paras_original.get("data_description")

    experiment_paras = copy.deepcopy(experiment_paras_original)
    del experiment_paras["nn_description"]
    del experiment_paras["data_description"]

    experiment_name=""
    label_pruning = ""
    label_fusion = ""
    original_net_perf = ""
    pruned_net_perf = ""
    fused_net_perf = ""
    PaF_perf = ""
    FaP_perf = ""

    partial_models = ""
    overall_model = ""


    if experiment is original_test_manager:

        experiment_name = "original_test_manager"
        original_net_perf = str(experiment(**experiment_paras))

    elif experiment is pruning_test_manager:

        experiment_name = "pruning_test_manager"
        pruned_models, pruned_models_accs, description_pruning = experiment(**experiment_paras)
        label_pruning = description_to_label(description_pruning)
        label_fusion = ""
        original_net_perf = ""
        pruned_net_perf = str(pruned_models_accs)
        fused_net_perf = ""
        PaF_perf = ""
        FaP_perf = ""

        partial_models = pruned_models
        overall_model = pruned_models

    elif experiment is fusion_test_manager:

        experiment_name = "fusion_test_manager"
        fused_model, fused_model_acc, description_fusion = experiment(**experiment_paras)

        label_pruning = ""
        label_fusion = description_to_label(description_fusion)
        original_net_perf = ""
        pruned_net_perf = ""
        fused_net_perf = str(fused_model_acc)
        PaF_perf = ""
        FaP_perf = ""

        partial_models = fused_model
        overall_model = fused_model

    elif experiment is PaF_test_manager:

        experiment_name = "PaF_test_manager"
        pruned_models, PaF_model, pruned_models_accs, PaF_model_acc, description_pruning, description_fusion = experiment(**experiment_paras)

        label_fusion = description_to_label(description_fusion)
        label_pruning = description_to_label(description_pruning)
        original_net_perf = ""
        pruned_net_perf = str(pruned_models_accs)
        fused_net_perf = ""
        PaF_perf = str(PaF_model_acc)
        FaP_perf = ""

        partial_models = pruned_models
        overall_model = PaF_model

    elif experiment is FaP_test_manager:

        experiment_name = "FaP_test_manager"
        fused_model, FaP_model, fused_model_acc, FaP_model_acc, description_fusion, description_pruning = experiment(**experiment_paras)
        
        label_fusion = description_to_label(description_fusion)
        label_pruning = description_to_label(description_pruning)
        original_net_perf = ""
        pruned_net_perf = ""
        fused_net_perf = str(fused_model_acc)
        PaF_perf = ""
        FaP_perf = str(FaP_model_acc)

        partial_models = fused_model
        overall_model = FaP_model
    
    result_string = experiment_name + ";" + nn_description + ";" + data_description + ";" + label_pruning + ";" + label_fusion + ";" + original_net_perf + ";" + pruned_net_perf + ";" + fused_net_perf + ";" + PaF_perf + ";" + FaP_perf
    return result_string, partial_models, overall_model
"""
"""
import os
def add_experiment_to_csv(result_string, FILE_PATH = "performance_logger.csv"):
    '''
    Exectues the requested pruning/fusion/PaF/FaP and adds the results to an existing csv file.

    CSV Format:
    experiment_name; nn_description; data_description; pruning_description, fusion_description; original_network_performances; pruned_networks_performance; fused_network_performance; PaF_performance; FaP_performance

    '''

    if not os.path.exists("./"+FILE_PATH):
        # if file does not exist yet, create a new file and put a header
        header = "experiment_name;nn_description;data_description;pruning_description,fusion_description;original_network_performances;pruned_networks_performance;fused_network_performance;PaF_performance;FaP_performance"
        with open("./"+FILE_PATH, "w") as logger:
            logger.write(header+"\n")
            logger.write(result_string+"\n")
    else:
        # if file does exist simply just append the new result_string
        with open("./"+FILE_PATH,'a') as logger:
            logger.write(result_string+"\n")
"""

def check_parameters(parameters):
    error = False
    message = ""
    if parameters["sparsity"]>= 1.0 or parameters["sparsity"] <= 0.0:
        error = True
        message +="Sparsity out of range\n"
    if len(parameters["models"]) == 0:
        error = True
        message += "At least one model required\n"
    """
    if parameters["num_models"] != 2:
        error = True
        message += "num models needs to be 2\n"
        """
    return error, message

def get_result_skeleton(parameters):
    result_final = {
        "experiment_parameters": parameters,
    }

    sparsity_list = experiment_params["sparsity"] if isinstance(experiment_params["sparsity"], list) else [experiment_params["sparsity"]]
    prune_type_list = experiment_params["prune_type"] if isinstance(experiment_params["prune_type"], list) else [experiment_params["prune_type"]]

    result_final["results"] = []
    for sparsity in sparsity_list:
        for prune_type in prune_type_list:
            result = {"sparsity": sparsity,
                        "prune_type": prune_type}
            for model in parameters["models"]:
                dict = {
                    "accuracy_PaF": {},
                    "accuracy_FaP": {},
                    "accuracy_fused": None,
                    }
                for epoch in range(parameters["num_epochs"]):
                    dict["accuracy_PaF"][epoch] = None
                    dict["accuracy_PaF"][epoch] = None

                for j in range(0, parameters["num_models"]):
                    dict_n = {}
                    dict_n[f"accuracy_original"] = {}
                    dict_n[f"accuracy_pruned"] = {}
                    dict_n[f"accuracy_pruned_and_fused"] = {}
                    for epoch in range(parameters["num_epochs"]):
                        dict_n[f"accuracy_original"][epoch] = None
                        dict_n[f"accuracy_pruned"][epoch] = None
                        dict_n[f"accuracy_pruned_and_fused"][epoch] = None
                    dict[f"model_{j}"] = dict_n
                result[model["name"]] = dict
            result_final["results"].append(result)
    return result_final

def float_format(number):
    return float("{:.3f}".format(number))

############################ TRYING TO USE THE PERFORMANCE ASSESSMENT CODE #######################

from base_resNet import get_untrained_resnet

if __name__ == '__main__':
    with open('./experiment_parameters.json', 'r') as f:
        experiment_params = json.load(f)
    
    """
    error, message = check_parameters(experiment_params)
    if error:
        print(f"Error in parsing parameters:\n{message}")
        exit()
    """
    result_final = get_result_skeleton(experiment_params)

    fusion_function = wrapper_first_fusion
    pruning_function = wrapper_structured_pruning      # still need to implement the structured pruning function
    eval_function = evaluate_performance_simple

    loaders = get_mnist_data_loader() if experiment_params["dataset"] == "mnist" else get_cifar_data_loader() 

    print(json.dumps(result_final,indent=4))

    for idx_result, result in enumerate(result_final["results"]):
        for model_dict in experiment_params["models"]:
            model_architecture = get_model(model_dict["name"])
            print(type(model_architecture))
            name, diff_weight_init = model_dict["name"], experiment_params["diff_weight_init"]

            print(f"models/{name}_diff_weight_init_{diff_weight_init}_{0}.pth")
            models_original = get_pretrained_models(name, diff_weight_init, -1, experiment_params["num_models"])

            print(type(models_original[0]))

            params = {}
            params["pruning_function"] = pruning_function
            params["fusion_function"] = fusion_function
            params["eval_function"] = eval_function
            params["loaders"] = loaders
            params["gpu_id"] = experiment_params["gpu_id"]

            original_model_accuracies = original_test_manager(input_model_list=models_original, **params)
            print("original_model_accuracies ")
            print(original_model_accuracies)
            for i in range(len(original_model_accuracies)):
                result[name][f"model_{i}"]["accuracy_original"] = float_format(original_model_accuracies[i])
            
            prune_params = {"prune_type": result["prune_type"], "sparsity": result["sparsity"], "num_epochs": experiment_params["num_epochs"],
                    "example_input": torch.randn(1,1, 28,28) if not ("vgg" in name) else torch.randn(1, 3, 32, 32),
                    "out_features": 10, "loaders": loaders, "gpu_id": experiment_params["gpu_id"]}

            pruned_models, pruned_model_accuracies,_ = pruning_test_manager(input_model_list=models_original, prune_params=prune_params, **params)
            for i in range(len(pruned_model_accuracies)):
                m,epoch_accuracy = train_during_pruning(copy.deepcopy(pruned_models[i]), loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"])
                for idx, accuracy in enumerate(epoch_accuracy):
                    result[name][f"model_{i}"]["accuracy_pruned"][idx] = float_format(accuracy)
            
            fusion_params = get_parameters()
            fusion_params.model_name = name

            fused_model, fused_model_accuracy,_ = fusion_test_manager(input_model_list=models_original, **params, accuracies=original_model_accuracies, num_epochs = experiment_params["num_epochs"], args=fusion_params)
            result[name]["accuracy_fused"] = float_format(fused_model_accuracy)

            for i in range(len(pruned_models)):
                pruned_and_fused_model, pruned_and_fused_model_accuracy,_ = fusion_test_manager(input_model_list=[pruned_models[i], models_original[i]], **params, num_epochs = experiment_params["num_epochs"], args=fusion_params)
                m,epoch_accuracy= train_during_pruning(copy.deepcopy(pruned_and_fused_model), loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"], prune=False)
                for idx, accuracy in enumerate(epoch_accuracy):
                    result[name][f"model_{i}"]["accuracy_pruned_and_fused"][idx] = float_format(accuracy)
            
            if experiment_params["PaF"]:
                paf_model, paf_model_accuracy,_ = fusion_test_manager(input_model_list=pruned_models, **params, num_epochs = experiment_params["num_epochs"], args=fusion_params)
                m,epoch_accuracy = train_during_pruning(paf_model, loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"], prune=False)
                for idx, accuracy in enumerate(epoch_accuracy):
                    result[name]["accuracy_PaF"][idx] = float_format(accuracy)
            
            if experiment_params["FaP"]:
                fap_models, fap_model_accuracies,_ = pruning_test_manager(input_model_list=[fused_model], prune_params=prune_params, **params)
                m,epoch_accuracy = train_during_pruning(fap_models[0], loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"], prune=False)
                for idx, accuracy in enumerate(epoch_accuracy):
                    result[name]["accuracy_FaP"][idx] = float_format(accuracy)
        print(json.dumps(result_final,indent=4))
        result_final["results"][idx_result] = result
    with open("results.json", "w") as outfile:
        json.dump(result_final, outfile, indent=4)
