import torch
from parameters import get_parameters
from pruning import prune_unstructured
#import main #from main import get_data_loader, test
from base_convNN import CNN, MLP
from torchvision import datasets
from torchvision.transforms import ToTensor
import copy

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

    cifar10_train = datasets.CIFAR10("data", train=True, transform = ToTensor(), download=True)
    cifar10_test = datasets.CIFAR10("data", train=False, transform = ToTensor(), download=True)

    loaders = {  
        'train'  : torch.utils.data.DataLoader(cifar10_train, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
        'test'  : torch.utils.data.DataLoader(cifar10_test, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
    }
    return loaders


def evaluate_performance_simple(input_model, loaders, args, para_dict):
    '''
    Computes the accuracy of a given model (input_model) on a given dataset (loaders["test"]).
    '''

    input_model.eval()

    accuracy_accumulated = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            if args.gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()
            
            try:
                test_output, _ = input_model(images)    # TODO: WHY DOES THIS RETURN TWO VALUES?!
            except:
                test_output = input_model(images)

            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            accuracy_accumulated += accuracy 
            total += 1
    return accuracy_accumulated / total


def original_test_manager(input_model_list, loaders, args, pruning_function, fusion_function, eval_function, para_dict):
    '''
    Evaluate the performance of a list of networks. Typically the original/unchanged networks.
    '''

    original_model_accuracies = []
    print("The accuracies of the original models are:")
    for i, input_model in enumerate(input_model_list):
        acc_this_model = eval_function(input_model=input_model, loaders=loaders, args=args, para_dict=para_dict)
        original_model_accuracies.append(acc_this_model)
        print(f"Model {i}:\t{acc_this_model}")
    
    return original_model_accuracies


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


def pruning_test_manager(input_model_list, loaders, args, pruning_function, fusion_function, eval_function, para_dict):
    '''
    Does fusion on all models included in input_model_list and evaluates the performance of the resulting models.
    '''

    pruned_models = []
    pruned_models_accuracies = []
    for i, input_model in enumerate(input_model_list):
        input_model_copy = copy.deepcopy(input_model)
        # Prune the individual networks (in place)
        _, description_pruning = pruning_function(input_model=input_model_copy, para_dict=para_dict)
        pruned_models.append(input_model_copy)
        # Evaluate the performance on the given data (loaders)
        acc_model_pruned = eval_function(input_model=input_model_copy, loaders=loaders, args=args, para_dict=para_dict)
        pruned_models_accuracies.append(acc_model_pruned)
        print(f"Model {i} pruned:\t{acc_model_pruned}")

    return pruned_models, pruned_models_accuracies, description_pruning


def fusion_test_manager(input_model_list, loaders, args, pruning_function, fusion_function, eval_function, para_dict):
    '''
    Does fusion of the models in input_model_list and evaluates the performance of the resulting model.
    '''
    
    fused_model, description_fusion = fusion_function(input_model_list, args, para_dict)
    acc_model_fused = eval_function(input_model=fused_model, loaders=loaders, args=args, para_dict=para_dict)
    print(f"Fused model:\t{acc_model_fused}")

    return fused_model, acc_model_fused, description_fusion


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
    fused_model, description_fusion = fusion_function(pruned_models, args, para_dict)
    acc_model_fused = eval_function(input_model=fused_model, loaders=loaders, args=args, para_dict=para_dict)
    print(f"PaF model:\t{acc_model_fused}")

    return pruned_models, fused_model, pruned_models_accuracies, acc_model_fused, description_pruning, description_fusion


def test_manager_wrapper():
    '''
    Iterates through a defined set of function lists (and parameter sets) and uses the ?_test_manager function to make tests on all those settings.
    '''

    pass


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


################################### THIS IS WHAT NEEDS TO BE IMPLEMENTED TO MAKE USE OF THE PERFORMANCE EVALUATION FUNCTIONS ####################################
from pruning import prune_unstructured
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


def wrapper_structured_pruning(input_model, para_dict):
    '''
    A function that makes the structured pruning function available.

    Special: it needs to provide a "retrain function" as parameter to the structured pruning function. 
    '''
    pass


def wrapper_fake_fusion(list_of_models, args, para_dict):
    '''
    This is an example for how a fusion function should be build. It takes a list of models and a dictionary of parameters (para_dict).

    Checks if all arguments requiered for the wanted fusion function exist and then uses the fusion function accordingly.
    
    ! JUST A DUMMY FUNCTION !
    '''
    return list_of_models[0], {"name":"Fake fusion, no paras."}


from fusion import fusion
def wrapper_first_fusion(list_of_models, args, para_dict):
    '''
    Uses the first simple fusion approach created by Alex in fusion.py.
    So far this can only handle two (simple -> CNN and MLP) networks in list_of_models.
    '''

    #assert "eps" in para_dict.keys()   
    name = "First Fusion"

    # eps does not necessarily need to be contained
    if "eps" in para_dict.keys():    
        eps = para_dict.get("eps")
        print(f"eps: {eps}")
        fused_model = fusion(networks=list_of_models, args=args, eps=eps)
        description = {"name": name, "eps": eps}
    else:
        fused_model = fusion(networks=list_of_models, args=args)
        description = {"name": name}

    return fused_model, description


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



from base_resNet import get_untrained_resnet

if __name__ == '__main__':

    # getting the models that should be evaluated: MLP, ConvNN
    MLP_MODEL_FILE_NAME = "./base_mlp_model_dict.pth"
    pretrained_mlp_model = MLP()
    pretrained_mlp_model.load_state_dict(torch.load(MLP_MODEL_FILE_NAME))

    CNN_MODEL_FILE_NAME = "./base_cnn_model_dict.pth"
    pretrained_cnn_model = CNN()
    pretrained_cnn_model.load_state_dict(torch.load(CNN_MODEL_FILE_NAME))

    RESNET_MODEL_FILE_NAME = "./resnet_cifar10_model_dict_20epochs.pth"
    pretrained_resnet_model = get_untrained_resnet()
    pretrained_resnet_model.load_state_dict(torch.load(RESNET_MODEL_FILE_NAME))


    # Getting data and arguments that are used in the pruning/fusion/evaluation function
    # Should maybe also include these in the "para_dict" for clarity/cleanliness
    loaders_mnist = get_mnist_data_loader()
    loaders_cifar = get_cifar_data_loader()
    args = get_parameters()

    # Setting the parameters of the pruning/fusion/evaluation functions
    amounts = [0.2, 0.3]
    prune_types = ["l1", "random"]
    eps = [1e-7]

    # They are all collected in the para_dict. The wrapper function of a pruning/fusion/evaluation function handles the handover. 
    para_dict = {"amount": amounts[0], "prune_type": prune_types[0]}

    # What functions to use in the process of pruning and fusion
    pretrained_cnn_model_2 = copy.deepcopy(pretrained_cnn_model)
    input_model_list_cnn = [pretrained_cnn_model, pretrained_cnn_model_2]
    
    pretrained_mlp_model_2 = copy.deepcopy(pretrained_mlp_model)
    input_model_list_mlp = [pretrained_mlp_model, pretrained_mlp_model_2]

    pretrained_resnet_model_2 = copy.deepcopy(pretrained_resnet_model)
    input_model_list_resnet = [pretrained_resnet_model, pretrained_resnet_model_2]
    fusion_function = wrapper_fake_fusion
    #fusion_function = wrapper_first_fusion
    pruning_function = wrapper_unstructured_pruning
    # pruning_function = wrapper_structured_pruning      # still need to implement the structured pruning function

    eval_function = evaluate_performance_simple


    # RESNET 
    # Show original models performance (resnet) and pruned resnet
    #print("RESNET - Eval resnet original:")
    #original_model_accuracies = original_test_manager(input_model_list=input_model_list_resnet, loaders=loaders_cifar, args=args,
    #                                                eval_function=eval_function, para_dict=para_dict)

    experiment_parameters = {"nn_description":"Two simple ConvNN implemented in CNN()", 
                            "data_description":"MNIST",
                            "input_model_list":input_model_list_mlp,
                            "loaders":loaders_mnist,
                            "args":args,
                            "pruning_function":pruning_function,
                            "fusion_function":fusion_function,
                            "eval_function":eval_function,
                            "para_dict":para_dict}

    # PROBLEMS:
    # CNN: PaF
    # MLP: PaF

    # Problem fixes:
    # Mistake 1: parameters.py is set to return the wrong "model_name" (cnn vs mlp)
    
    experiment_to_execute = PaF_test_manager
    result_str, partial_models, overall_model = create_csv_entry_from_experiment(experiment_to_execute, experiment_parameters)
    add_experiment_to_csv(result_str)
    print(result_str)
    
    experiment_to_execute = fusion_test_manager
    result_str, partial_models, overall_model = create_csv_entry_from_experiment(experiment_to_execute, experiment_parameters)
    add_experiment_to_csv(result_str)
    print(result_str)

    experiment_to_execute = pruning_test_manager
    result_str_original, partial_models, overall_model = create_csv_entry_from_experiment(experiment_to_execute, experiment_parameters)
    add_experiment_to_csv(result_str_original)
    print(result_str_original)
    
    experiment_to_execute = FaP_test_manager
    result_str, partial_models, overall_model = create_csv_entry_from_experiment(experiment_to_execute, experiment_parameters)
    add_experiment_to_csv(result_str)
    print(result_str)

    experiment_to_execute = PaF_test_manager
    result_str, partial_models, overall_model = create_csv_entry_from_experiment(experiment_to_execute, experiment_parameters)
    add_experiment_to_csv(result_str)
    print(result_str)

    
    print("INTERESTING STUFF DONE!")
    
    fusion(input_model_list_mlp, args, eps=1e-7)

    print("ConvNN - Eval PaF")
    del experiment_parameters["nn_description"]
    del experiment_parameters["data_description"]


    pruned_models, PaF_model, pruned_models_accs, PaF_model_acc, description_pruning, description_fusion = PaF_test_manager(**experiment_parameters)
    
    print("ConvNN - Eval FaP")
    fused_model, FaP_model, fused_model_acc, FaP_model_acc, description_fusion, description_pruning = FaP_test_manager(**experiment_parameters)

    print("RESNET - Eval pruning on resnet:")
    pruned_models, pruned_models_accs, description_pruning = pruning_test_manager(**experiment_parameters)
    print(description_to_label(description_pruning))

    # ConvNN
    print("ConvNN - Eval orig models")
    original_model_accuracies = original_test_manager(**experiment_parameters)
                                                    
    print("ConvNN - Eval pruning")
    pruned_models, pruned_models_accs, description_pruning = pruning_test_manager(**experiment_parameters)

    print("ConvNN - Eval fusion")
    fused_model, fused_model_acc, description_fusion = fusion_test_manager(**experiment_parameters)