import torch
from models import get_pretrained_models_by_name
from parameters import get_parameters
from models import get_model
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import copy
import json
from performance_tester import get_mnist_data_loader, get_cifar_data_loader, evaluate_performance_simple
from performance_tester import original_test_manager, pruning_test_manager, fusion_test_manager
from performance_tester import wrapper_structured_pruning, wrapper_first_fusion
from performance_tester import train_during_pruning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_multiple_settings(sparsity_list, fusion_weights_list, input_model_names, RESULT_FILE_PATH):
    '''
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
    '''
    
    if len(sparsity_list) > 1 and len(fusion_weights_list) == 1:
        only_varying_sparsity_not_weights = True
    else:
        only_varying_sparsity_not_weights = False
    
    with open('./logger.txt', 'w') as logger:
        logger.write(f"Starting process with models:\n")
        logger.write(f"{input_model_names[0]}\n")
        logger.write(f"{input_model_names[1]}\n\n\n")

    start_time = time.time()

    '''
    with open("./"+RESULT_FILE_PATH,'a') as logger:
        logger.write("Model names:\n")
        for model_name in input_model_names:
            logger.write(model_name+"\n")
        logger.write("\n")
    '''
        
    hyper_performance_measurements_dict = {}

    # load the relevant experiment parameters
    with open('./experiment_parameters.json', 'r') as f:
        experiment_params = json.load(f)

    # load the relevant data and models
    loaders = get_mnist_data_loader() if experiment_params["dataset"] == "mnist" else get_cifar_data_loader()
    name, diff_weight_init = experiment_params["models"][0]["name"], experiment_params["diff_weight_init"]
    model_architecture = get_model(name)
    original_models = get_pretrained_models_by_name(name, diff_weight_init, experiment_params["gpu_id"], experiment_params["num_models"], input_model_names)


    ##################### START COMPUTATIONS THAT ARE COMMON OVER VARIATIONS OF SPARSITY AND WEIGHTS ####################
    fusion_function = wrapper_first_fusion
    pruning_function = wrapper_structured_pruning
    eval_function = evaluate_performance_simple
    params = {}
    params["pruning_function"] = pruning_function
    params["fusion_function"] = fusion_function
    params["eval_function"] = eval_function
    params["loaders"] = loaders
    params["gpu_id"] = experiment_params["gpu_id"]
    
    performance_measurements = {}
    # Performance of: original models
    with open('./logger.txt', 'a') as logger:
        logger.write(f"Computing original models performances...\n")
    original_model_accuracies = original_test_manager(input_model_list=original_models, **params)
    performance_measurements["original_model_accuracies"] = original_model_accuracies

    # Performance of: original models + extra training
    if experiment_params["num_epochs"] > 0:
        original_model_retrained_accuracies_list = []
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Training original models for additional epochs...\n")
        for this_model in original_models:
            retrained_model, retrained_model_accuracy = train_during_pruning(copy.deepcopy(this_model), loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"], prune=False)
            original_model_retrained_accuracies_list.append(retrained_model_accuracy)
        performance_measurements["original_model_retrained_accuracies_list"] = original_model_retrained_accuracies_list    
    with open('./logger.txt', 'a') as logger:
        logger.write(f"\n\n")
    ##################### DONE WITH COMPUTATIONS THAT ARE COMMON OVER VARIATIONS OF SPARSITY AND WEIGHTS ####################


    ### ITERATE THE FUSION_WEIGHTS AND SPARSITIES
    if not only_varying_sparsity_not_weights:
        # in the case that weights are not fixed we create a file per sparsity, in the case that weights are fixed and sparsity varies we create a single file at the fixed weights (includes varying sparsities)
        for sparsity in sparsity_list:
            # create a file per sparsity: for each sparsity we iterate the different fusion weights
            start_time = time.time()
            for fusion_weights in fusion_weights_list:
                performance_measurements_dict = test_one_setting(sparsity, fusion_weights, input_model_names, RESULT_FILE_PATH, original_models, original_model_accuracies, experiment_params, performance_measurements, params)
                performance_measurements_dict["sparsity"] = sparsity
                performance_measurements_dict["fusion_weights"] = fusion_weights
                hyper_performance_measurements_dict[str(fusion_weights)] = copy.deepcopy(performance_measurements_dict)
            # Combining results into an output .json
            end_time = time.time()
            execution_time = end_time - start_time
            with open('./experiment_parameters.json', 'r') as f:
                experiment_params_write = json.load(f)
            experiment_params_dict = {}
            experiment_params_write["Execution Time"] = execution_time
            experiment_params_write["sparsity"] = sparsity
            experiment_params_write["fusion_weights"] = fusion_weights_list
            experiment_params_dict["Experiment Parameters"] = experiment_params_write
            hyper_performance_measurements_dict_complete = {**experiment_params_dict, **hyper_performance_measurements_dict}
            RESULT_FILE_PATH_MOD = f"{RESULT_FILE_PATH}_sparsity{int(sparsity*100)}.json"
            with open(RESULT_FILE_PATH_MOD, 'a') as outfile:
                json.dump(hyper_performance_measurements_dict_complete, outfile, indent=4)
    else:
        # the weights are fixed and the sparsity is varied
        for fusion_weights in fusion_weights_list: # should only contain one in this case
            start_time = time.time()
            for sparsity in sparsity_list:
                performance_measurements_dict = test_one_setting(sparsity, fusion_weights, input_model_names, RESULT_FILE_PATH, original_models, original_model_accuracies, experiment_params, performance_measurements, params)
                performance_measurements_dict["sparsity"] = sparsity
                performance_measurements_dict["fusion_weights"] = fusion_weights
                hyper_performance_measurements_dict[str(sparsity)] = copy.deepcopy(performance_measurements_dict)
            # Combining results into an output .json
            end_time = time.time()
            execution_time = end_time - start_time
            with open('./experiment_parameters.json', 'r') as f:
                experiment_params_write = json.load(f)
            experiment_params_dict = {}
            experiment_params_write["Execution Time"] = execution_time
            experiment_params_write["sparsity"] = sparsity_list
            experiment_params_write["fusion_weights"] = fusion_weights
            experiment_params_dict["Experiment Parameters"] = experiment_params_write
            hyper_performance_measurements_dict_complete = {**experiment_params_dict, **hyper_performance_measurements_dict}
            RESULT_FILE_PATH_MOD = f"{RESULT_FILE_PATH}_weight{int(fusion_weights[0]*100)}.json"
            with open(RESULT_FILE_PATH_MOD, 'a') as outfile:
                json.dump(hyper_performance_measurements_dict_complete, outfile, indent=4)



def test_one_setting(sparsity, fusion_weights, input_model_names, RESULT_FILE_PATH, original_models, original_model_accuracies, experiment_params, performance_measurements, params):
    '''
    Given a single sparsity and fusion_weights setting, this function executes all experiments (e.g. PaF, FaP, ...) - besides MSF and SSF - that are set to true in the experiment_parameters.json.
    The resulting performance measures (including epoch-wise test performance for possible retraining) are returned in the dictionary "performance_measurements".

    The corresponding trained models are also available (see e.g. FaP_model, PaF_model, ...), but are not returned. This can be easily changed.
    '''
    # Can only handle one sparsity type (e.g. "l1") at a time

    with open('./logger.txt', 'a') as logger:
        logger.write(f"Starting with sparsity: {sparsity}, fusion_weights: {fusion_weights}.\n")
    
    """
    error, message = check_parameters(experiment_params)
    if error:
        print(f"Error in parsing parameters:\n{message}")
        exit()
    """
    
    name, diff_weight_init = experiment_params["models"][0]["name"], experiment_params["diff_weight_init"]
    loaders = params["loaders"]


    # FUSION AND PRUNING SPECIFICATIONS
    fusion_params = get_parameters()
    fusion_params.model_name = name
    # Can only handle one sparsity type at a time (takes the first if multiple in list)
    prune_type = experiment_params["prune_type"] if isinstance(experiment_params["prune_type"], str) else experiment_params["prune_type"][0]
    prune_params = {"prune_type": prune_type, "sparsity": sparsity, "num_epochs": experiment_params["num_epochs"],
            "example_input": torch.randn(1,1, 28,28) if not ("vgg" in name) else torch.randn(1, 3, 32, 32),
            "out_features": 10, "loaders": loaders, "gpu_id": experiment_params["gpu_id"]}


    ### Combinations that start with pruning
    # P
    if (experiment_params["PaF"] or experiment_params["PaFaP"] or experiment_params["PaFaT"] or experiment_params["PaT"] or experiment_params["PaTaF"] or experiment_params["PaTaFaT"]):
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Pruning the original models...\n")
        pruned_models, original_pruned_models_accuracies,_ = pruning_test_manager(input_model_list=copy.deepcopy(original_models), prune_params=prune_params, **params)
        performance_measurements["original_pruned_models_accuracies"] = original_pruned_models_accuracies

    # PaF
    if (experiment_params["PaF"] or experiment_params["PaFaP"] or experiment_params["PaFaT"]):
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Computing PaF...\n")
        PaF_model, PaF_model_accuracy,_ = fusion_test_manager(input_model_list=copy.deepcopy(pruned_models), **params, num_epochs = experiment_params["num_epochs"], args=fusion_params, importance=fusion_weights)
        performance_measurements["PaF_model_accuracy"] = PaF_model_accuracy

    # PaFaP
    if experiment_params["PaFaP"]:
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Computing PaFaP...\n")
        PaFaP_model, PaFaP_model_accuracy, _ = pruning_test_manager(input_model_list=copy.deepcopy([PaF_model]), prune_params=prune_params, **params)
        performance_measurements["PaFaP_model_accuracy"] = PaFaP_model_accuracy[0]

    # PaFaT
    if experiment_params["num_epochs"] > 0 and experiment_params["PaFaT"]:
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Computing PaFaT...\n")
        PaFaT_model, PaFaT_model_epoch_accuracy = train_during_pruning(copy.deepcopy(PaF_model), loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"], prune=False)
        performance_measurements["PaFaT_model_epoch_accuracy"] = PaFaT_model_epoch_accuracy

    # PaT
    if experiment_params["num_epochs"] > 0 and (experiment_params["PaT"] or experiment_params["PaTaF"] or experiment_params["PaTaFaT"]):
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Computing PaT...\n")
        PaT_model_list = []
        PaT_models_epoch_accuracies_list = []
        for i in range(len(original_pruned_models_accuracies)):
            #torch.save(pruned_models[i].state_dict(), "models/{}_pruned_{}_.pth".format(name, i))
            PaT_model, PaT_model_epoch_accuracies = train_during_pruning(copy.deepcopy(pruned_models[i]), loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"], prune=False)
            PaT_model_list.append(PaT_model)
            PaT_models_epoch_accuracies_list.append(PaT_model_epoch_accuracies)
        #### -> PaT_models_epoch_accuracies_list
        performance_measurements["PaT_models_epoch_accuracies_list"] = PaT_models_epoch_accuracies_list

    # PaTaF
    if experiment_params["num_epochs"] > 0 and (experiment_params["PaTaF"] or experiment_params["PaTaFaT"]):
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Computing PaTaF...\n")
        PaTaF_model, PaTaF_model_accuracy, _ = fusion_test_manager(input_model_list=copy.deepcopy(PaT_model_list), **params, num_epochs = experiment_params["num_epochs"], args=fusion_params, importance=fusion_weights)
        performance_measurements["PaTaF_model_accuracy"] = PaTaF_model_accuracy
    
    # PaTaFaT
    if experiment_params["num_epochs"] > 0 and experiment_params["PaTaFaT"]:
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Computing PaTaFaT...\n")
        PaTaFaT_model, PaTaFaT_model_accuracy = train_during_pruning(copy.deepcopy(PaTaF_model), loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"], prune=False)
        performance_measurements["PaTaFaT_model_accuracy"] = PaTaFaT_model_accuracy


    ### Combinations that start with fusion
    # F
    if (experiment_params["FaP"] or experiment_params["FaPaT"] or experiment_params["FaT"] or experiment_params["FaTaP"] or experiment_params["FaTaPaT"] ):
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Fusing the original models...\n")
        original_fused_model, original_fused_model_accuracy,_ = fusion_test_manager(input_model_list=copy.deepcopy(original_models), **params, accuracies=original_model_accuracies, num_epochs = experiment_params["num_epochs"], args=fusion_params, importance=fusion_weights)
        performance_measurements["original_fused_model_accuracy"] = original_fused_model_accuracy

    # FaP
    if (experiment_params["FaP"] or experiment_params["FaPaT"]):
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Computing FaP...\n")
        FaP_model, FaP_model_accuracy,_ = pruning_test_manager(input_model_list=copy.deepcopy([original_fused_model]), prune_params=prune_params, **params)
        performance_measurements["FaP_model_accuracy"] = FaP_model_accuracy

    # FaPaT
    if experiment_params["num_epochs"] > 0 and experiment_params["FaPaT"]:
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Computing FaPaT...\n")
        FaPaT_model, FaPaT_model_epoch_accuracy = train_during_pruning(copy.deepcopy(FaP_model[0]), loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"], prune=False)
        performance_measurements["FaPaT_model_epoch_accuracy"] = FaPaT_model_epoch_accuracy

    # FaT
    if experiment_params["num_epochs"] > 0 and (experiment_params["FaT"] or experiment_params["FaTaP"] or experiment_params["FaTaPaT"]):
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Computing FaT...\n")
        FaT_model, FaT_epoch_accuracies = train_during_pruning(copy.deepcopy(original_fused_model), loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"], prune=False)
        performance_measurements["FaT_epoch_accuracies"] = FaT_epoch_accuracies

    # FaTaP
    if experiment_params["num_epochs"] > 0 and (experiment_params["FaTaP"] or experiment_params["FaTaPaT"]):
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Computing FaTaP...\n")
        FaTaP_model, FaTaP_model_accuracy, _ = pruning_test_manager(input_model_list=copy.deepcopy([FaT_model]), prune_params=prune_params, **params)
        performance_measurements["FaTaP_model_accuracy"] = FaTaP_model_accuracy
        
    # FaTaPaT
    if experiment_params["num_epochs"] > 0 and experiment_params["FaTaPaT"]:
        with open('./logger.txt', 'a') as logger:
            logger.write(f"Computing FaTaPaT...\n")
        FaTaPaT_model, FaTaPaT_model_accuracy = train_during_pruning(copy.deepcopy(FaTaP_model[0]), loaders=loaders, num_epochs=experiment_params["num_epochs"], gpu_id =experiment_params["gpu_id"], prune=False)
        performance_measurements["FaTaPaT_model_accuracy"] = FaTaPaT_model_accuracy
    

    with open('./logger.txt', 'a') as logger:
        logger.write(f"Done with sparsity: {sparsity}, fusion_weights: {fusion_weights}.\n\n\n")

    return performance_measurements



import random, os
import numpy as np
import time
if __name__ == '__main__':
    '''
    See description of test_multiple_settings() for details whats happening in this script.

    The format of output are .json files that are stored in "results_and_plots" in the subfolder specified by the model name.
    The location can be changed.
    Models are loaded from the below given files (without the .pth ending).
    '''

    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(3)
    random.seed(3)
    os.environ['PYTHONHASHSEED'] = str(3)

    # Loading experiment parameters from corresponding file
    with open('./experiment_parameters.json', 'r') as f:
        experiment_params = json.load(f)
    model_name = experiment_params["models"][0]["name"]
    network_difference = "DiffInit" if experiment_params["diff_weight_init"] else "DiffData"
    
    # Define where the resulting .json file should be stored 
    result_folder_name = f"./results_and_plots/fullDict_results_{model_name}"
    result_file_name = f"TEST_this_experiments_result_nets{network_difference}"     # without .json ending!
    result_file_path = result_folder_name+"/"+result_file_name
    if not os.path.exists(result_folder_name):
        os.makedirs(result_folder_name)     # create folder if doesnt exist before
    
    # Path to the models that should be used (without .pth)
    input_model_names = ["models/cnn_diff_weight_init_False_0", 
                         "models/cnn_diff_weight_init_False_1"]
    
    # Running the experiments on different sparsity and fusion_weight combinations
    this_weight_list = experiment_params["fusion_weights"] #[[0.5, 0.5], [0.7, 0.3]] #[[round(x, 1), round(1-x,1)] for x in np.linspace(0, 1, 11)]
    this_sparsity_list = experiment_params["sparsity"] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_path)
    