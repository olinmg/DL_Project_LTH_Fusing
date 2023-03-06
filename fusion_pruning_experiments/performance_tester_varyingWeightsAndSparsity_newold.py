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

'''
def get_mnist_data_loader():

    mnist_train = datasets.MNIST("data", train=True, transform = ToTensor(), download=True)
    mnist_test = datasets.MNIST("data", train=False, transform = ToTensor(), download=True)

    loaders = {  
        'train'  : torch.utils.data.DataLoader(mnist_train, 
                                            batch_size=128, 
                                            shuffle=False, 
                                            num_workers=4),
        'test'  : torch.utils.data.DataLoader(mnist_test, 
                                            batch_size=128, 
                                            shuffle=False, 
                                            num_workers=4),
    }
    return loaders
    

def get_cifar_data_loader():
    # allgined with ALEX
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
    
    # Computes the accuracy of a given model (input_model) on a given dataset (loaders["test"]).
    
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


    #return [0 for i in input_model_list]

def original_test_manager(input_model_list, loaders, eval_function, pruning_function, fusion_function, gpu_id):
    #return [0 for i in input_model_list]
    #Evaluate the performance of a list of networks. Typically the original/unchanged networks.
    

    original_model_accuracies = []
    print("The accuracies of the original models are:")
    for i, input_model in enumerate(input_model_list):
        acc_this_model = eval_function(input_model=input_model, loaders=loaders, gpu_id=gpu_id)
        original_model_accuracies.append(acc_this_model)
        print(f"Model {i}:\t{acc_this_model}")
    
    return original_model_accuracies


def pruning_test_manager(input_model_list, loaders, pruning_function, fusion_function, eval_function, gpu_id, prune_params):
    #return input_model_list, [0 for i in input_model_list], ""
    
    #Does fusion on all models included in input_model_list and evaluates the performance of the resulting models.
    

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


    
def fusion_test_manager(input_model_list, loaders, pruning_function, fusion_function, eval_function, gpu_id, args, num_epochs, accuracies=None, importance=None):
    #return input_model_list[0], 0, ""
    
    #Does fusion of the models in input_model_list and evaluates the performance of the resulting model.
    
    fused_model, description_fusion = fusion_function(input_model_list, args, accuracies=accuracies, importance=importance)
    #fused_model,_ = train_during_pruning(model=fused_model, loaders=loaders, num_epochs=num_epochs, gpu_id = gpu_id, prune=False)
    acc_model_fused = eval_function(input_model=fused_model, loaders=loaders, gpu_id = gpu_id)
    print(f"Fused model:\t{acc_model_fused}")

    return fused_model, acc_model_fused, description_fusion
'''
'''
from torch import optim
import torch.nn as nn
def train_during_pruning(model, loaders, num_epochs, gpu_id, prune=True):
    return model, [0 for i in range(num_epochs)]
    #Has to be a function that loads a dataset. 
    #A given model and an amount of epochs of training will be given.
    
    if gpu_id != -1:
        model = model.cuda(gpu_id)

    loss_func = nn.CrossEntropyLoss()   
    #optimizer = optim.Adam(model.parameters(), lr = 0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.05,
                                momentum=0.9)
    model.train()
        
    # Train the model
    total_step = len(loaders['train'])
    
    best_accuracy = 0
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
        # only return the best model from the training
        if this_epoch_acc > best_accuracy:
            best_model = copy.deepcopy(model)
            best_accuracy = this_epoch_acc
    if prune:
        model.cpu()
        best_model.cpu()
    return best_model, val_acc_per_epoch
'''


'''
from pruning_modified import prune_structured
def wrapper_structured_pruning(input_model, prune_params):
    
    #A function that makes the structured pruning function available.
    

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
'''
########################################################################################



'''
from fusion import fusion, fusion_old, fusion_old2
def wrapper_first_fusion(list_of_models, args, accuracies=None, importance=None):
    
    #Uses the first simple fusion approach created by Alex in fusion.py.
    #So far this can only handle two (simple -> CNN and MLP) networks in list_of_models.
    

    #assert "eps" in para_dict.keys()   
    name = "First Fusion"

    fused_model = fusion(networks=list_of_models, args=args, accuracies=accuracies, importance=importance)

    description = {"name": name}

    return fused_model, description


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
'''


############################ DIFFERENT PRUNING DEGREES AT 50/50 FUSION WEIGHTS ##############################
def test_multiple_settings(sparsity_list, fusion_weights_list, input_model_names, RESULT_FILE_PATH):
    
    
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

    '''            
    # Iterate over the different combinations of sparsity and fusion_weights given in the corresponding lists
    for fusion_weights in fusion_weights_list:
        for sparsity in sparsity_list:
            performance_measurements_dict = test_one_setting(sparsity, fusion_weights, input_model_names, RESULT_FILE_PATH, original_models, original_model_accuracies, experiment_params, performance_measurements, params)
            performance_measurements_dict["sparsity"] = sparsity
            performance_measurements_dict["fusion_weights"] = fusion_weights
            if varying_sparsity:
                hyper_performance_measurements_dict[str(sparsity)] = copy.deepcopy(performance_measurements_dict)
            else:
                hyper_performance_measurements_dict[str(fusion_weights)] = copy.deepcopy(performance_measurements_dict)

            with open("./"+RESULT_FILE_PATH,'a') as logger:
                if varying_sparsity:
                    logger.write("Sparsity: "+str(sparsity)+"\n")
                else:
                    logger.write("Fusion weights: "+str(fusion_weights))
                logger.write(str(performance_measurements_dict)+"\n\n\n\n\n")
    '''     

    return hyper_performance_measurements_dict



def test_one_setting(sparsity, fusion_weights, input_model_names, RESULT_FILE_PATH, original_models, original_model_accuracies, experiment_params, performance_measurements, params):
    # Can only handle one sparsity type at a time

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



############################ USING DIFFERENT FUSION WEIGHTS AT A FIXED PRUNING AMOUNT #######################

from base_resNet import get_untrained_resnet
import random, os
import numpy as np
import time
if __name__ == '__main__':

    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(3)
    random.seed(3)
    os.environ['PYTHONHASHSEED'] = str(3)


    # Path to models to use (without .pth)
    input_model_names = ["models/cnn_diff_weight_init_False_0", 
                         "models/cnn_diff_weight_init_False_1"]
    
    
    # For two given models (in input_model_names) test PaF, FaP, pruned etc at multiple sparsities/fusion weights
    # can only vary one: weights or sparsities! One list has to contain only one element!
    
    
    ##### EXAMPLE ON HOW TO RUN THE EXPERIMENTS

    # TESTING a list of sparsities for fixed fusion weights
    with open('./experiment_parameters.json', 'r') as f:
        experiment_params = json.load(f)
    this_weight_list = experiment_params["fusion_weights"] #[[0.5, 0.5]] #[[round(x, 1), round(1-x,1)] for x in np.linspace(0, 1, 11)]
    this_sparsity_list = experiment_params["sparsity"] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # full result dictionary will be stored to ../results_and_plots/fullDict_results_non_retrained_vgg/this_experiments_result_varyingSparsity.json
    model_name = experiment_params["models"][0]["name"]
    network_difference = "DiffInit" if experiment_params["diff_weight_init"] else "DiffData"
    
    
    result_folder_name = f"./results_and_plots/fullDict_results_{model_name}"
    result_file_name = f"TEST_this_experiments_result_nets{network_difference}"
    result_file_path = result_folder_name+"/"+result_file_name
    if not os.path.exists(result_folder_name):
        os.makedirs(result_folder_name)     # create folder if doesnt exist before
    
    test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_path)
    
    '''
    # TESTING a list of fusion weights for fixed sparsity
    this_weight_list = [[round(x, 1), round(1-x,1)] for x in np.linspace(0, 1, 11)]
    this_sparsity_list = [0.7]  # sparsity 70% -> only 30% of original model size
    # full result dictionary will be stored to ../results_and_plots/fullDict_results_non_retrained_vgg/this_experiments_result_varyingFusionImportance.json
    result_file_name = "olins_temp_folder/TEST_this_experiments_result_varyingFusionImportance.json"
    test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_name)
    


    ##### RUNNING EXPERIMENTS FROM PAPER FOR A GIVEN LIST OF MODELS (input_model_names)
    to_analyse = "diffData"     #"diffData" or "diffInit"
    # fixing the sparsity, varying the weights per experiment. Execute experiments for multiple sparsities
    this_weight_list = [[round(x, 1), round(1-x,1)] for x in np.linspace(0, 1, 11)]

    
    this_sparsity_list = [0.9] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result_file_name = "olins_temp_folder/TEST_VGG_withoutRetraining_varyingFusionWeights_sparsity09_0vs1_"+to_analyse+".json"
    test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_name)

    this_sparsity_list = [0.8] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result_file_name = "olins_temp_folder/TEST_VGG_withoutRetraining_varyingFusionWeights_sparsity08_0vs1_"+to_analyse+".json"
    #test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_name)

    this_sparsity_list = [0.7] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result_file_name = "olins_temp_folder/TEST_VGG_withoutRetraining_varyingFusionWeights_sparsity07_0vs1_"+to_analyse+".json"
    test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_name)

    this_sparsity_list = [0.6] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result_file_name = "olins_temp_folder/TEST_VGG_withoutRetraining_varyingFusionWeights_sparsity06_0vs1_"+to_analyse+".json"
    #test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_name)

    this_sparsity_list = [0.5] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result_file_name = "olins_temp_folder/TEST_VGG_withoutRetraining_varyingFusionWeights_sparsity05_0vs1_"+to_analyse+".json"
    test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_name)

    this_sparsity_list = [0.4] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result_file_name = "results_non_retrained_vgg/TEST_VGG_withoutRetraining_varyingFusionWeights_sparsity04_0vs1_"+to_analyse+".json"
    #test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_name)

    this_sparsity_list = [0.3] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result_file_name = "olins_temp_folder/TEST_VGG_withoutRetraining_varyingFusionWeights_sparsity03_0vs1_"+to_analyse+".json"
    test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_name)

    this_sparsity_list = [0.2] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result_file_name = "olins_temp_folder/TEST_VGG_withoutRetraining_varyingFusionWeights_sparsity02_0vs1_"+to_analyse+".json"
    #test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_name)

    this_sparsity_list = [0.1] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result_file_name = "olins_temp_folder/TEST_VGG_withoutRetraining_varyingFusionWeights_sparsity01_0vs1_"+to_analyse+".json"
    test_multiple_settings(sparsity_list = this_sparsity_list, fusion_weights_list = this_weight_list, input_model_names=input_model_names, RESULT_FILE_PATH=result_file_name)
    '''