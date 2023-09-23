import copy
from train_2 import get_pretrained_model
from pruning_modified import prune_structured_new
from torch_pruning_new.optimal_transport import OptimalTransport
import torch_pruning_new as tp
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def find_ignored_layers(model_original, out_features):
    ignored_layers = []
    for m in model_original.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == out_features:
            ignored_layers.append(m)
    
    return ignored_layers


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
        num_workers=2,
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
        num_workers=2,
        pin_memory=True,
    )

    return {"train": train_loader, "test": val_loader}

def evaluate(input_model, loaders, gpu_id):
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

    input_model = input_model.cpu()
    return accuracy_accumulated / total

def model_to_weight_vec(model):
    weights_dict = model.cpu().state_dict()
    layer_weights_vec = torch.Tensor([])
    for _, layer_weights in weights_dict.items():
        layer_weights_vec = torch.cat((layer_weights_vec,layer_weights.flatten()), dim=0)
    return layer_weights_vec


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product


#Gram Schmidt Process
def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        for b in basis:
            dot_product = np.dot(v, b)
            v = v - dot_product * b.astype(v.dtype)  # Ensure v and b have the same data type
        if np.linalg.norm(v) > 1e-10:
            basis.append(v / np.linalg.norm(v))
    return np.array(basis)

# Create an orthonormal basis from three points
def create_orthonormal_basis(point1, point2, point3):
    vector1 = point2 - point1  # form the first vector
    vector2 = point3 - point1  # form the second vector
    return gram_schmidt([vector1, vector2])  # orthonormalize them

# Project a vector onto a basis
def project_to_basis_space(vector, basis):
    return np.dot(vector, basis.T)  # project vector onto the basis

# Project a vector back to k-dim space
def project_to_k_dim_space(components, basis):
    return np.dot(components, basis) 

# Create a 2D grid given the x_min, x_max, y_min,y_max
def create_2d_grid(point1, point2, point3, point4, n):

    # Generate linear spaces
    alpha = np.linspace(point1, point2, n)
    beta = np.linspace(point3, point4, n)
        
    return np.array(list(product(alpha, beta)))

def get_grid(points, n_grid=6, margin=0.2, plot=False):

    #Create Basis
    basis = create_orthonormal_basis(points[0], points[1], points[2])

    #Project to 2d
    two_dims = [project_to_basis_space(point, basis) for point in points]

    #Create Grid, make sure margin is kept
    x_min = min([point[0] for point in two_dims])
    x_max = max([point[0] for point in two_dims])
    y_min = min([point[1] for point in two_dims])
    y_max = max([point[1] for point in two_dims])
    y_range = y_max - y_min
    x_range = x_max - x_min

    #Create points in grid
    grid = create_2d_grid(x_min-margin*x_range, x_max+margin*x_range, y_min-margin*y_range, y_max+margin*y_range, n_grid)

    if plot:
        x_o, y_o = two_dims[0]
        x_1, y_1 = two_dims[1]
        x_2, y_2 = two_dims[2]
        xx,yy = grid.T
        plt.plot(xx,yy,'o')
        plt.scatter(x_o, y_o, c='r', s=50, label='Given Points')
        plt.scatter(x_1, y_1, c='r', s=50)
        plt.scatter(x_2, y_2, c='r', s=50)
        print(x_min, x_max, y_min, y_max)
        plt.show()

    #Compute the offset
    dif = points[0]-project_to_k_dim_space(two_dims[0], basis)

    #return grid in k-dim space
    return project_to_k_dim_space(grid, basis)+dif, grid

def weight_vec_to_model(weight_vec, target_model_shape):
    new_model = copy.deepcopy(target_model_shape)
    weights_dict = new_model.state_dict()
    layer_weights_vec = weight_vec
    vec_idx_counter = 0
    for key, layer_weights in weights_dict.items():
        layer_target_shape = layer_weights.shape
        layer_target_size = layer_weights.numel()
        this_layer_flattened = layer_weights_vec[vec_idx_counter:vec_idx_counter+layer_target_size]
        weight_vec_in_right_shape = this_layer_flattened.view(layer_target_shape)
        weights_dict[key].copy_(weight_vec_in_right_shape)
        vec_idx_counter += layer_target_size
    return new_model

import json
if __name__ == '__main__':


    '''    # Visualizing for k=3
        points = np.array([[0,3,0],[1,4,2], [0,0,2]])
        grid = get_grid(points)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*points.T, c='r', s=50, label='Given Points')
        ax.scatter(*grid.T, c='b', s=10, alpha=0.5, label='Grid Points')
        ax.legend()
        plt.show()
    '''

    model_name = "vgg11"
    dataset = "Cifar10"
    example_inputs = torch.randn(1, 3, 32, 32)
    out_features = 10
    file_name = "vgg11_bn_diff_weight_init_False_cifar10_eps300_A"
    gpu_id = -1
    

    config = dict(
        dataset=dataset,
        model=model_name,
        optimizer="SGD",
        optimizer_decay_at_epochs=[30, 60, 90, 120, 150, 180, 210, 240, 270],
        optimizer_decay_with_factor=2.0,
        optimizer_learning_rate=0.05,
        optimizer_momentum=0.9,
        optimizer_weight_decay=0.0001,
        batch_size=256,
        num_epochs=300,
        seed=42,
    )

    print("Loading model...")
    model_original,_ = get_pretrained_model(config, "./fusion_pruning_experiments/trained_models/vgg11_nobn_cifar10.checkpoint")
    #model_original = get_pretrained_models(model_name, file_name, gpu_id, 1, output_dim=out_features)[0]


    # Define the transform for the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to ResNet-18 input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])

    print("Loading dataset...")
    # Download and load the CIFAR-10 test dataset
    loaders = get_cifar10_data_loader()

    #print("Doing inference...")
    # Perform inference on the CIFAR-10 test dataset
    #print(evaluate(model_original, loaders, gpu_id=gpu_id))

    
    
    ot = OptimalTransport(gpu_id=gpu_id)
    meta_pruning_types = [None, ot]
    sparsities = [0.6]
    prune_types = ["l1"]
    n_grid = 200

    print("Doing pruning...")
    if_pruned_model = copy.deepcopy(model_original)
    prune_structured_new(
                        if_pruned_model,
                        None,
                        None,
                        example_inputs,
                        out_features,
                        prune_types[0],
                        gpu_id,
                        sparsity=sparsities[0],
                        prune_iter_steps=1,
                        optimal_transport=ot,
                        backward_pruning=True,
                        group_idxs=None,
                        train_fct=None,
                        dimensionality_preserving=True)
    
    pruned_model = copy.deepcopy(model_original)
    prune_structured_new(
                        pruned_model,
                        None,
                        None,
                        example_inputs,
                        out_features,
                        prune_types[0],
                        gpu_id,
                        sparsity=sparsities[0],
                        prune_iter_steps=1,
                        optimal_transport=None,
                        backward_pruning=True,
                        group_idxs=None,
                        train_fct=None,
                        dimensionality_preserving=True)
    
    print("Original Model:")  
    orig_perf = evaluate(model_original, loaders, gpu_id=gpu_id)
    orig_model_vec = model_to_weight_vec(model_original)
    print(orig_perf)
    
    print("Intra-Fusion Model:")
    if_perf = evaluate(if_pruned_model, loaders, gpu_id=gpu_id)
    if_model_vec = model_to_weight_vec(if_pruned_model)
    del if_pruned_model
    print(if_perf)

    print("Pruned Model:")
    pr_perf = evaluate(pruned_model, loaders, gpu_id=gpu_id)
    pr_model_vec = model_to_weight_vec(pruned_model)
    del pruned_model
    print(pr_perf)

    # see where these nD models lie in the 2D subspace
    basis = create_orthonormal_basis(if_model_vec.numpy(), pr_model_vec.numpy(), orig_model_vec.numpy())
    if_model_2D = project_to_basis_space(if_model_vec.numpy(), basis)
    pr_model_2D = project_to_basis_space(pr_model_vec.numpy(), basis)
    orig_model_2D = project_to_basis_space(orig_model_vec.numpy(), basis)

    if_dict = {"model_2D_vec": if_model_2D.tolist(), "model_perf": if_perf}
    pr_dict = {"model_2D_vec": pr_model_2D.tolist(), "model_perf": pr_perf}
    orig_dict = {"model_2D_vec": orig_model_2D.tolist(), "model_perf": orig_perf}

    results_dict = {"original": orig_dict, "pruned": pr_dict, "intra_fusion": if_dict}

    grid_nD_list, grid_2D_list = get_grid([if_model_vec.numpy(), pr_model_vec.numpy(), orig_model_vec.numpy()], n_grid=n_grid)
    
    performance_list = []
    grid_results_list = []
    for idx, diff_vec in enumerate(grid_nD_list):
        model_reconstructed = weight_vec_to_model(torch.from_numpy(diff_vec), model_original)
        perf = evaluate(model_reconstructed, loaders, gpu_id=gpu_id)
        del model_reconstructed
        performance_list.append(perf)
        print(perf)
        this_result_dict = {"model_2D_vec": grid_2D_list[idx].tolist(), "model_perf": perf}
        grid_results_list.append(this_result_dict)
        #print("This grid sample: ", evaluate(model_reconstructed, loaders, gpu_id=gpu_id))
    print("Done with loop")
    results_dict["grid"] = grid_results_list
    with open(f"./results_of_loss_landscape_{dataset}_{model_name}_grid{n_grid}.json", "w") as json_file:
        json.dump(results_dict, json_file, indent=4)
    print("Done with loop saving")

    #performance_list.extend([if_perf, pr_perf, orig_perf])
    #grid_2D_list = np.vstack((grid_2D_list, np.array([if_model_2D, pr_model_2D, orig_model_2D])))

    #plt.scatter(*zip(*grid_2D_list), c=performance_list, cmap='viridis', marker='o', s=100)
    #plt.show()

    #print("Original Model: ", evaluate(pruned_model, loaders, gpu_id=gpu_id))
    #print("Vec reconstruction: ", evaluate(model_reconstructed, loaders, gpu_id=gpu_id))

    """print(if_model_vec.shape)
    print(pr_model_vec.shape)
    print(orig_model_vec.shape)

    print(if_model_vec)
    print(pr_model_vec)
    print(orig_model_vec)




    for ((name_orig, module_orig), (name, module)) in list(zip(model_original.named_modules(), pruned_model.named_modules())):
        if isinstance(module_orig, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"{module_orig.weight.shape} -> {module.weight.shape}")"""





