import functools
import itertools
import torch
import numpy as np
import ot
import copy
from torch import linalg as LA
from fusion_utils import preprocess_parameters
from sklearn.cluster import KMeans
import torch.nn as nn
from models import get_model

#from scipy.optimize import linear_sum_assignment # Could accomplish the same as OT with Hungarian Algorithm

# get_histogram creates uniform historgram, i.e. [1/cardinality, 1/cardinality, ...]
def get_histogram(cardinality, indices, add=False):

    if add == False:
        return np.ones(cardinality)/cardinality # uniform probability distribution

    else:
        result = np.ones(cardinality)
        for indice in indices:
            result[indice] = cardinality/indices.size()[0]

        return result/np.sum(result)

def normalize_vector(coordinates, eps=1e-9):
    norms = torch.norm(coordinates, dim=-1, keepdim=True)
    return coordinates / (norms + eps)

# compute_euclidian_distance_matrix computes a matrix where c[i, j] corresponds to the euclidean distance of x[i] and y[j]
def compute_euclidian_distance_matrix(x, y, p=2, squared=True): # For some reason TA prefers squared to be True
    x_col = x.unsqueeze(1).cpu()
    y_lin = y.unsqueeze(0).cpu()
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    if not squared:
        c = c ** (1/2)
    return c

# get_ground_metric computes the cost matrix
# if bias is present, the bias will be appended to the weight matrix and subsequently used to calculate the cost
# cost matrix is based on the weights, not the activations
def get_ground_metric(coordinates1, coordinates2, bias1, bias2):
    if bias1 != None: # and bias2 != None
        assert bias2 != None
        coordinates1 = torch.cat((coordinates1, bias1.view(bias1.shape[0], -1)), 1)
        coordinates2 = torch.cat((coordinates2, bias2.view(bias2.shape[0], -1)), 1)
    coordinates1 = normalize_vector(coordinates1)
    coordinates2 = normalize_vector(coordinates2)
    return compute_euclidian_distance_matrix(coordinates1, coordinates2)

# create_network_from_params creates a network given the list of weights
def create_network_from_params(reference_model, param_list, gpu_id = -1,sparsity=1.0):
    model = copy.deepcopy(reference_model)

    assert len(list(model.parameters())) == len(param_list) # Assumption: We are fusing into a model of same architecture

    if gpu_id != -1:
        model = model.cuda(gpu_id)
    
    layer_idx = 0
    model_state_dict = model.state_dict()

    for key in model_state_dict.keys():
        model_state_dict[key] = param_list[layer_idx]
        layer_idx += 1
    
    model.load_state_dict(model_state_dict)

    return model

def create_network_from_parameters(reference_model, param_list, gpu_id = -1):
    model = copy.deepcopy(reference_model)
    
    model_state_dict = model.state_dict()
    keys = list(model_state_dict.keys())
    
    idx = -1

    for layer in param_list:
        idx += 1
        model_state_dict[keys[idx]] = layer.weight
        if layer.bias != None:
            idx += 1
            model_state_dict[keys[idx]] = layer.bias
        if layer.bn:
            if layer.bn_gamma != None:
                idx += 1
                model_state_dict[keys[idx]] = layer.bn_gamma
                idx += 1
                model_state_dict[keys[idx]] = layer.bn_beta
            idx += 1
            model_state_dict[keys[idx]] = layer.bn_mean
            idx += 1
            model_state_dict[keys[idx]] = layer.bn_var

            idx += 1
            model_state_dict[keys[idx]] = torch.Tensor([1])
    
    model.load_state_dict(model_state_dict)

    return model




def update_iterators(iterators):
    for iter in iterators:
        next(iter)

def find_smallest_model(networks):
    num_models = len(networks)
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    layer_iters = [networks[i].named_parameters() for i in range(num_models)]
    smaller_idx = 0
    found_smallest = False
    for idx in range(num_layers):
        _, avg_layer = next(layer_iters[0])
        for idx_model in range(1, num_models): 
            _, layer = next(layer_iters[idx_model])
            if layer.shape[0] < avg_layer.shape[0]:
                smaller_idx = idx_model
                found_smallest = True
            elif layer.shape[0] != avg_layer.shape[0]:
                found_smallest = True
        if found_smallest:
            return smaller_idx
    
    return smaller_idx

def comparator(model0_args, model1_args):
    model0, model1 = model0_args[0], model1_args[0]
    accuracy0, accuracy1 = model0_args[1], model1_args[1]
    importance0, importance1 = model0_args[2], model1_args[2]

    for _, ((_, layer0_weight), (_, layer1_weight)) in \
            enumerate(zip(model0.named_parameters(), model1.named_parameters())):
        if layer0_weight.shape[0] < layer1_weight.shape[0]:
            return -1
        elif layer0_weight.shape[0] > layer1_weight.shape[0]:
            return 1
    
    if importance0 > importance1:
        return -1
    elif importance0 < importance1:
        return 1
    
    if accuracy0 > accuracy1:
        return -1
    elif accuracy0 < accuracy1:
        return 1
    else:
        return 0

def IntraFusion_Clustering(network, sparsity = 0.65, gpu_id = -1, metric=-1, eps=1e-7, resnet=False, output_dim=10):
    num_layers = len(list(network.parameters()))
    avg_aligned_layers = []
    T_var = None
    bias = False
    bias_weight = None

    softmax = torch.nn.Softmax(dim=0)

    for idx, (layer_name, layer_weight) in enumerate(network.named_parameters()):
        if bias:
            # If in the last layer we detected bias, this layer will be the bias layer we handled before, so we can just skip it
            bias=False
            continue

        # Check if this current layer has a bias
        if (idx != num_layers-1):
            next_layer = next(itertools.islice(network.named_parameters(), idx+1, None))
            bias = True if "bias" in next_layer[0] else False
            bias_weight = next_layer[1] if bias else None
        else:
            bias = False
        
        print("idx {} and layer {}".format(idx, layer_name))
        print("Bias: {}".format(bias))
        #assert fc_layer0_weight.shape == fc_layer1_weight.shape

        #mu_cardinality = fc_layer0_weight.shape[0]
        #nu_cardinality = fc_layer1_weight.shape[0]

        layer_shape = layer_weight.shape



        if len(layer_shape) > 2:
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            # fc_layer0_weight_data has shape: (*out_channels, #in_channels, height*width)
            layer_weight_data = layer_weight.data.view(layer_weight.shape[0], layer_weight.shape[1], -1)
        else:
            is_conv = False
            layer_weight_data = layer_weight.data
        

        amount_pruned = round(layer_shape[0]*(1-sparsity))
        amount_pruned = 1 if amount_pruned == 0 else amount_pruned # Make sure we're not cancelling a whole layer

        norm_layer = None
        indices = None
        basis_weight = None
        basis_bias = None

        if idx == 0:
            norm_layer = LA.norm(layer_weight_data.view(layer_weight_data.shape[0], -1), ord=metric, dim=1)
            _, indices = torch.topk(norm_layer, amount_pruned)
            basis_bias = torch.index_select(bias_weight, 0, indices) if bias else None

            if is_conv:
                # input to ground_metric has shape: (#out_channels, #in_channels*height*width)
                layer_flattened = layer_weight_data.view(layer_weight_data.shape[0], -1)
                basis_weight = torch.index_select(layer_flattened, 0, indices)

                M = get_ground_metric(layer_flattened,
                                basis_weight, bias_weight, basis_bias)
            else:
                basis_weight = torch.index_select(layer_weight_data, 0, indices)
                M = get_ground_metric(layer_weight_data, basis_weight, bias_weight, basis_bias)

            aligned_wt = layer_weight_data
        else:
            if is_conv:
                T_var_conv = T_var.unsqueeze(0).repeat(layer_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(layer_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

            else:
                if layer_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    layer_unflattened = layer_weight.data.view(layer_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        layer_unflattened,
                        T_var.unsqueeze(0).repeat(layer_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)

                else:
                    aligned_wt = torch.matmul(layer_weight.data, T_var)
            
            aligned_wt = aligned_wt.reshape(aligned_wt.shape[0], -1)
            norm_layer = LA.norm(aligned_wt, ord=metric, dim=1)
            _, indices = torch.topk(norm_layer, amount_pruned)
            basis_weight = torch.index_select(aligned_wt, 0, indices)
            basis_bias = torch.index_select(bias_weight, 0, indices) if bias else None

            M = get_ground_metric(aligned_wt,
                basis_weight, bias_weight, basis_bias)
        

        mu = get_histogram(basis_weight.shape[0], indices = None)

        nu = get_histogram(layer_shape[0], indices = indices, add=True)
        #nu = norm_layer.cpu().numpy()/np.sum(norm_layer.cpu().numpy())
        nu /= np.sum(nu)

        #nu = softmax(norm_layer).cpu().numpy()
        #nu = torch.argsort(norm_layer, dim=- 1, descending=False, stable=True).cpu().numpy()
        #nu = nu/np.sum(nu)

        cpuM = M.data.cpu().numpy() # POT does not accept array on GPU

        ########################## CLUSTERING STARTS HERE ##################################################
        histo = torch.from_numpy(get_histogram(layer_shape[0], indices = indices, add=False)) # Every data point gets the same weight
        #histo = torch.from_numpy(get_histogram(layer_shape[0], indices = indices, add=True)) # The top amount_pruned data points get the most weight
        #histo = norm_layer # The data points are weighted by their norm
        remove_outlier = True
        if remove_outlier:
            # We set the indices of histo to 0 if it's an outlier. That way it has no effect in the clustering or in the weighted averaging
            model = KMeans(n_clusters=amount_pruned, random_state=0).fit(aligned_wt.view(aligned_wt.shape[0], -1).numpy(), sample_weight=histo)
            labels = model.labels_
            centroids = model.cluster_centers_
            
            M = get_ground_metric(torch.from_numpy(centroids),
                    aligned_wt.view(aligned_wt.shape[0], -1), None, None)
            
            T = torch.zeros(amount_pruned, aligned_wt.shape[0])
            indice_col = np.arange(0, labels.shape[0], 1, dtype=int)

            T[torch.from_numpy(labels).long(), indice_col] = 1

            MT = M*T

            MT[MT == 0] = torch.nan
            quantile = torch.unsqueeze(MT.nanquantile(0.5, dim=1), 1).expand(MT.shape)
            MT = (MT > quantile).nonzero() # Throw away

            indice_outlier = MT[:,1]
            # In the next line we make sure we don't remove so many outliers such that there are less data points than clusters
            indice_outlier = indice_outlier[:amount_pruned] if aligned_wt.shape[0]-indice_outlier.shape[0] < amount_pruned else indice_outlier
            histo[indice_outlier] = 0.0


        model = KMeans(n_clusters=amount_pruned, random_state=0).fit(aligned_wt.view(aligned_wt.shape[0], -1).numpy(), sample_weight=histo)
        

        labels = torch.from_numpy(model.labels_)
        T = torch.zeros(amount_pruned, aligned_wt.shape[0])
        indice_col = np.arange(0, labels.shape[0], 1, dtype=int)
        T[labels.long(), indice_col] = 1
        T = T.t()
        T *= torch.unsqueeze(histo, 1).expand(norm_layer.shape[0], amount_pruned)
        T = T.numpy()
        ########################## CLUSTERING ENDS HERE ##################################################


        if gpu_id!=-1:
            T_var = torch.from_numpy(T).cuda(gpu_id).float()
        else:
            T_var = torch.from_numpy(T).float()

        # ----- Assumption: correction = TRUE, proper_marginals = FALSE ---------
        if gpu_id != -1:
            marginals = torch.ones(T_var.shape[1]).cuda(gpu_id) / T_var.shape[0]
        else:
            marginals = torch.ones(T_var.shape[1]) / T_var.shape[0]
        marginals = torch.diag(1.0/(marginals + eps))  # take inverse
        T_var = torch.matmul(T_var, marginals)

        T_var = T_var / T_var.sum(dim=0)
        # -----------------------------------------------------------------------
        # ---- Assumption: Past correction = True (Anything else doesn't really make sense?)
        geometric_fc = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1)) if idx != num_layers-1 else aligned_wt

        if is_conv:
            print(geometric_fc.shape[1]/(layer_shape[2]*layer_shape[3]))
            geometric_fc = geometric_fc.view(torch.Size([geometric_fc.shape[0], int(geometric_fc.shape[1]/(layer_shape[2]*layer_shape[3])), layer_shape[2], layer_shape[3]]))

        avg_aligned_layers.append(geometric_fc)
        if bias:
            geometric_bias = torch.matmul(T_var.t(), bias_weight.view(bias_weight.shape[0], -1)).flatten() if idx != num_layers-1 else bias_weight
            avg_aligned_layers.append(geometric_bias)
        
        mu = None
        nu = None
        layer_weight_data = None
        basis_weight = None
        layer_flattened = None
        basis_weight = None
        basis_bias = None
        norm_layer = None


        M = None
        cpuM = None
        marginals = None
        geometric_fc = None
        bias_weight = None
        geometric_bias = None
        Maligned_wt = None 

    
    return create_network_from_params(gpu_id=gpu_id, param_list=avg_aligned_layers, reference_model = get_model(model_name="vgg11", sparsity=1-sparsity, output_dim=output_dim))


def MSF(network, sparsity = 0.65, gpu_id = -1, metric=-1, eps=1e-7, resnet=False, output_dim=10):
    num_layers = len(list(network.parameters()))
    avg_aligned_layers = []
    T_var = None
    bias = False
    bias_weight = None

    softmax = torch.nn.Softmax(dim=0)

    for idx, (layer_name, layer_weight) in enumerate(network.named_parameters()):
        if bias:
            # If in the last layer we detected bias, this layer will be the bias layer we handled before, so we can just skip it
            bias=False
            continue

        # Check if this current layer has a bias
        if (idx != num_layers-1):
            next_layer = next(itertools.islice(network.named_parameters(), idx+1, None))
            bias = True if "bias" in next_layer[0] else False
            bias_weight = next_layer[1] if bias else None
        else:
            bias = False
        
        print("idx {} and layer {}".format(idx, layer_name))
        print("Bias: {}".format(bias))
        #assert fc_layer0_weight.shape == fc_layer1_weight.shape

        #mu_cardinality = fc_layer0_weight.shape[0]
        #nu_cardinality = fc_layer1_weight.shape[0]

        layer_shape = layer_weight.shape



        if len(layer_shape) > 2:
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            # fc_layer0_weight_data has shape: (*out_channels, #in_channels, height*width)
            layer_weight_data = layer_weight.data.view(layer_weight.shape[0], layer_weight.shape[1], -1)
        else:
            is_conv = False
            layer_weight_data = layer_weight.data
        

        amount_pruned = round(layer_shape[0]*(1-sparsity))
        amount_pruned = 1 if amount_pruned == 0 else amount_pruned # Make sure we're not cancelling a whole layer

        norm_layer = None
        indices = None
        basis_weight = None
        basis_bias = None

        if idx == 0:
            norm_layer = LA.norm(layer_weight_data.view(layer_weight_data.shape[0], -1), ord=metric, dim=1)
            _, indices = torch.topk(norm_layer, amount_pruned)
            basis_bias = torch.index_select(bias_weight, 0, indices) if bias else None

            if is_conv:
                # input to ground_metric has shape: (#out_channels, #in_channels*height*width)
                layer_flattened = layer_weight_data.view(layer_weight_data.shape[0], -1)
                basis_weight = torch.index_select(layer_flattened, 0, indices)

                M = get_ground_metric(layer_flattened,
                                basis_weight, bias_weight, basis_bias)
            else:
                basis_weight = torch.index_select(layer_weight_data, 0, indices)
                M = get_ground_metric(layer_weight_data, basis_weight, bias_weight, basis_bias)

            aligned_wt = layer_weight_data
        else:
            if is_conv:
                T_var_conv = T_var.unsqueeze(0).repeat(layer_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(layer_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

            else:
                if layer_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    layer_unflattened = layer_weight.data.view(layer_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        layer_unflattened,
                        T_var.unsqueeze(0).repeat(layer_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)

                else:
                    aligned_wt = torch.matmul(layer_weight.data, T_var)
            
            aligned_wt = aligned_wt.reshape(aligned_wt.shape[0], -1)
            norm_layer = LA.norm(aligned_wt, ord=metric, dim=1)
            _, indices = torch.topk(norm_layer, amount_pruned)
            basis_weight = torch.index_select(aligned_wt, 0, indices)
            basis_bias = torch.index_select(bias_weight, 0, indices) if bias else None

            M = get_ground_metric(aligned_wt,
                basis_weight, bias_weight, basis_bias)
        

        mu = get_histogram(basis_weight.shape[0], indices = None)

        #nu = norm_layer.cpu().numpy()/np.sum(norm_layer.cpu().numpy())
        nu = get_histogram(layer_shape[0], indices = indices, add=True)

        #nu = softmax(norm_layer).cpu().numpy()
        #nu = torch.argsort(norm_layer, dim=- 1, descending=False, stable=True).cpu().numpy()
        #nu = nu/np.sum(nu)

        cpuM = M.data.cpu().numpy() # POT does not accept array on GPU


        T = ot.emd(nu, mu, cpuM)

        if gpu_id!=-1:
            T_var = torch.from_numpy(T).cuda(gpu_id).float()
        else:
            T_var = torch.from_numpy(T).float()

        # ----- Assumption: correction = TRUE, proper_marginals = FALSE ---------
        if gpu_id != -1:
            marginals = torch.ones(T_var.shape[1]).cuda(gpu_id) / T_var.shape[0]
        else:
            marginals = torch.ones(T_var.shape[1]) / T_var.shape[0]
        marginals = torch.diag(1.0/(marginals + eps))  # take inverse
        T_var = torch.matmul(T_var, marginals)
        #print(T_var.shape)
        #print(T_var[:,0])
        T_var = T_var / T_var.sum(dim=0)
        #print(T_var[:,0])
        # -----------------------------------------------------------------------
        # ---- Assumption: Past correction = True (Anything else doesn't really make sense?)
        geometric_fc = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1)) if idx != num_layers-1 else aligned_wt
        #print(geometric_fc.shape)
        #print("layer_shape: ", layer_shape)

        if is_conv:
            #print(geometric_fc.shape[1]/(layer_shape[2]*layer_shape[3]))
            geometric_fc = geometric_fc.view(torch.Size([geometric_fc.shape[0], int(geometric_fc.shape[1]/(layer_shape[2]*layer_shape[3])), layer_shape[2], layer_shape[3]]))

        avg_aligned_layers.append(geometric_fc)
        if bias:
            geometric_bias = torch.matmul(T_var.t(), bias_weight.view(bias_weight.shape[0], -1)).flatten() if idx != num_layers-1 else bias_weight
            avg_aligned_layers.append(geometric_bias)
        
        mu = None
        nu = None
        layer_weight_data = None
        basis_weight = None
        layer_flattened = None
        basis_weight = None
        basis_bias = None
        norm_layer = None


        M = None
        cpuM = None
        marginals = None
        geometric_fc = None
        bias_weight = None
        geometric_bias = None
        Maligned_wt = None 

    
    return create_network_from_params(gpu_id=gpu_id, param_list=avg_aligned_layers, reference_model = get_model(model_name="vgg11", sparsity=1-sparsity, output_dim=output_dim))

def fusion_bn(networks, gpu_id = -1, accuracies=None, importance=None, eps=1e-7, resnet=False, activation_based=False, train_loader=False, model_name=None, num_samples=200):
    """
    fusion fuses arbitrary many models into the model that is the smallest
    :param networks: A list of networks to be fused
    :param accuracies: A list of accuracies of the networks. The code will order the networks with respect to the accuracies for optimal accuracy
    :param importance: A list of floats. If importance = [0.9, 0.1], then linear combination of weights will be: network[0]*0.9 + network[1]*0.1
    :return: the fused model
    """ 

    num_models = len(networks)
    T_var_list = [None]*num_models
    skip_T_var_list = [None]*num_models
    skip_T_var_idx_list = [-1]*num_models
    residual_T_var_list = [None]*num_models
    residual_T_var_idx_list = [-1]*num_models

    if accuracies == None:
        accuracies = [0]*num_models

    importance = [1]*num_models if importance == None else importance
    networks_zip = sorted(list(zip(networks, accuracies, importance)), key=functools.cmp_to_key(comparator))
    networks = [network_zip[0] for network_zip in networks_zip]
    importance = [network_zip[2] for network_zip in networks_zip]
    importance = torch.tensor(importance)
    sum = torch.sum(importance)
    importance = importance *(importance.shape[0]/sum)
    if gpu_id != -1:
        for model in networks:
            model = model.cuda(gpu_id)

    fusion_layers = preprocess_parameters(networks, activation_based=activation_based, train_loader=train_loader, model_name=model_name, gpu_id=gpu_id, num_samples=200)
    num_layers = len(fusion_layers)

    for idx in range(num_layers):

        fusion_layers[idx] = list(fusion_layers[idx])

        #fusion_layers[idx][0].to(gpu_id)
        avg_layer = fusion_layers[idx][0]

        mu_cardinality = avg_layer.weight.shape[0]
        mu = get_histogram(mu_cardinality, None)

        is_conv = avg_layer.is_conv

        for idx_m, fusion_layer in enumerate(fusion_layers[idx][1:]):
            #fusion_layer.to(gpu_id)
            idx_model = idx_m + 1
            T_var = T_var_list[idx_model]
            skip_T_var = skip_T_var_list[idx_model]
            skip_T_var_idx = skip_T_var_idx_list[idx_model]
            residual_T_var = residual_T_var_list[idx_model]
            residual_T_var_idx = residual_T_var_idx_list[idx_model]

            nu_cardinality = fusion_layer.weight.shape[0]
            layer_shape = fusion_layer.weight.shape

            if idx != 0:
                if resnet and is_conv:
                    assert len(layer_shape) == 4
                    # save skip_level transport map if there is block ahead
                    if layer_shape[1] != layer_shape[0]:
                        if not (layer_shape[2] == 1 and layer_shape[3] == 1):
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:

                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var

                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2

                
                fusion_layer.align_weights(T_var)
            
            M = get_ground_metric(fusion_layer.create_comparison_vec(), avg_layer.create_comparison_vec(), None, None)

            nu = get_histogram(nu_cardinality, None)

            cpuM = M.data.cpu().numpy() # POT does not accept array on GPU

            T = ot.emd(nu, mu, cpuM)        

            if gpu_id != -1:
                T_var = torch.from_numpy(T).cuda(gpu_id).float()
            else:
                T_var = torch.from_numpy(T).float()
            

            if gpu_id != -1:
                marginals = torch.ones(T_var.shape[1]).cuda(gpu_id) / T_var.shape[0]
            else:
                marginals = torch.ones(T_var.shape[1]) / T_var.shape[0]
            marginals = torch.diag(1.0/(marginals + eps))  # take inverse

            T_var = torch.matmul(T_var, marginals)
            T_var = T_var / T_var.sum(dim=0)


            fusion_layer.permute_parameters(T_var)

            avg_layer.update_weights(fusion_layer, torch.sum(importance[:idx_model]), importance[idx_model])


            T_var_list[idx_model] = T_var
            skip_T_var_list[idx_model] = skip_T_var
            skip_T_var_idx_list[idx_model] = skip_T_var_idx
            residual_T_var_list[idx_model] = residual_T_var
            residual_T_var_idx_list[idx_model] = residual_T_var_idx
            fusion_layers[idx][idx_model] = None
            #fusion_layer.to(-1)
            del fusion_layer
            mu = None
            nu = None
            M = None
            T = None
            T_var = None
            skip_T_var = None
            skip_t_var_idx = None
            residual_T_var = None
            residual_T_var_idx = None


            cpuM = None
            marginals = None


    return create_network_from_parameters(gpu_id=gpu_id, param_list=[layers[0] for layers in fusion_layers], reference_model = networks[0])

def merge_conv_bn(conv, bn):
    """
    Merge a Conv2D layer and a BatchNorm layer into a single Conv2D layer.
    
    Args:
        conv (nn.Conv2d): The convolutional layer to merge.
        bn (nn.BatchNorm2d): The batch normalization layer to merge.
    
    Returns:
        nn.Conv2d: The merged Conv2D layer.
    """
    merged_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                            conv.padding, conv.dilation, conv.groups, conv.bias is not None)
    merged_conv.weight = nn.Parameter(conv.weight * bn.weight[:, None, None, None] / torch.sqrt(bn.running_var + bn.eps)[:, None, None, None])
    merged_conv.bias = nn.Parameter((conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)) - bn.running_mean * bn.weight / torch.sqrt(bn.running_var + bn.eps) + bn.bias)
    
    return merged_conv

def merge_layers(modules):
    """
    Merge Conv2D and BatchNorm layers in a list of PyTorch modules.
    
    Args:
        modules (list): A list of PyTorch modules.
    
    Returns:
        list: A list of PyTorch modules with Conv2D and BatchNorm layers merged.
    """
    new_modules = []
    skip_next = False
    for i, m in enumerate(modules):
        if skip_next:
            skip_next = False
            continue
        if i < len(modules) - 1 and isinstance(m, nn.Conv2d) and isinstance(modules[i + 1], nn.BatchNorm2d):
            new_modules.append(merge_conv_bn(m, modules[i + 1]))
            skip_next = True
        else:
            new_modules.append(m)
    return new_modules

def merge_conv_bn_layers(net):
    """
    Create a deep copy of a PyTorch neural network and merge Conv2D layers with the following BatchNorm layers.
    
    Args:
        net (nn.Module): The neural network to copy and modify.
    
    Returns:
        nn.Module: A deep copy of the input network with Conv2D and BatchNorm layers merged.
    """
    new_net = copy.deepcopy(net)
    
    for name, module in new_net.named_modules():
        if isinstance(module, nn.Sequential):
            new_modules = merge_layers(list(module.children()))
            new_seq = nn.Sequential(*new_modules)
            
            name_parts = name.rsplit('.', 1)
            if len(name_parts) == 2:
                parent, key = name_parts
                parent_module = dict(new_net.named_modules())[parent]
                parent_module._modules[key] = new_seq
            else:
                key = name_parts[0]
                new_net._modules[key] = new_seq
                
    return new_net


def fusion_bn_alt(networks, gpu_id = -1, accuracies=None, importance=None, eps=1e-7, resnet=False):
    networks_without_bn = [merge_conv_bn_layers(net) for net in networks]
    print(networks_without_bn[0])
    torch.cuda.empty_cache()
    return fusion(networks_without_bn, gpu_id, accuracies, importance, eps, resnet)


def fusion(networks, gpu_id = -1, accuracies=None, importance=None, eps=1e-7, resnet=False):
    """
    fusion fuses arbitrary many models into the model that is the smallest
    :param networks: A list of networks to be fused
    :param accuracies: A list of accuracies of the networks. The code will order the networks with respect to the accuracies for optimal accuracy
    :param importance: A list of floats. If importance = [0.9, 0.1], then linear combination of weights will be: network[0]*0.9 + network[1]*0.1
    :return: the fused model
    """ 

    print("RESNET IS: ", resnet)
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))

    num_models = len(networks)
    avg_aligned_layers = []
    T_var_list = [None]*num_models
    skip_T_var_list = [None]*num_models
    skip_T_var_idx_list = [-1]*num_models
    residual_T_var_list = [None]*num_models
    residual_T_var_idx_list = [-1]*num_models

    bias = False
    bias_weight = None
    avg_bias_weight = None

    if accuracies == None:
        accuracies = [0]*num_models
    print("accuracies are: ", accuracies)
    print("importance beginning is: ", importance)
    print(list(zip(networks, accuracies)))

    importance = [1]*num_models if importance == None else importance
    networks_zip = sorted(list(zip(networks, accuracies, importance)), key=functools.cmp_to_key(comparator))
    networks = [network_zip[0] for network_zip in networks_zip]
    importance = [network_zip[2] for network_zip in networks_zip]
    importance = torch.tensor(importance)
    sum = torch.sum(importance)
    print("sum is: ", sum)
    importance = importance *(importance.shape[0]/sum)
    print("importance: ", importance)

    layer_iters = [networks[i].named_parameters() for i in range(num_models)]
        

    for idx in range(num_layers):
        if bias:
            # If in the last layer we detected bias, this layer will be the bias layer we handled before, so we can just skip it
            bias=False
            update_iterators(layer_iters)
            continue
        
        if (idx != num_layers-1):
            bias = True if "bias" in next(itertools.islice(networks[0].named_parameters(), idx+1, None))[0] else False
            avg_bias_weight = next(itertools.islice(networks[0].named_parameters(), idx+1, None))[1] if bias else None
        else:
            bias = False
            avg_bias_weight = None

        avg_layer_name, avg_layer = next(layer_iters[0])
        layer_shape = avg_layer.shape
        mu_cardinality = avg_layer.shape[0]
        mu = get_histogram(mu_cardinality, None)
        is_conv = len(layer_shape) > 2

        print("idx {} and layer {}".format(idx, avg_layer_name))
        print("Bias: {}".format(bias))
        
        
        for idx_model in range(1, num_models): # We skip the first model because we align our models according to the first model
            T_var = T_var_list[idx_model]
            skip_T_var = skip_T_var_list[idx_model]
            skip_T_var_idx = skip_T_var_idx_list[idx_model]
            residual_T_var = residual_T_var_list[idx_model]
            residual_T_var_idx = residual_T_var_idx_list[idx_model]
            _, layer = next(layer_iters[idx_model])

            bias_weight = next(itertools.islice(networks[idx_model].named_parameters(), idx+1, None))[1] if bias else None


            #assert avg_layer.shape == layer.shape

            nu_cardinality = layer.shape[0]

            if is_conv:
                # For convolutional layers, it is (#out_channels, #in_channels, height, width)
                # fc_layer0_weight_data has shape: (*out_channels, #in_channels, height*width)
                avg_layer_data = avg_layer.data.view(avg_layer.shape[0], avg_layer.shape[1], -1)
                layer_data = layer.data.view(layer.shape[0], layer.shape[1], -1)
            else:
                avg_layer_data = avg_layer.data
                layer_data = layer.data

            if idx == 0:
                if is_conv:
                    # input to ground_metric has shape: (#out_channels, #in_channels*height*width)
                    avg_layer_flattened = avg_layer_data.view(avg_layer_data.shape[0], -1)
                    layer_flattened = layer_data.view(layer_data.shape[0], -1)
                    M = get_ground_metric(layer_flattened,avg_layer_flattened, bias_weight, avg_bias_weight)
                else:
                    M = get_ground_metric(layer_data, avg_layer_data, bias_weight, avg_bias_weight)

                aligned_wt = layer_data
            else:
                if is_conv:
                    if resnet:
                        assert len(layer_shape) == 4
                        # save skip_level transport map if there is block ahead
                        if layer_shape[1] != layer_shape[0]:
                            if not (layer_shape[2] == 1 and layer_shape[3] == 1):
                                print(f'saved skip T_var at layer {idx} with shape {layer_shape}')
                                skip_T_var = T_var.clone()
                                skip_T_var_idx = idx
                            else:
                                print(
                                    f'utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}')
                                # if it's a shortcut (128, 64, 1, 1)
                                residual_T_var = T_var.clone()
                                residual_T_var_idx = idx  # use this after the skip
                                T_var = skip_T_var
                            print("shape of previous transport map now is", T_var.shape)
                        else:
                            if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                                T_var = (T_var + residual_T_var) / 2
                                print("averaging multiple T_var's")
                            else:
                                print("doing nothing for skips")
                    T_var_conv = T_var.unsqueeze(0).repeat(layer_data.shape[2], 1, 1)
                    aligned_wt = torch.bmm(layer_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
                    
                    M = get_ground_metric(
                        aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                        avg_layer_data.view(avg_layer_data.shape[0], -1), 
                        bias_weight, avg_bias_weight
                    )
                else:
                    if layer.data.shape[1] != T_var.shape[0]:
                        # Handles the switch from convolutional layers to fc layers
                        layer_unflattened = layer.data.view(layer.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                        aligned_wt = torch.bmm(
                            layer_unflattened,
                            T_var.unsqueeze(0).repeat(layer_unflattened.shape[0], 1, 1)
                        ).permute(1, 2, 0)
                        aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)

                    else:
                        aligned_wt = torch.matmul(layer, T_var)
                    M = get_ground_metric(aligned_wt, avg_layer.data, bias_weight, avg_bias_weight)
            
            nu = get_histogram(nu_cardinality, None)

            
            cpuM = M.data.cpu().numpy() # POT does not accept array on GPU
            print(cpuM.shape)
            print(mu.shape)
            print(nu.shape)
            T = ot.emd(nu, mu, cpuM)

            if gpu_id != -1:
                T_var = torch.from_numpy(T).cuda(gpu_id).float()
            else:
                T_var = torch.from_numpy(T).float()
            
            # ----- Assumption: correction = TRUE, proper_marginals = FALSE ---------
            if gpu_id != -1:
                marginals = torch.ones(T_var.shape[1]).cuda(gpu_id) / T_var.shape[0]
            else:
                marginals = torch.ones(T_var.shape[1]) / T_var.shape[0]
            marginals = torch.diag(1.0/(marginals + eps))  # take inverse
            T_var = torch.matmul(T_var, marginals)
            T_var = T_var / T_var.sum(dim=0)

            # -----------------------------------------------------------------------

            # ---- Assumption: Past correction = True (Anything else doesn't really make sense?)
            t_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))

            # ---------------------------------------------------------------------------------
            # t_model corresponds to aligned weights according to weights of model 1

            # Averaging of aligned weights (Could potentially also favor one model over the other)
            avg_layer = (t_model*importance[idx_model] + avg_layer_data.view(avg_layer_data.shape[0], -1)*torch.sum(importance[:idx_model]))/(torch.sum(importance[:idx_model])+importance[idx_model])

            if is_conv and layer_shape != avg_layer.shape:
                avg_layer = avg_layer.view(layer_shape)

            if bias:
                t_bias_aligned = torch.matmul(T_var.t(), bias_weight.view(bias_weight.shape[0], -1)).flatten()
                avg_bias_weight = (t_bias_aligned*importance[idx_model] + avg_bias_weight*torch.sum(importance[:idx_model]))/(torch.sum(importance[:idx_model])+importance[idx_model])

            T_var_list[idx_model] = T_var
            skip_T_var_list[idx_model] = skip_T_var
            skip_T_var_idx_list[idx_model] = skip_T_var_idx
            residual_T_var_list[idx_model] = residual_T_var
            residual_T_var_idx_list[idx_model] = residual_T_var_idx
        
        avg_aligned_layers.append(avg_layer)
        if bias:
            avg_aligned_layers.append(avg_bias_weight)
        
        cpuM = None 
        M = None
        T = None
        avg_layer_data = None 
        layer_data = None

    return create_network_from_params(gpu_id=gpu_id, param_list=avg_aligned_layers, reference_model = networks[0])

# def fusion_bn_alt(networks, gpu_id = -1, accuracies=None, importance=None, eps=1e-7, resnet=False):
#     def scale_bn(net):
#         """
#         Create a deep copy of a PyTorch neural network and remove all BatchNorm layers from the copy.
        
#         Args:
#             net (nn.Module): The neural network to copy and modify.
        
#         Returns:
#             nn.Module: A deep copy of the input network with all BatchNorm layers removed.
#         """
#         # Create a deep copy of the input network.
#         new_net = copy.deepcopy(net)



#         # Remove BatchNorm layers from the copy.
#         for i, key in enumerate(new_net._modules):
#             m1 = new_net._modules[key]
#             if isinstance(m1, nn.Sequential):
#                 new_net._modules[list(new_net._modules.keys())[i]] = scale_bn(m1)
#             elif isinstance(m1, nn.BatchNorm2d) or isinstance(m1, nn.BatchNorm1d):
#                 idx = i
#                 while idx >= 0:
#                     m2 = list(new_net.modules())[idx]
#                     if isinstance(m2, nn.Linear) or isinstance(m2, nn.Conv2d):
#                         m2.weight = nn.Parameter(m2.weight / m1.weight[:, None, None, None])
#                         m2.bias = nn.Parameter(m2.bias - m1.bias)
#                         del new_net._modules[list(new_net._modules.keys())[i]]
#                         break
#                     idx -= 1
#         return new_net

#     networks_without_bn = [scale_bn(net) for net in networks]
#     return fusion(networks_without_bn, gpu_id, accuracies, importance, eps, resnet)   

# fusion fuses two models with cost matrix based on weights, NOT ACTIVATIONS
def fusion_old(networks, args, gpu_id = -1, importance = [], eps=1e-7, resnet=False):
    assert len(networks) == 2 # Temporary assert. Later we can do more models

    if gpu_id:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(gpu_id))
    
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))

    skip_T_var = None
    skip_T_var_idx = -1
    residual_T_var = None
    residual_T_var_idx = -1
    

    avg_aligned_layers = []
    T_var = None
    bias = False
    bias0_weight, bias1_weight = (None, None)


    smallest_model_idx = None
    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
        if fc_layer0_weight.shape[0] < fc_layer1_weight.shape[0]:
            smallest_model_idx = 0
            break
        if fc_layer0_weight.shape[0] > fc_layer1_weight.shape[0]:
            smallest_model_idx = 1
            break
    
    if smallest_model_idx != None:
        change_model = networks[1]
        networks[1] = networks[smallest_model_idx]
        networks[smallest_model_idx] = change_model




    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
        
        if bias:
            # If in the last layer we detected bias, this layer will be the bias layer we handled before, so we can just skip it
            bias=False
            continue

        # Check if this current layer has a bias
        if (idx != num_layers-1):
            next_layer0, next_layer1 = next(itertools.islice(networks[0].named_parameters(), idx+1, None)), next(itertools.islice(networks[1].named_parameters(), idx+1, None))
            bias = True if "bias" in next_layer0[0] else False
            bias0_weight, bias1_weight = (next_layer0[1], next_layer1[1]) if bias else (None, None)
        else:
            bias = False
        
        print("idx {} and layer {}".format(idx, layer0_name))
        print("Bias: {}".format(bias))
        #assert fc_layer0_weight.shape == fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        layer_shape = fc_layer1_weight.shape

        if len(layer_shape) > 2:
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            # fc_layer0_weight_data has shape: (*out_channels, #in_channels, height*width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            if is_conv:
                # input to ground_metric has shape: (#out_channels, #in_channels*height*width)
                fc_layer0_flattened = fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1)
                fc_layer1_flattened =fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                M = get_ground_metric(fc_layer0_flattened,
                                fc_layer1_flattened, bias0_weight, bias1_weight)
            else:
                M = get_ground_metric(fc_layer0_weight_data, fc_layer1_weight_data, bias0_weight, bias1_weight)

            aligned_wt = fc_layer0_weight_data
        else:
            if is_conv:
                if resnet:
                    assert len(layer_shape) == 4
                    # save skip_level transport map if there is block ahead
                    if layer_shape[1] != layer_shape[0]:
                        if not (layer_shape[2] == 1 and layer_shape[3] == 1):
                            print(f'saved skip T_var at layer {idx} with shape {layer_shape}')
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            print(
                                f'utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}')
                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var
                        print("shape of previous transport map now is", T_var.shape)
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2
                            print("averaging multiple T_var's")
                        else:
                            print("doing nothing for skips")
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

                M = get_ground_metric(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1), 
                    bias0_weight, bias1_weight
                )
            else:
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)

                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)

                M = get_ground_metric(aligned_wt, fc_layer1_weight, bias0_weight, bias1_weight)


        mu = get_histogram(mu_cardinality)
        nu = get_histogram(nu_cardinality)

        cpuM = M.data.cpu().numpy() # POT does not accept array on GPU

        T = ot.emd(mu, nu, cpuM)

        if gpu_id!=-1:
            T_var = torch.from_numpy(T).cuda(gpu_id).float()
        else:
            T_var = torch.from_numpy(T).float()

        # ----- Assumption: correction = TRUE, proper_marginals = FALSE ---------
        if gpu_id != -1:
            marginals = torch.ones(T_var.shape[1]).cuda(gpu_id) / T_var.shape[0]
        else:
            marginals = torch.ones(T_var.shape[1]) / T_var.shape[0]
        marginals = torch.diag(1.0/(marginals + eps))  # take inverse
        T_var = torch.matmul(T_var, marginals)
        print(T_var.shape)
        print(T_var[:,0])
        T_var = T_var / T_var.sum(dim=0)
        print(T_var[:,0])
        # -----------------------------------------------------------------------
        # ---- Assumption: Past correction = True (Anything else doesn't really make sense?)
        t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        #sum = torch.sum(T_var, 0)
        #sum = torch.diag(1.0/sum)
        #t_fc0_model = torch.matmul(t_fc0_model.t(), sum).t()

        # ---------------------------------------------------------------------------------
        # t_fc0_model corresponds to aligned weights according to weights of model 1

        # Averaging of aligned weights (Could potentially also favor one model over the other)
        geometric_fc = (t_fc0_model+ fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))/2

        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)

        avg_aligned_layers.append(geometric_fc)
        if bias:
            t_bias_aligned = torch.matmul(T_var.t(), bias0_weight.view(bias0_weight.shape[0], -1)).flatten()
            #sum = torch.sum(T_var, 0)
            #t_bias_aligned/sum
            geometric_bias = (t_bias_aligned + bias1_weight)/2
            avg_aligned_layers.append(geometric_bias)
        
        mu = None
        nu = None
        fc_layer0_weight_data = None
        fc_layer1_weight_data = None
        fc_layer0_flattened = None
        fc_layer1_flattened = None
        M = None
        cpuM = None
        marginals = None
        geometric_fc = None
        bias0_weight = None
        bias1_weight = None

    
    return create_network_from_params(args=args, param_list=avg_aligned_layers, reference_model = networks[1])

def fusion_old2(networks, args, eps=1e-7):
    assert len(networks) == 2 # Temporary assert. Later we can do more models

    if args.gpu_id==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    

    avg_aligned_layers = []
    T_var = None
    bias = False
    bias0_weight, bias1_weight = (None, None)

    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
        
        if bias:
            # If in the last layer we detected bias, this layer will be the bias layer we handled before, so we can just skip it
            bias=False
            continue

        # Check if this current layer has a bias
        if (idx != num_layers-1):
            next_layer0, next_layer1 = next(itertools.islice(networks[0].named_parameters(), idx+1, None)), next(itertools.islice(networks[1].named_parameters(), idx+1, None))
            bias = True if "bias" in next_layer0[0] else False
            bias0_weight, bias1_weight = (next_layer0[1], next_layer1[1]) if bias else (None, None)
        else:
            bias = False
        
        print("idx {} and layer {}".format(idx, layer0_name))
        print("Bias: {}".format(bias))
        assert fc_layer0_weight.shape == fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        layer_shape = fc_layer0_weight.shape

        if len(layer_shape) > 2:
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            # fc_layer0_weight_data has shape: (*out_channels, #in_channels, height*width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            if is_conv:
                # input to ground_metric has shape: (#out_channels, #in_channels*height*width)
                fc_layer0_flattened = fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1)
                fc_layer1_flattened =fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                M = get_ground_metric(fc_layer0_flattened,
                                fc_layer1_flattened, bias0_weight, bias1_weight)
            else:
                M = get_ground_metric(fc_layer0_weight_data, fc_layer1_weight_data, bias0_weight, bias1_weight)

            aligned_wt = fc_layer0_weight_data
        else:
            if is_conv:
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
                
                M = get_ground_metric(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1), 
                    bias0_weight, bias1_weight
                )
            else:
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)

                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                M = get_ground_metric(aligned_wt, fc_layer1_weight, bias0_weight, bias1_weight)


        mu = get_histogram(mu_cardinality)
        nu = get_histogram(nu_cardinality)

        cpuM = M.data.cpu().numpy() # POT does not accept array on GPU

        T = ot.emd(mu, nu, cpuM)

        if args.gpu_id!=-1:
            T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
        else:
            T_var = torch.from_numpy(T).float()
        

        # ----- Assumption: correction = TRUE, proper_marginals = FALSE ---------
        if args.gpu_id != -1:
            marginals = torch.ones(T_var.shape[0]).cuda(args.gpu_id) / T_var.shape[0]
        else:
            marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
        marginals = torch.diag(1.0/(marginals + eps))  # take inverse
        T_var = torch.matmul(T_var, marginals)

        # -----------------------------------------------------------------------

        # ---- Assumption: Past correction = True (Anything else doesn't really make sense?)
        t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))

        # ---------------------------------------------------------------------------------
        # t_fc0_model corresponds to aligned weights according to weights of model 1

        # Averaging of aligned weights (Could potentially also favor one model over the other)
        geometric_fc = (t_fc0_model+ fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))/2

        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)

        avg_aligned_layers.append(geometric_fc)
        if bias:
            t_bias_aligned = torch.matmul(T_var.t(), bias0_weight.view(bias0_weight.shape[0], -1)).flatten()
            geometric_bias = (t_bias_aligned + bias1_weight)/2
            avg_aligned_layers.append(geometric_bias)

    
    return create_network_from_params(args=args, param_list=avg_aligned_layers, reference_model = networks[1])



def fusion_sidak_multimodel(networks, gpu_id = -1, accuracies=None, importance=None, eps=1e-7, resnet=False):
    """
    Implementing sidaks multi model fusion. Basically NOT fusing network by network into a "running" avg_layer,
    but by alligning all layers at the same time and then doing the average on the result.

    fusion fuses arbitrary many models into the model that is the smallest
    :param networks: A list of networks to be fused
    :param accuracies: A list of accuracies of the networks. The code will order the networks with respect to the accuracies for optimal accuracy
    :param importance: A list of floats. If importance = [0.9, 0.1], then linear combination of weights will be: network[0]*0.9 + network[1]*0.1
    :return: the fused model
    """ 

    num_models = len(networks)
    
    # preparing some lists that we will work with:
    T_var_list = [None]*num_models
    skip_T_var_list = [None]*num_models
    skip_T_var_idx_list = [-1]*num_models
    residual_T_var_list = [None]*num_models
    residual_T_var_idx_list = [-1]*num_models

    # to sort the networks by for better fusion order
    if accuracies == None:
        accuracies = [0]*num_models
    # setting all networks to the same importance in case no importances were given
    importance = [1]*num_models if importance == None else importance

    # sort the networks by: 1. size of layers (smallest first), 2. importance, 3. accuracy. With self written comparatorF
    networks_zip = sorted(list(zip(networks, accuracies, importance)), key=functools.cmp_to_key(comparator))
    networks = [network_zip[0] for network_zip in networks_zip]
    importance = [network_zip[2] for network_zip in networks_zip]
    importance = torch.tensor(importance)
    # normalize importance
    sum = torch.sum(importance)
    importance = importance *(importance.shape[0]/sum)

    # takes a list of networks and pairs the layers that are going to be fused (Linear, Conv2D, BN)
    fusion_layers = preprocess_parameters(networks)
    # each entry in fusion_layers contains the corresponding layer of all the "networks": [[L1_net1, L1_net2], [L3_net1, L3_net2], ...] 
    num_layers = len(fusion_layers) # number of layers that are going to be fused


    
    # iterate through all the layers that are going to be fused
    for idx in range(num_layers):
        # CHANGE: do the matchings of all the layers to the same layer (NO running average layer)
        
        # In fusion_layers[idx] all the idx-layers from the different networks are given.
        # initialize the result of the averaging process with the first layer in fusion_layers[idx] (was sorted in networks_zip)
        avg_layer = fusion_layers[idx][0]

        # how many nodes are in the layer is defined by the dimensionality of the weight matrix
        mu_cardinality = avg_layer.weight.shape[0]
        # gets a normalized histogram that has as many entries as nodes are in the fusion_layers[idx]-layer
        mu = get_histogram(mu_cardinality, None)

        # checks if we are working on a convolutional layer (can be seen from avg_layer, since it contains an instance of nn.Conv2d/nn.Linear)
        is_conv = avg_layer.is_conv

        # OLD: update the avg_layer with all the idx-layers that are stored in fusion_layers (besides first, since its initialized with that)
        # NEW: allign all with the same initial layer! (init to fusion_layers[idx][0])
        # CHANGED: include these two lists to keep track of the collected permutations
        aligned_fusions_to_average_over=[]
        aligned_fusions_to_average_over_importance=[]
        aligned_fusions_to_average_over_own_importance=[]
        for idx_m, fusion_layer in enumerate(fusion_layers[idx][1:]):
            # idx-layer of the idx_m-th model 

            # index of the idx-layer of the idx_m-th model (since skipping [0] - due to init - we need to adapt)
            idx_model = idx_m + 1
            T_var = T_var_list[idx_model]   # = None

            # QUESTION: only for resnets
            skip_T_var = skip_T_var_list[idx_model] # = None, only used for resnets?
            skip_T_var_idx = skip_T_var_idx_list[idx_model] # = -1, only used for resnets?
            residual_T_var = residual_T_var_list[idx_model] # = None, only used for resnets?
            residual_T_var_idx = residual_T_var_idx_list[idx_model] # = -1, only used for resnets?

            nu_cardinality = fusion_layer.weight.shape[0]   # initialize with [0, 0, 0, ...] -> for number of nodes contained in the layer
            layer_shape = fusion_layer.weight.shape

            aligned_wt = fusion_layer.weight
            
            
            # Skipping residual connection within the first iteration over the code
            if idx != 0:
                if resnet and is_conv:
                    assert len(layer_shape) == 4
                    # save skip_level transport map if there is block ahead
                    if layer_shape[1] != layer_shape[0]:
                        if not (layer_shape[2] == 1 and layer_shape[3] == 1):
                            print(f'saved skip T_var at layer {idx} with shape {layer_shape}')
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            print(
                                f'utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}')
                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var
                        print("shape of previous transport map now is", T_var.shape)
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2
                            print("averaging multiple T_var's")
                        else:
                            print("doing nothing for skips")
                
                fusion_layer.align_weights(T_var)
            

            # computes a matrix of similarities between nodes - based on the euclidean distance of the normalized weight vectors
            M = get_ground_metric(fusion_layer.create_comparison_vec(), avg_layer.create_comparison_vec(), None, None)

            # creates a normalized histogram that contains a value for each node in the layer. All are set to the same value.
            nu = get_histogram(nu_cardinality, None)

            # gives the similarities between the nodes of the already averaged ad the currently considered fusion_layer(M: normalized and then euclidean distance)
            cpuM = M.data.cpu().numpy() # POT does not accept array on GPU

            # computes the OT between "this" (fusion_layer) layers' nodes and the ones from the running averaged layer (avg_layer)
            # QUESTION2: what is the resulting matrix exactly???
            T = ot.emd(nu, mu, cpuM)        

            # QUESTION3: can we just exchange this for a GMM aproach for doing the matching

            # QUESTION4: is this the part that organizes how the nodes are permutated to allign with avg_layer nodeas (according to OT)????
            if gpu_id != -1:
                T_var = torch.from_numpy(T).cuda(gpu_id).float()
            else:
                T_var = torch.from_numpy(T).float()
            if gpu_id != -1:
                marginals = torch.ones(T_var.shape[1]).cuda(gpu_id) / T_var.shape[0]
            else:
                marginals = torch.ones(T_var.shape[1]) / T_var.shape[0]
            marginals = torch.diag(1.0/(marginals + eps))  # take inverse
            T_var = torch.matmul(T_var, marginals)
            T_var = T_var / T_var.sum(dim=0)
            ###################

            # align the nodes in the fusion_layer with the nodeas of the avg_layer. According to findings of OT (stored in T_var(OT result))
            fusion_layer.permute_parameters(T_var)

            # CHANGED: update the running avg_layer weights with the weights of the fusion_layer: weighted average (importance of all layers in avg_layer vs importance of fusion_layer)
            aligned_fusions_to_average_over.append(fusion_layer)
            aligned_fusions_to_average_over_importance.append(torch.sum(importance[:idx_model]))  
            aligned_fusions_to_average_over_own_importance.append(importance[idx_model])  
            # Do NOT already here incorporate the newly matched nodes
            # avg_layer.update_weights(fusion_layer, torch.sum(importance[:idx_model]), importance[idx_model])

            # QUESTION: just stuff we are keeping track of so the next layers can look at it?
            T_var_list[idx_model] = T_var
            skip_T_var_list[idx_model] = skip_T_var
            skip_T_var_idx_list[idx_model] = skip_T_var_idx
            residual_T_var_list[idx_model] = residual_T_var
            residual_T_var_idx_list[idx_model] = residual_T_var_idx

        # CHANGED: update the avg_layer here: now that all the alginments already have been computed!
        for idx, to_include_layer in enumerate(aligned_fusions_to_average_over):
            avg_layer.update_weights(to_include_layer, aligned_fusions_to_average_over_importance[idx], aligned_fusions_to_average_over_own_importance[idx])

        ####THIS LAYER IS BEING FUSED##########################

    return create_network_from_parameters(gpu_id=gpu_id, param_list=[layers[0] for layers in fusion_layers], reference_model = networks[0])
