import itertools
import torch
import numpy as np
import ot
from models import get_model
#from scipy.optimize import linear_sum_assignment # Could accomplish the same as OT with Hungarian Algorithm

# get_histogram creates uniform historgram, i.e. [1/cardinality, 1/cardinality, ...]
def get_histogram(cardinality):
    return np.ones(cardinality)/cardinality # uniform probability distribution

def normalize_vector(coordinates, eps=1e-9):
    norms = torch.norm(coordinates, dim=-1, keepdim=True)
    return coordinates / (norms + eps)

# compute_euclidian_distance_matrix computes a matrix where c[i, j] corresponds to the euclidean distance of x[i] and y[j]
def compute_euclidian_distance_matrix(x, y, p=2, squared=True): # For some reason TA prefers squared to be True
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
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
    else:
        assert bias2 == None

    coordinates1 = normalize_vector(coordinates1)
    coordinates2 = normalize_vector(coordinates2)
    return compute_euclidian_distance_matrix(coordinates1, coordinates2)

# create_network_from_params creates a network given the list of weights
def create_network_from_params(args, param_list):
    model = get_model(args)

    assert len(list(model.parameters())) == len(param_list) # Assumption: We are fusing into a model of same architecture

    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)
    
    layer_idx = 0
    model_state_dict = model.state_dict()

    for key in model_state_dict.keys():
        model_state_dict[key] = param_list[layer_idx]
        layer_idx += 1
    
    model.load_state_dict(model_state_dict)

    return model

def update_iterators(iterators):
    for iter in iterators:
        next(iter)


# fusion fuses two models with cost matrix based on weights, NOT ACTIVATIONS
def fusion(networks, args, eps=1e-7):
    #assert len(networks) == 2 # Temporary assert. Later we can do more models
    
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    num_models = args.num_models

    avg_aligned_layers = []
    T_var_list = [None]*num_models
    bias = False
    bias_weight = None
    avg_bias_weight = None
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
        mu = get_histogram(mu_cardinality)
        is_conv = len(layer_shape) > 2

        print("idx {} and layer {}".format(idx, avg_layer_name))
        print("Bias: {}".format(bias))
        
        
        for idx_model in range(1, num_models): # We skip the first model because we align our models according to the first model
            T_var = T_var_list[idx_model]
            _, layer = next(layer_iters[idx_model])

            bias_weight = next(itertools.islice(networks[idx_model].named_parameters(), idx+1, None))[1] if bias else None


            assert avg_layer.shape == layer.shape

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
            t_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))

            # ---------------------------------------------------------------------------------
            # t_model corresponds to aligned weights according to weights of model 1

            # Averaging of aligned weights (Could potentially also favor one model over the other)
            avg_layer = (t_model + avg_layer_data.view(avg_layer_data.shape[0], -1)*idx_model)/(idx_model+1)

            if is_conv and layer_shape != avg_layer.shape:
                avg_layer = avg_layer.view(layer_shape)

            if bias:
                t_bias_aligned = torch.matmul(T_var.t(), bias_weight.view(bias_weight.shape[0], -1)).flatten()
                avg_bias_weight = (t_bias_aligned + avg_bias_weight*idx_model)/(idx_model+1)

            T_var_list[idx_model] = T_var
        
        avg_aligned_layers.append(avg_layer)
        if bias:
            avg_aligned_layers.append(avg_bias_weight)
        
        cpuM = None 
        M = None
        T = None
        avg_layer_data = None 
        layer_data = None

    return create_network_from_params(args, avg_aligned_layers)

