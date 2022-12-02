import itertools
import torch
import numpy as np
import ot
from base_convNN import CNN, get_model
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
    if bias1 != None: # and bias2 == None
        assert bias2 == None
        coordinates1 = torch.cat((coordinates1, bias1.view(bias1.shape[0], -1)), 1)
        coordinates2 = torch.cat((coordinates2, bias2.view(bias2.shape[0], -1)), 1)
    coordinates1 = normalize_vector(coordinates1)
    coordinates2 = normalize_vector(coordinates2)
    return compute_euclidian_distance_matrix(coordinates1, coordinates2)

# create_network_from_params creates a network given the list of weights
def create_network_from_params(args, param_list):
    model = get_model(args.model_name)

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

# fusion fuses two models with cost matrix based on weights, NOT ACTIVATIONS
def fusion(networks, args, eps=1e-7):
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

    
    return create_network_from_params(args, avg_aligned_layers)


