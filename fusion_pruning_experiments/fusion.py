import functools
import itertools
import torch
import numpy as np
import ot
import copy
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
def get_ground_metric(coordinates1, coordinates2, bias1, bias2, BN=None):
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



def fusion(networks, gpu_id = -1, accuracies=None, importance=None, eps=1e-7, resnet=False):
    """
    fusion fuses arbitrary many models into the model that is the smallest

    :param networks: A list of networks to be fused
    :param accuracies: A list of accuracies of the networks. The code will order the networks with respect to the accuracies for optimal accuracy
    :param importance: A list of floats. If importance = [0.9, 0.1], then linear combination of weights will be: network[0]*0.9 + network[1]*0.1
    :return: the fused model
    """ 
    print("fuse_Start")
    gpu_id = -1
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
        print(f"Layer num{idx+1}")
        if bias:
            # If in the last layer we detected bias, this layer will be the bias layer we handled before, so we can just skip it
            bias=False
            update_iterators(layer_iters)
            continue
        
        if (idx != num_layers-1):
            bias = True if "bias" in next(itertools.islice(networks[0].named_parameters(), idx+1, None))[0] else False
            print(bias)
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
            
            nu = get_histogram(nu_cardinality)

            
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

def fusion_bn_alt(networks, gpu_id = -1, accuracies=None, importance=None, eps=1e-7, resnet=False):
    def scale_bn(net):
        """
        Create a deep copy of a PyTorch neural network and remove all BatchNorm layers from the copy.
        
        Args:
            net (nn.Module): The neural network to copy and modify.
        
        Returns:
            nn.Module: A deep copy of the input network with all BatchNorm layers removed.
        """
        # Create a deep copy of the input network.
        new_net = copy.deepcopy(net)
        


        # Remove BatchNorm layers from the copy.
        for i, key in enumerate(new_net._modules):
            m1 = new_net._modules[key]
            if isinstance(m1, nn.Sequential):
                new_net._modules[list(new_net._modules.keys())[i]] = scale_bn(m1)
            elif isinstance(m1, nn.BatchNorm2d) or isinstance(m1, nn.BatchNorm1d):
                idx = i
                while idx >= 0:
                    m2 = list(new_net.modules())[idx]
                    if isinstance(m2, nn.Linear) or isinstance(m2, nn.Conv2d):
                        m2.weight = nn.Parameter(m2.weight / m1.weight[:, None, None, None])
                        m2.bias = nn.Parameter(m2.bias - m1.bias)
                        del new_net._modules[list(new_net._modules.keys())[i]]
                        break
                    idx -= 1
        return new_net
    
    networks_without_bn = [scale_bn(net) for net in networks]
    return fusion(networks_without_bn, gpu_id = -1, accuracies=None, importance=None, eps=1e-7, resnet=False)


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