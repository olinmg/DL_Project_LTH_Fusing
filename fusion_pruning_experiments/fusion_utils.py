import torch.nn as nn
import torch

class Fusion_Layer():
    def __init__(self, super_type, weight, bias=None, bn=False, bn_mean=None, bn_var=None, bn_gamma=None, bn_beta=None):
        self.super_type = super_type
        self.weight = weight
        self.bias = bias
        self.bn = bn
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self.bn_gamma = bn_gamma
        self.bn_beta = bn_beta
        self.is_conv = self.super_type == "Conv2d"
    
    def update_bn(self,mean, var, gamma=None, beta=None):   
        self.bn = True
        self.bn_mean = mean
        self.bn_var = var
        self.bn_gamma = gamma
        self.bn_beta = beta
    
    def align_weights(self,T_var):
        if self.is_conv:
            weight_reshaped = self.weight.view(self.weight.shape[0], self.weight.shape[1], -1)
            T_var_conv = T_var.unsqueeze(0).repeat(weight_reshaped.shape[2], 1, 1)
            aligned_wt = torch.bmm(weight_reshaped.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
        else:
            if self.weight.data.shape[1] != T_var.shape[0]:
                # Handles the switch from convolutional layers to fc layers
                layer_unflattened = self.weight.view(self.weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                aligned_wt = torch.bmm(
                    layer_unflattened,
                    T_var.unsqueeze(0).repeat(layer_unflattened.shape[0], 1, 1)
                ).permute(1, 2, 0)
                aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
            else:
                aligned_wt = torch.matmul(self.weight, T_var)
        
        self.weight = aligned_wt
    
    def create_comparison_vec(self):
        concat = None
        if self.bias != None:
            concat = self.bias
        if self.bn:
            if concat != None:
                concat = concat.clone() - self.bn_mean
            else:
                concat = -self.bn_mean
        
        comparison = self.weight.contiguous().view(self.weight.shape[0], -1)
        if concat != None:
            comparison = torch.cat((comparison, concat.view(concat.shape[0], -1)), 1)

        if self.bn:
            comparison /= self.bn_var[:, None]

            if self.bn_gamma != None:
                assert self.bn_beta != None
                comparison *= self.bn_gamma[:, None]
                comparison += self.bn_beta[:, None]
        
        return comparison
    
    def permute_parameters(self, T_var):
        self.weight = torch.matmul(T_var.t(), self.weight.contiguous().view(self.weight.shape[0], -1)).view(self.weight.shape)

        if self.bias != None:
            self.bias = torch.matmul(T_var.t(), self.bias.view(self.bias.shape[0], -1)).flatten()
        
        if self.bn:
            self.bn_mean = torch.matmul(T_var.t(), self.bn_mean.view(self.bn_mean.shape[0], -1)).flatten()
            self.bn_var = torch.matmul(T_var.t(), self.bn_var.view(self.bn_var.shape[0], -1)).flatten()
            if self.bn_gamma != None:
                self.bn_gamma = torch.matmul(T_var.t(), self.bn_gamma.view(self.bn_gamma.shape[0], -1)).flatten()
                self.bn_beta = torch.matmul(T_var.t(), self.bn_beta.view(self.bn_beta.shape[0], -1)).flatten()

    def weighted_average(self, w0, w1, mult0, mult1, divisor, adjust):
        weight_new = None
        if len(w0.shape) > 1:
            weight_new = w0*mult0[:, None] + w1*mult1[:, None]*adjust[:, None]
            weight_new /= divisor[:, None]
        else:
            weight_new = w0*mult0 + w1*mult1*adjust
            weight_new /= divisor
        return weight_new


    def update_weights(self, other_fusion_layer, imp0, imp1):
        weight0 = self.weight.view(self.weight.shape[0], -1)
        weight1 = other_fusion_layer.weight.view(other_fusion_layer.weight.shape[0], -1)
        if self.bn:
            var1 = other_fusion_layer.bn_var
            mean1 = other_fusion_layer.bn_mean


            divisor = (imp0+imp1)*var1
            mult0 = imp0*var1
            mult1 = imp1*self.bn_var
            adjust = torch.ones(var1.shape)
            if self.bn_gamma != None:
                adjust = torch.div(other_fusion_layer.bn_gamma, self.bn_gamma)

            self.weight = self.weighted_average(weight0, weight1, mult0, mult1, divisor, adjust).view(self.weight.shape)
            if self.bias != None:
                self.bias = self.weighted_average(self.bias, other_fusion_layer.bias, mult0, mult1, divisor, adjust)
            self.bn_mean = self.weighted_average(self.bn_mean, other_fusion_layer.bn_mean, mult0, mult1, divisor, adjust)
            # We don't have to update our own variance and our gamma

            if self.bn_beta != None:
                self.bn_beta = (imp0*self.bn_beta+imp1*other_fusion_layer.bn_beta)/(imp0+imp1)
            
        else:
            self.weight = ((weight1*imp1 + weight0*imp0)/(imp0+imp1)).view(self.weight.shape)
            if self.bias != None:
                self.bias = (self.bias*imp0 + other_fusion_layer.bias*imp1)/(imp0+imp1)
        
 


def preprocess_parameters(models):
    all_fusion_layers = []
    for model in models:
        fusion_layers = []
        for _, module in model.named_modules():
            print(type(module))

            fusion_layer = None

            if isinstance(module, nn.Linear):
                fusion_layer = Fusion_Layer("Linear", module.weight, bias=module.bias)
            elif isinstance(module, nn.Conv2d):
                fusion_layer = Fusion_Layer("Conv2d", module.weight, bias=module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                fusion_layers[-1].update_bn(module.running_mean, module.running_var, gamma=module.weight, beta = module.bias)
                continue
            else:
                continue

            fusion_layers.append(fusion_layer)
        all_fusion_layers.append(fusion_layers)
    
    return list(zip(*all_fusion_layers))