import enum
import math

import torch
import torch.nn as nn
from torch import linalg as LA


class BaseEnum(enum.Enum):
    @classmethod
    def has_name(cls, name):
        return name in cls._member_names_

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class FusionType(str, BaseEnum):
    WEIGHT = "weight"
    ACTIVATION = "activation"
    GRADIENT = "gradient"


class MetaPruneType(str):
    IF = "intra-fusion"
    DEFAULT = "default"


class PruneType(str):
    L1 = "l1"
    L2 = "l2"


class Fusion_Layer:
    def __init__(
        self,
        super_type,
        weight,
        name,
        fusion_type,
        intrafusion=False,
        bias=None,
        bn=False,
        bn_mean=None,
        bn_var=None,
        bn_gamma=None,
        bn_beta=None,
    ):
        self.super_type = super_type
        self.weight = torch.clone(weight).detach()
        self.bias = bias
        self.bn = bn
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self.bn_gamma = bn_gamma
        self.bn_beta = bn_beta
        self.is_conv = self.super_type == "Conv2d"
        assert FusionType.has_value(fusion_type)
        self.fusion_type = fusion_type
        self.proxy = (
            None  # This is used to compare weights when the fusion type is not weight-based
        )
        self.final_name = name
        self.next = None
        self.prev = None
        self.skip_next = None
        self.skip_align = None
        self.skip_ot = None
        self.intrafusion = intrafusion
        self.T = None
        self.skip_t = None
        self.skip_ot_2 = None

    def to(self, gpu_id):
        if gpu_id != -1:
            self.weight = self.weight.cuda(gpu_id)
            self.bias = self.bias.cuda(gpu_id) if self.bias != None else None
            if self.bn:
                self.bn_mean = self.bn_mean.cuda(gpu_id)
                self.bn_var = self.bn_var.cuda(gpu_id)
                if self.bn_gamma != None:
                    self.bn_gamma = self.bn_gamma.cuda(gpu_id)
                    self.bn_beta = self.bn_beta.cuda(gpu_id)
            self.activations = self.activations.cuda(gpu_id) if self.activations != None else None

        else:
            self.weight = self.weight.cpu()
            self.bias = self.bias.cpu() if self.bias != None else None
            if self.bn:
                self.bn_mean = self.bn_mean.cpu()
                self.bn_var = self.bn_var.cpu()
                if self.bn_gamma != None:
                    self.bn_gamma = self.bn_gamma.cpu()
                    self.bn_beta = self.bn_beta.cpu()
            self.activations = self.activations.cpu() if self.activations != None else None

    def set_next(self, fusion_layer):
        self.next = fusion_layer

    def set_prev(self, fusion_layer):
        self.prev = fusion_layer

    def get_incoming_features(self):
        w = self.weight.clone()

        w = w.view(self.weight.shape)
        if self.is_conv == False and self.prev.is_conv:
            return None
        else:
            perm = [i for i in range(1, len(self.weight.shape))]
            perm.insert(1, 0)
            return w.permute(perm)

    def set_proxy(self, proxy):
        self.proxy = proxy

    def update_bn(self, mean, var, name=None, gamma=None, beta=None):
        if self.fusion_type == FusionType.ACTIVATION:
            self.final_name = name
        self.bn = True
        self.bn_mean = mean
        self.bn_var = var
        self.bn_gamma = gamma
        self.bn_beta = beta

    def align_weights(self, T_var):
        if self.skip_align != None:
            self.skip_align.align_weights(T_var)
        if self.is_conv:
            weight_reshaped = self.weight.view(self.weight.shape[0], self.weight.shape[1], -1)
            T_var_conv = T_var.unsqueeze(0).repeat(weight_reshaped.shape[2], 1, 1)
            aligned_wt = torch.bmm(weight_reshaped.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
        else:
            if self.prev.is_conv:
                # Handles the switch from convolutional layers to fc layers
                layer_unflattened = self.weight.view(
                    self.weight.shape[0], T_var.shape[0], -1
                ).permute(2, 0, 1)
                aligned_wt = torch.bmm(
                    layer_unflattened, T_var.unsqueeze(0).repeat(layer_unflattened.shape[0], 1, 1)
                ).permute(1, 2, 0)
                aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
            else:
                aligned_wt = torch.matmul(self.weight, T_var)

        self.weight = aligned_wt

    def create_comparison_vec(self):
        if self.fusion_type != FusionType.WEIGHT:
            assert self.proxy != None
            result = [self.proxy.reshape(self.proxy.shape[0], -1).contiguous()]
            if self.skip_ot != None:
                result.extend(self.skip_ot.create_comparison_vec())
            if self.skip_ot_2 != None:
                result.extend(self.skip_ot_2.create_comparison_vec())
            return result

        concat = None
        if self.bias != None:
            concat = self.bias
        if self.bn:
            if concat != None:
                concat = concat.clone() - self.bn_mean
            else:
                concat = -self.bn_mean.clone()

        comparison = self.weight.clone().contiguous().view(self.weight.shape[0], -1)

        if self.bn:
            comparison /= self.bn_var[:, None]
            concat /= self.bn_var

            if self.bn_gamma != None:
                assert self.bn_beta != None
                comparison *= self.bn_gamma[:, None]
                concat *= self.bn_gamma
                concat += self.bn_beta

        if concat != None:
            comparison = torch.cat((comparison, concat.view(concat.shape[0], -1)), 1)

        result = [comparison]
        """if concat != None:
            result.append(concat.view(-1,1))"""

        if self.next != None and self.intrafusion:
            next_incoming_feat = self.next.get_incoming_features()
            if next_incoming_feat != None:
                result.append(
                    next_incoming_feat.reshape(next_incoming_feat.shape[0], -1).contiguous()
                )
        if self.skip_next != None and self.intrafusion:
            next_incoming_feat = self.skip_next.get_incoming_features()
            if next_incoming_feat != None:
                result.append(
                    next_incoming_feat.reshape(next_incoming_feat.shape[0], -1).contiguous()
                )

        if self.skip_ot != None:
            result.extend(self.skip_ot.create_comparison_vec())
        if self.skip_ot_2 != None:
            result.extend(self.skip_ot_2.create_comparison_vec())
        return result

    def get_norm(self, prune_type):
        comp_vecs = self.create_comparison_vec()
        norm_layer = []

        for idx, comp_vec in enumerate(comp_vecs):
            norm_layer.append(torch.norm(comp_vec, p=1 if prune_type == "l1" else 2, dim=1))

        norm_layer = torch.stack(norm_layer, dim=0)
        norm_layer = norm_layer.mean(dim=0)
        return norm_layer

    def permute_parameters(self, T_var):
        self.T = T_var
        if self.skip_ot != None:
            self.skip_ot.permute_parameters(T_var)
        if self.skip_ot_2 != None:
            self.skip_ot_2.permute_parameters(T_var)

        w_shape = self.weight.shape
        self.weight = torch.matmul(
            T_var.t(), self.weight.contiguous().view(self.weight.shape[0], -1)
        )
        if len(w_shape) == 3:
            self.weight = self.weight.view(
                -1, w_shape[1], int(math.sqrt(w_shape[2])), int(math.sqrt(w_shape[2]))
            )
        elif len(w_shape) == 4:
            self.weight = self.weight.view(-1, w_shape[1], w_shape[2], w_shape[3])

        if self.bias != None:
            self.bias = torch.matmul(T_var.t(), self.bias.view(self.bias.shape[0], -1)).flatten()

        if self.bn:
            self.bn_mean = torch.matmul(T_var.t(), self.bn_mean).flatten()
            self.bn_var = torch.matmul(T_var.t(), self.bn_var).flatten()
            if self.bn_gamma != None:
                self.bn_gamma = torch.matmul(T_var.t(), self.bn_gamma).flatten()
                self.bn_beta = torch.matmul(T_var.t(), self.bn_beta).flatten()

    def weighted_average(self, w0, w1, mult0, mult1, divisor, adjust):
        weight_new = None
        if len(w0.shape) > 1:
            weight_new = w0 * mult0[:, None] + w1 * mult1[:, None] * adjust[:, None]
            weight_new /= divisor[:, None]
        else:
            weight_new = w0 * mult0 + w1 * mult1 * adjust
            weight_new /= divisor
        return weight_new

    def get_intra_weights(self, T_var_col):
        indices = T_var_col.nonzero().view(-1)
        weight_ind = torch.index_select(self.weight, 0, indices)

        mean = torch.index_select(self.bn_mean, 0, indices)
        var = torch.index_select(self.bn_var, 0, indices)
        gamma = torch.index_select(self.bn_gamma, 0, indices)
        beta = torch.index_select(self.bn_beta, 0, indices)

        adjust = (
            (torch.prod(var, 0).repeat(var.shape[0], 1) / var.view(-1, 1)) * gamma.view(-1, 1)
        ) / (torch.sum(T_var_col) * torch.prod(var, 0))

        return adjust

    def update_weights_intra(self, T_var):
        self.T = T_var
        w_shape = self.weight.shape
        if not self.bn:
            self.permute_parameters(T_var)
            return
        if self.skip_ot != None:
            self.skip_ot.update_weights_intra(T_var)
        if self.skip_ot_2 != None:
            self.skip_ot_2.update_weights_intra(T_var)

        T_var_adjust = T_var.clone()
        for i in range(T_var.shape[1]):
            adjust = self.get_intra_weights(T_var_adjust[:, i])
            temp = torch.zeros(T_var_adjust[:, i].shape).cuda(0)

            temp = temp.index_copy_(0, T_var_adjust[:, i].nonzero().flatten(), adjust.flatten())

            T_var_adjust[:, i] *= temp

        self.weight = torch.matmul(
            T_var_adjust.t(), self.weight.contiguous().view(self.weight.shape[0], -1)
        )
        self.bn_mean = torch.matmul(T_var_adjust.t(), self.bn_mean).flatten()
        self.bn_var = torch.ones(self.bn_mean.shape)
        self.bn_gamma = torch.ones(self.bn_mean.shape)
        self.bn_beta = torch.matmul(T_var.t(), self.bn_beta).flatten()

        if len(w_shape) == 3:
            self.weight = self.weight.view(
                -1, w_shape[1], int(math.sqrt(w_shape[2])), int(math.sqrt(w_shape[2]))
            )
        elif len(w_shape) == 4:
            self.weight = self.weight.view(-1, w_shape[1], w_shape[2], w_shape[3])

    def update_weights(self, other_fusion_layer, imp0, imp1):
        if self.skip_ot != None:
            self.skip_ot.update_weights(other_fusion_layer.skip_ot, imp0, imp1)
        weight0 = self.weight.view(self.weight.shape[0], -1)
        weight1 = other_fusion_layer.weight.view(other_fusion_layer.weight.shape[0], -1)
        if self.bn:
            var1 = other_fusion_layer.bn_var
            mean1 = other_fusion_layer.bn_mean

            divisor = (imp0 + imp1) * var1
            mult0 = imp0 * var1
            mult1 = imp1 * self.bn_var
            adjust = torch.ones(var1.shape)
            if self.bn_gamma != None:
                adjust = torch.div(other_fusion_layer.bn_gamma, self.bn_gamma)

            self.weight = self.weighted_average(
                weight0, weight1, mult0, mult1, divisor, adjust
            ).view(self.weight.shape)
            if self.bias != None:
                self.bias = self.weighted_average(
                    self.bias, other_fusion_layer.bias, mult0, mult1, divisor, adjust
                )
            self.bn_mean = self.weighted_average(
                self.bn_mean, other_fusion_layer.bn_mean, mult0, mult1, divisor, adjust
            )
            # We don't have to update our own variance and our gamma

            if self.bn_beta != None:
                self.bn_beta = (imp0 * self.bn_beta + imp1 * other_fusion_layer.bn_beta) / (
                    imp0 + imp1
                )

        else:
            self.weight = ((weight1 * imp1 + weight0 * imp0) / (imp0 + imp1)).view(
                self.weight.shape
            )
            if self.bias != None:
                self.bias = (self.bias * imp0 + other_fusion_layer.bias * imp1) / (imp0 + imp1)


# def compute_activations(model, train_loader, num_samples, fusion_layers, mode='mean', gpu_id=-1):
def compute_gradients(model, train_loader, num_samples, fusion_layers, gpu_id=-1):
    torch.manual_seed(42)

    # hook that computes the mean activations across data samples
    def get_activation(activation, name):
        def hook(model, input, output):
            if name not in activation:
                activation[name] = []

            out = output[0].detach()

            """out_mean = torch.mean(out, dim=0)
            out_mean = out_mean.unsqueeze(0)"""
            out_mean = out
            activation[name].append(out_mean)

        return hook

    # Prepare all the models
    activations = {}
    forward_hooks = []

    # handle below for bias later on!
    param_names = [fusion_layer.final_name for fusion_layer in fusion_layers]  # ------------CHANGED
    for fusion_layer in fusion_layers:
        if fusion_layer.skip_ot != None:
            param_names.append(fusion_layer.skip_ot.final_name)

    # Initialize the activation dictionary for each model
    layer_hooks = []
    # Set forward hooks for all layers inside a model
    for name, layer in model.named_modules():
        if name == "":
            continue
        elif name not in param_names:
            continue
        layer_hooks.append(layer.register_full_backward_hook(get_activation(activations, name)))

    forward_hooks.append(layer_hooks)
    # Set the model in train mode
    # model.train()

    loss_func = nn.CrossEntropyLoss()

    # Run the same data samples ('num_samples' many) across all the models
    num_samples_processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if num_samples_processed >= num_samples:
            break
        if gpu_id != -1:
            data = data.cuda(gpu_id)
            target = target.cuda(gpu_id)

        model.zero_grad()
        predictions = model(data)
        loss = loss_func(predictions, target)

        # backpropagation, compute gradients
        loss.backward()

        num_samples_processed += data.shape[0]

    for idx, layer in enumerate(activations):
        activations[layer] = torch.stack(activations[layer])

        if len(activations[layer].shape) == 2:
            activations[layer] = activations[layer].t()
        else:
            perm = [i for i in range(2, len(activations[layer].shape))]
            perm.append(0)
            perm.append(1)

            acts = activations[layer].permute(*perm)
            activations[layer] = acts

    # Remove the hooks (as this was intefering with prediction ensembling)
    for idx in range(len(forward_hooks)):
        for hook in forward_hooks[idx]:
            hook.remove()

    return activations


# def compute_activations(model, train_loader, num_samples, fusion_layers, mode='mean', gpu_id=-1):
def compute_activations(model, train_loader, num_samples, fusion_layers, gpu_id=-1):
    torch.manual_seed(42)

    # hook that computes the mean activations across data samples
    def get_activation(activation, name):
        def hook(model, input, output):
            if name not in activation:
                activation[name] = []

            activation[name].append(output.detach())

        return hook

    # Prepare all the models
    activations = {}
    forward_hooks = []

    # handle below for bias later on!
    param_names = [fusion_layer.final_name for fusion_layer in fusion_layers]  # ------------CHANGED
    for fusion_layer in fusion_layers:
        if fusion_layer.skip_ot != None:
            param_names.append(fusion_layer.skip_ot.final_name)
    # Initialize the activation dictionary for each model
    layer_hooks = []
    # Set forward hooks for all layers inside a model
    for name, layer in model.named_modules():
        if name == "":
            continue
        elif name not in param_names:
            continue
        layer_hooks.append(layer.register_forward_hook(get_activation(activations, name)))

    forward_hooks.append(layer_hooks)
    # Set the model in train mode
    # model.train()

    # Run the same data samples ('num_samples' many) across all the models
    num_samples_processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if num_samples_processed >= num_samples:
            break
        if gpu_id != -1:
            data = data.cuda(gpu_id)

        model(data)

        num_samples_processed += data.shape[0]

    for idx, layer in enumerate(activations):
        activations[layer] = torch.stack(activations[layer])

        if len(activations[layer].shape) == 2:
            activations[layer] = activations[layer].t()
        else:
            perm = [i for i in range(2, len(activations[layer].shape))]
            perm.append(0)
            perm.append(1)

            acts = activations[layer].permute(*perm)
            activations[layer] = acts.reshape(acts.shape[0], -1).contiguous()

    # Remove the hooks (as this was intefering with prediction ensembling)
    for idx in range(len(forward_hooks)):
        for hook in forward_hooks[idx]:
            hook.remove()

    return activations


def preprocess_parameters(
    models,
    fusion_type,
    intrafusion=False,
    resnet=False,
    train_loader=None,
    model_name=None,
    gpu_id=-1,
    num_samples=200,
):
    all_fusion_layers = []

    for idx, model in enumerate(models):
        fusion_layers = []
        prev_name = ""
        for name, module in model.named_modules():
            fusion_layer = None
            if isinstance(module, nn.Linear):
                fusion_layer = Fusion_Layer(
                    "Linear",
                    module.weight,
                    intrafusion=intrafusion,
                    name=name,
                    bias=module.bias,
                    fusion_type=fusion_type,
                )
                if resnet and len(fusion_layers) > 1:
                    if (
                        model_name != "resnet50"
                        and model_name != "resnet101"
                        and model_name != "resnet152"
                    ):
                        if fusion_layers[-2].weight.shape[0] == fusion_layers[-2].weight.shape[1]:
                            fusion_layers[-3].skip_ot_2 = fusion_layers[-1]
                    else:
                        if fusion_layers[-1].skip_ot == None:
                            fusion_layers[-4].skip_ot_2 = fusion_layers[-1]

            elif isinstance(module, nn.Conv2d):
                fusion_layer = Fusion_Layer(
                    "Conv2d",
                    module.weight,
                    name=name,
                    intrafusion=intrafusion,
                    bias=module.bias,
                    fusion_type=fusion_type,
                )
                if resnet and len(fusion_layers) > 0:
                    if (
                        model_name != "resnet50"
                        and model_name != "resnet101"
                        and model_name != "resnet152"
                    ):
                        if module.weight.shape[2] == 1 and module.weight.shape[3] == 1:
                            fusion_layers[-1].skip_ot = fusion_layer
                            assert (
                                fusion_layers[-2].weight.shape[:2] == fusion_layer.weight.shape[:2]
                            )
                            fusion_layers[-2].skip_align = fusion_layer
                            fusion_layers[-3].skip_next = fusion_layer

                            continue
                        if (
                            len(fusion_layers) > 1
                            and "conv1" in name
                            and fusion_layers[-2].weight.shape[0]
                            == fusion_layers[-2].weight.shape[1]
                        ):
                            fusion_layers[-3].skip_ot_2 = fusion_layers[-1]
                    else:
                        if "shortcut" in name or "downsample" in name:
                            fusion_layers[-1].skip_ot = fusion_layer
                            assert fusion_layers[-3].weight.shape[1] == fusion_layer.weight.shape[1]
                            fusion_layers[-3].skip_align = fusion_layer
                            fusion_layers[-4].skip_next = fusion_layer

                            continue
                        elif (
                            len(fusion_layers) > 1
                            and "conv1" in name
                            and fusion_layers[-1].skip_ot == None
                        ):
                            fusion_layers[-4].skip_ot_2 = fusion_layers[-1]

            elif isinstance(module, nn.BatchNorm2d):
                ########## IMPORTANT: CHANGE TO NAME AGAIN!!!!!
                if fusion_layers[-1].bn:
                    if fusion_layers[-1].skip_ot == None:
                        raise Exception(
                            "Found batchnorm layer that belongs to layer we can't identify."
                        )
                    else:
                        fusion_layers[-1].skip_ot.update_bn(
                            module.running_mean,
                            module.running_var,
                            name=name,
                            gamma=module.weight,
                            beta=module.bias,
                        )
                else:
                    fusion_layers[-1].update_bn(
                        module.running_mean,
                        module.running_var,
                        name=name,
                        gamma=module.weight,
                        beta=module.bias,
                    )
                continue
            else:
                continue

            if len(fusion_layers) > 0:
                fusion_layers[-1].set_next(fusion_layer)
                fusion_layer.set_prev(fusion_layers[-1])
            fusion_layers.append(fusion_layer)
            prev_name = name
        # exit()
        if fusion_type != FusionType.WEIGHT:
            proxy = None
            if fusion_type == FusionType.ACTIVATION:
                proxy = compute_activations(
                    model,
                    train_loader=train_loader,
                    num_samples=num_samples,
                    fusion_layers=fusion_layers,
                    gpu_id=gpu_id,
                )
            else:
                proxy = compute_gradients(
                    model,
                    train_loader=train_loader,
                    num_samples=num_samples,
                    fusion_layers=fusion_layers,
                    gpu_id=gpu_id,
                )
            for fusion_layer in fusion_layers:
                fusion_layer.set_proxy(proxy[fusion_layer.final_name])
                if fusion_layer.skip_ot:
                    fusion_layer.skip_ot.set_proxy(proxy[fusion_layer.skip_ot.final_name])

        all_fusion_layers.append(fusion_layers)

    return list(zip(*all_fusion_layers))
