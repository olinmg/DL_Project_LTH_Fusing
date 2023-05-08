from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as model_archs

from mobilenetv1 import MobileNetV1
from resnet import BasicBlock, Bottleneck, ResNet
from vgg import VGG, make_layers


class MLP(nn.Module):
    def __init__(self, sparsity=1.0):
        super(MLP, self).__init__()
        bias = True
        self.sparsity = sparsity
        self.lin1 = nn.Sequential(
            nn.Linear(28 * 28, int(128 * self.sparsity), bias=bias),
            nn.ReLU(),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(int(128 * self.sparsity), int(512 * self.sparsity), bias=bias),
            nn.ReLU(),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(int(512 * self.sparsity), 10, bias=bias)

    def forward(self, x):
        x = self.lin1(x.view(-1, 28 * 28))
        x = self.lin2(x)
        output = self.out(x)
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        bias = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=bias,  # Needs to change later!
            ),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2, bias=bias),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class CNN_batchNorm(nn.Module):
    def __init__(self):
        super(CNN_batchNorm, self).__init__()
        bias = True
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=bias,  # Needs to change later!
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def ResNet18(num_classes=10, use_batchnorm=False, linear_bias=True):
    print("linear_bias is: ", linear_bias)
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        use_batchnorm=use_batchnorm,
        linear_bias=linear_bias,
    )


def ResNet34(num_classes=10, use_batchnorm=False, linear_bias=True):
    return ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        use_batchnorm=use_batchnorm,
        linear_bias=linear_bias,
    )


def ResNet50(num_classes=10, use_batchnorm=False, linear_bias=True):
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        use_batchnorm=use_batchnorm,
        linear_bias=linear_bias,
    )


def ResNet101(num_classes=10, use_batchnorm=False, linear_bias=True):
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        use_batchnorm=use_batchnorm,
        linear_bias=linear_bias,
    )


def ResNet152(num_classes=10, use_batchnorm=False, linear_bias=True):
    return ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        num_classes=num_classes,
        use_batchnorm=use_batchnorm,
        linear_bias=linear_bias,
    )


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def vgg11(bias=False, sparsity=1.0, output_dim=10):
    """VGG 11-layer model (configuration "A")"""
    params = cfg["A"]
    params = [
        (round(i * sparsity) if round(i * sparsity) > 1 else 1) if isinstance(i, int) else i
        for i in params
    ]
    print(params)
    return VGG(
        make_layers(params, bias=bias),
        bias=bias,
        sparsity=sparsity,
        output_dim=output_dim,
    )


def vgg11_bn(bias=False):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg["A"], bias=bias, batch_norm=True))


def vgg13(bias=True):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg["B"], bias=bias), bias=bias)


def vgg16(bias=True):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg["D"], bias=bias), bias=bias)


def vgg19(bias=False):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg["E"], bias=bias), bias=bias)


def get_model(model_name, sparsity=1.0, output_dim=10):
    print("Went in with sparsity: ", sparsity)
    print("Model entered: ", model_name)
    if model_name == "cnn":
        return CNN()
    elif model_name == "cnn_bn":
        return CNN_batchNorm()
    elif model_name == "mlp":
        return MLP(sparsity)
    elif model_name == "mobilenetv1":
        return MobileNetV1(ch_in=3, n_classes=output_dim)
    elif model_name == "vgg11":
        return vgg11(bias=False, sparsity=sparsity, output_dim=output_dim)
    elif model_name == "vgg11_bn":
        return vgg11_bn(bias=False)
    elif model_name == "vgg13":
        return vgg13()
    elif model_name == "vgg16":
        return vgg16()
    elif model_name == "vgg19":
        return vgg19()
    elif model_name == "resnet18":
        return ResNet18(linear_bias=False)
    elif model_name == "resnet34":
        return ResNet34()
    elif model_name == "resnet50":
        return ResNet50()
    elif model_name == "resnet101":
        return ResNet101()
    elif model_name == "resnet152":
        return ResNet152()
    else:
        print("Invalid model name. Using CNN instead.")
        return CNN()


def get_pretrained_model_by_name(model_file_path, gpu_id):
    if "resnet50" in model_file_path:
        model = model_archs.__dict__["resnet50"]()
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(f"{model_file_path}.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])
        model = model.module.to("cpu")
    else:
        try:
            model = torch.load(f"{model_file_path}.pth")
        except:
            model = torch.load(f"{model_file_path}.pth", map_location=torch.device("cpu"))
    model = model.cuda(gpu_id) if gpu_id != -1 else model.to("cpu")
    return model


def get_pretrained_models(model_name, basis_name, gpu_id, num_models, output_dim=10):
    models = []

    for idx in range(num_models):
        if model_name == "resnet50":
            model = model_archs.__dict__["resnet50"]()
            model = torch.nn.DataParallel(model)
            checkpoint = torch.load(f"models/{basis_name}_{idx}.pth.tar")
            model.load_state_dict(checkpoint["state_dict"])
            model = model.module.to("cpu")
        else:
            try:
                model = torch.load(f"models/{basis_name}_{idx}.pth")
            except:
                model = torch.load(
                    f"models/{basis_name}_{idx}.pth", map_location=torch.device("cpu")
                )
        model = model.cuda(gpu_id) if gpu_id != -1 else model.to("cpu")
        models.append(model)
    return models


"""
def get_pretrained_models_by_name(
    model_name, diff_weight_init, gpu_id, num_models, model_file_names, output_dim=10
):
    models = []

    # e.g. model_file_names = ["models/cnn10_sameInit_MNISTsameData_SGD_lr01_momentum09",
    #                    "models/cnn10_sameInit_MNISTsameData_SGD_lr005_momentum09"]

    for idx in range(num_models):
        try:
            state = torch.load(f"./{model_file_names[idx]}.pth")
        except:
            state = torch.load(f"./{model_file_names[idx]}.pth", map_location=torch.device("cpu"))

        print(f"Getting state from: ./{model_file_names[idx]}.pth")

        model = get_model(model_name, output_dim=output_dim)
        if "vgg" in model_name:
            new_state_dict = OrderedDict()
            for k, v in state.items():
                name = k
                name = name.replace(".module", "")
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state)
        if gpu_id != -1:
            model = model.cuda(gpu_id)
        models.append(model)
    return models
"""
