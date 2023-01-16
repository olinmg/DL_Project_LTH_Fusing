from collections import OrderedDict
import torch
import torch.nn as nn
from vgg import VGG, make_layers
from resnet import BasicBlock, Bottleneck, ResNet


class MLP(nn.Module):
    def __init__(self, sparsity=1.0):
        super(MLP, self).__init__()
        bias = True 
        self.sparsity=sparsity
        self.lin1 = nn.Sequential(
            nn.Linear(28*28, int(128*self.sparsity), bias=bias),
            nn.ReLU(),  
        )
        self.lin2 = nn.Sequential(
            nn.Linear(int(128*self.sparsity), int(512*self.sparsity), bias=bias),
            nn.ReLU(),  
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(int(512*self.sparsity), 10, bias=bias)
    
    def forward(self, x):

        x = self.lin1(x.view(-1, 28*28))
        x = self.lin2(x)  
        output = self.out(x)
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        bias = True
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,
                bias = bias # Needs to change later!                  
                ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )

        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2, bias=bias),     
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


def ResNet18(num_classes=10, use_batchnorm=False, linear_bias=True):
    print("linear_bias is: ", linear_bias)
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def ResNet34(num_classes=10, use_batchnorm=False, linear_bias=True):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def ResNet50(num_classes=10, use_batchnorm=False, linear_bias=True):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def ResNet101(num_classes=10, use_batchnorm=False, linear_bias=True):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def ResNet152(num_classes=10, use_batchnorm=False, linear_bias=True):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}
def vgg11(bias=False):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], bias=bias), bias=bias)


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))

def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))

def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))

def get_model(model_name, sparsity=1.0):
    if model_name == "cnn":
        return CNN()
    elif model_name == "mlp":
        return MLP(sparsity)
    elif model_name == "vgg11":
        return vgg11(bias=False)
    elif model_name == "vgg13":
        return vgg13()
    elif model_name == "vgg16":
        return vgg16()
    elif model_name == "vgg19":
        return vgg19()
    elif model_name == "resnet18":
        return ResNet18()
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

def get_pretrained_models(model_name, diff_weight_init, gpu_id, num_models, model_file_names):
    models = []

    #model_file_names = ["cnn10_sameInit_MNISTsameData_SGD_lr01_momentum09",
    #                    "cnn10_sameInit_MNISTsameData_SGD_lr005_momentum09"]

    for idx in range(num_models):
        #state = torch.load(f"models/{model_name}_diff_weight_init_{diff_weight_init}_{idx}.pth")
        state = torch.load(f"./{model_file_names[idx]}.pth", map_location=torch.device('cpu'))
        print(f"Getting state from: ./{model_file_names[idx]}.pth")

        model = get_model(model_name)
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
