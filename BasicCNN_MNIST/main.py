from collections import OrderedDict
from parameters import get_parameters
from base_convNN import get_model
import torch
from fusion import fusion, fusion_old
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable


def get_data_loader(args):
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )    

    # 2. defining the data loader for train and test set using the downloaded MNIST data
    loaders = {  
        'test'  : torch.utils.data.DataLoader(test_data, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
    }
    return loaders


def get_pretrained_models(model_name, diff_weight_init, gpu_id, num_models):
    models = []

    for idx in range(num_models):
        state = torch.load(f"models/{model_name}_diff_weight_init_{diff_weight_init}_{idx}.pth")
        model = get_model(model_name)
        if "vgg" in model_name:
            new_state_dict = OrderedDict()
            for k, v in state.items():
                print(k)
                name = k
                name = name.replace(".module", "")
                print("new name is: ", name)
                new_state_dict[name] = v
            print(new_state_dict.keys())
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state)
        if gpu_id != -1:
            model = model.cuda(gpu_id)
        models.append(model)
    return models

def test(model, loaders, args):
    model.eval()

    accuracy_accumulated = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            if args.gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()
            test_output,_ = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            accuracy_accumulated += accuracy 
            total += 1
    return accuracy_accumulated / total


if __name__ == '__main__':
    args = get_parameters()
    num_models = args.num_models

    assert num_models == 2 # Only temporary, later we can extend to more layers

    models = get_pretrained_models(args.model_name, args.diff_weight_init, args.gpu_id, args.num_models)

    fused_model = fusion_old(models, args)

    loaders = get_data_loader(args)
    accuracy_0 = test(models[0], loaders, args)
    accuracy_1 = test(models[1], loaders, args)
    accuracy_fused = test(fused_model, loaders, args)

    print('Test Accuracy of the model 0: %.2f' % accuracy_0)
    print('Test Accuracy of the model 1: %.2f' % accuracy_1)
    print('Test Accuracy of the model fused: %.2f' % accuracy_fused)



    


