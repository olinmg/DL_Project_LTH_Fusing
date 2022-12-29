# already pretrained networks of all kinds: torchvision.models
# https://pytorch.org/vision/0.8/models.html
# get test data from: https://pytorch.org/vision/0.8/datasets.html#cifar


# build a basic ResNet and evaluate pruning on it.
# https://medium.com/analytics-vidhya/resnet-10f4ef1b9d4c

# https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

import torch
from torchvision import models
from torchvision import datasets
from torchvision.transforms import ToTensor
from parameters import get_parameters
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
# path to save the cifar dataset to
data_path = 'data'


def get_cifar_data_loader():

    cifar10_train = datasets.CIFAR10(data_path, train=True, transform = ToTensor(), download=True)
    cifar10_test = datasets.CIFAR10(data_path, train=False, transform = ToTensor(), download=True)

    loaders = {  
        'train'  : torch.utils.data.DataLoader(cifar10_train, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
        'test'  : torch.utils.data.DataLoader(cifar10_test, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
    }
    return loaders


def get_pretrained_resnet_imagenet():
    '''
    Be careful. This is pretrained on ImageNet, not on CIFAR10 ! 
    '''
    resnet = models.resnet18(pretrained=True)
    return resnet


def get_untrained_resnet():
    '''
    Gets an untrained resnet that is supposed to be used for ImageNet classification.
    Changes the output dimension so it can be used to predict CIFAR10.
    '''
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    return resnet


def evaluate_performance_simple(input_model, loaders, args):
    '''
    Computes the accuracy of a given model (input_model) on a given dataset (loaders["test"]).
    '''
    
    input_model.eval()

    accuracy_accumulated = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            if args.gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()
            test_output, _ = input_model(images)        # TODO: WHY DOES THIS RETURN TWO VALUES?!
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            accuracy_accumulated += accuracy 
            total += 1
    return accuracy_accumulated / total


def train(num_epochs, model, loaders, args):

    loss_func = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    model.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    val_acc_per_epoch = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            if args.gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y

            predictions = model(b_x)
            loss = loss_func(predictions, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
        this_epoch_acc = evaluate_performance_simple(input_model=model, loaders=loaders, args=args, para_dict={})
        val_acc_per_epoch.append(this_epoch_acc)
    return model, val_acc_per_epoch


def train_resnet_on_cifar10(num_epochs):

    cifar_loader = get_cifar_data_loader()
    resnet_model = get_untrained_resnet()
    args = get_parameters()

    # need to modify resnet, since its written to predict ImageNet (1000 Classes), not Cifar10 (10 classes)
    #resnet_model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    trained_model, val_acc_per_epoch = train(num_epochs, resnet_model, cifar_loader, args)

    return resnet_model, val_acc_per_epoch


def test_model_on_cifar10(model):
    cifar_loaders = get_cifar_data_loader()
    args = get_parameters()
    print(f"Accuracy of the model when predicting cifar10: {evaluate_performance_simple(input_model=model, loaders=cifar_loaders, args=args, para_dict={})}")


if __name__ == '__main__':
    # get an untrained resnet model
    num_epochs = 25
    trained_resnet, val_acc_per_epoch = train_resnet_on_cifar10(num_epochs=num_epochs)

    # Store the trained resnet
    torch.save(trained_resnet.state_dict(), "resnet_cifar10_model_dict_25new.pth")

    # Evaluate the performance of the trained resnet
    print(val_acc_per_epoch)
    plt.plot(val_acc_per_epoch)
    plt.show()

    test_model_on_cifar10(trained_resnet)