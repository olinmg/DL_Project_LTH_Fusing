import copy
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from vgg import vgg11, vgg13
from parameters import get_parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# 1. import MNIST dataset
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)



# 2. defining the data loader for train and test set using the downloaded MNIST data
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}

# returns the DataLoader for test and train data, where the train data is split into the amount of models specified
def get_data_loaders(num_models, args):
    fraction = 1/num_models
    train_data_split = torch.utils.data.random_split(train_data, [fraction]*num_models) if not args.diff_weight_init else [train_data]*num_models
    
    return {
        "train": [torch.utils.data.DataLoader(train_data_subset, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1) for train_data_subset in train_data_split],
        "test" : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
        }

# 3. defining a basic CNN model
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
        return output, x    # return x for visualization

# 3. defining a basic CNN model
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
        return output, x    # return x for visualization

# 4. instantiating necessary objects: putting the pieces together
loss_func = nn.CrossEntropyLoss()   


# 5. define the training function
num_epochs = 10
def train(num_epochs, model, loaders, args):
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    model.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            if args.gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))



# 6. define the testing function
def test(model, args):
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
    return accuracy_accumulated/total


def get_model(model_name, sparsity=1.0):
    if model_name == "cnn":
        return CNN()
    elif model_name == "mlp":
        return MLP(sparsity)
    elif model_name == "vgg11":
        return vgg11()
    elif model_name == "vgg13":
        return vgg13()
    else:
        print("Invalid model name. Using CNN instead.")
        return CNN()

# 7. actually execute the training and testing
if __name__ == '__main__':
    args = get_parameters()
    num_models = args.num_models
    loaders = get_data_loaders(num_models, args)

    sparsity = 1.0
    
    model_parent = get_model(args.model_name)
    for idx in range(num_models):
        model = copy.deepcopy(model_parent) if not args.diff_weight_init else get_model(args.model_name, sparsity=1.0 if idx==0 else sparsity)
        if args.gpu_id != -1:
            model = model.cuda(args.gpu_id)
        train(num_epochs, model, {"train": loaders["train"][idx], "test": loaders["test"]}, args)
        accuracy = test(model, args)

        print('Test Accuracy of the model %d: %.2f' % (idx, accuracy))
        # 8. store the trained model and performance
        torch.save(model.state_dict(), "models/{}_diff_weight_init_{}_{}.pth".format(args.model_name, args.diff_weight_init, idx))

