import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
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



# 3. defining a basic CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
                ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )

        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),          
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization



# 4. instantiating necessary objects: putting the pieces together
cnn_model = CNN()
loss_func = nn.CrossEntropyLoss()   
optimizer = optim.Adam(cnn_model.parameters(), lr = 0.01)



# 5. define the training function
num_epochs = 10
def train(num_epochs, cnn_model, loaders):
    
    cnn_model.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn_model(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
            
        pass
    pass



# 6. define the testing function
def test(model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
            print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    pass



# 7. actually execute the training and testing
if __name__ == '__main__':
    train(num_epochs, cnn_model, loaders)
    test(cnn_model)

    # test outputs
    sample = next(iter(loaders['test']))
    imgs, lbls = sample
    actual_number = lbls[:10].numpy()
    test_output, last_layer = cnn_model(imgs[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(f'Prediction number: {pred_y}')
    print(f'Actual number: {actual_number}')
    print("Done.")
    # 8. store the trained model and performance
    torch.save(cnn_model.state_dict(), "./base_cnn_model_dict_weak.pth")

