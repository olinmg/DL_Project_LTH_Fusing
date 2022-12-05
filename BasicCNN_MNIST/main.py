from parameters import get_parameters_main
from models import get_model
import torch
from fusion import fusion
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import ToTensor


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


def get_pretrained_models(args, num_models):
    models = []

    for idx in range(num_models):
        state = torch.load("models/model_{}_{}.pth".format(args.model_name, idx))
        model = get_model(args)
        model.load_state_dict(state)
        if args.gpu_id != -1:
            model = model.cuda(args.gpu_id)
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
    args = get_parameters_main()
    num_models = args.num_models

    models = get_pretrained_models(args, num_models)

    fused_model = fusion(models, args)

    loaders = get_data_loader(args)
    results = [test(models[i], loaders, args) for i in range(len(models))]
    result_fused = test(fused_model, loaders, args)

    for i in range(len(models)):
        print('Test Accuracy of the model %d: %.2f' % (i, results[i]))
    print('Test Accuracy of the fused model: %.2f' % result_fused)

    torch.save(fused_model.state_dict(), "result/model_fused_{}.pth".format(fused_model.name)) if args.save_result else None



    


