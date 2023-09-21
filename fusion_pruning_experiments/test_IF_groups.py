import copy
import torchvision.transforms as transforms
from train_2 import get_pretrained_model
from pruning_modified import prune_structured_new
from torch_pruning_new.optimal_transport import OptimalTransport
import torch_pruning_new as tp
import torch
from torchvision import datasets

def find_ignored_layers(model_original, out_features):
    ignored_layers = []
    for m in model_original.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == out_features:
            ignored_layers.append(m)
    
    return ignored_layers

def get_cifar10_data_loader():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return {"train": train_loader, "test": val_loader}

def evaluate(input_model, loaders, gpu_id):
    """
    Computes the accuracy of a given model (input_model) on a given dataset (loaders["test"]).
    """
    if gpu_id != -1:
        input_model = input_model.cuda(gpu_id)
    input_model.eval()

    accuracy_accumulated = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders["test"]:
            if gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()

            test_output = input_model(images)

            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            accuracy_accumulated += accuracy
            total += 1

    return accuracy_accumulated / total


if __name__ == '__main__':

    example_inputs = torch.randn(1, 3, 32, 32)
    out_features = 10
    gpu_id = -1


    config = dict(
        dataset="Cifar10",
        model="vgg11_bn",
        optimizer="SGD",
        optimizer_decay_at_epochs=[150, 250],
        optimizer_decay_with_factor=10.0,
        optimizer_learning_rate=0.1,
        optimizer_momentum=0.9,
        optimizer_weight_decay=0.0001,
        batch_size=256,
        num_epochs=300,
        seed=42,
    )

    loaders = get_cifar10_data_loader()

    model_original,_ = get_pretrained_model(config, "./vgg11_bn_cifar10_300eps.checkpoint")

    DG = tp.DependencyGraph().build_dependency(model_original, example_inputs=example_inputs)

    ignored_layers = find_ignored_layers(model_original=model_original, out_features=out_features)
    num_groups = 0
    for group in DG.get_all_groups_in_order(ignored_layers=ignored_layers):
        num_groups += 1
    
    
    ot = OptimalTransport(gpu_id=gpu_id)
    meta_pruning_types = [None, ot]
    sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    prune_types = ["l1"]
    groups = [i for i in range(num_groups)]

    for prune_type in prune_types:
        for group_idx in groups:
            for sparsity in sparsities:
                for meta_prune in meta_pruning_types:
                    pruned_model = copy.deepcopy(model_original)
                    prune_structured_new(
                        pruned_model,
                        None,
                        None,
                        example_inputs,
                        out_features,
                        prune_type,
                        gpu_id,
                        sparsity=sparsity,
                        prune_iter_steps=1,
                        optimal_transport=meta_prune,
                        backward_pruning=True,
                        group_idxs=[group_idx],
                        train_fct=None)
                    
                    for ((name_orig, module_orig), (name, module)) in list(zip(model_original.named_modules(), pruned_model.named_modules())):
                        if isinstance(module_orig, (torch.nn.Conv2d, torch.nn.Linear)):
                            print(f"{module_orig.weight.shape} -> {module.weight.shape}")
                    
                    print(evaluate(pruned_model, loaders, gpu_id=gpu_id))


                    
                    










