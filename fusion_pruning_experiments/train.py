import copy

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor

from models import get_model
from parameters import get_parameters, get_train_parameters
from vgg import vgg11, vgg13


def get_mnist_data_loader(num_models, args):
    train_data = datasets.MNIST("data", train=True, transform=ToTensor(), download=True)
    test_data = datasets.MNIST("data", train=False, transform=ToTensor(), download=True)

    if args.kfold < 3:
        fraction = 1 / num_models
        train_data_split = (
            torch.utils.data.random_split(train_data, [fraction] * num_models)
            if not args.diff_weight_init
            else [train_data] * num_models
        )
    else:
        import itertools

        fraction = 1 / args.kfold
        train_data_kfold = torch.utils.data.random_split(train_data, [fraction] * args.kfold)
        # for args.kfold==4, we want the (4 choose 2) = 6 possibly pairings of the data subsets
        # each model always gets half the data, since picking args.kfold/2 per set and kfold%2==0
        combinations_of_2_subsets = itertools.combinations(train_data_kfold, args.kfold / 2)
        combinations_of_2_subsets_lis = [combo for combo in combinations_of_2_subsets]

        train_data_split = [
            torch.utils.data.ConcatDataset([combo[0], combo[1]])
            for combo in combinations_of_2_subsets_lis
        ]

    print(f"Total dataset size: {len(train_data)}")
    print("Returning Datasets - sizes:")
    for dat in train_data_split:
        print(len(dat))

    return {
        "train": [
            torch.utils.data.DataLoader(
                train_data_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
            )
            for train_data_subset in train_data_split
        ],
        "test": torch.utils.data.DataLoader(
            test_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
        ),
    }


def get_cifar_data_loader(num_models, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = datasets.CIFAR10(
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
    )

    test_data = datasets.CIFAR10(
        root="./data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    if args.kfold < 3:
        fraction = 1 / num_models
        train_data_split = (
            torch.utils.data.random_split(train_data, [fraction] * num_models)
            if not args.diff_weight_init
            else [train_data] * num_models
        )
    else:
        import itertools

        fraction = 1 / args.kfold
        train_data_kfold = torch.utils.data.random_split(train_data, [fraction] * args.kfold)
        # for args.kfold==4, we want the (4 choose 2) = 6 possibly pairings of the data subsets
        # each model always gets half the data, since picking args.kfold/2 per set and kfold%2==0
        combinations_of_2_subsets = itertools.combinations(train_data_kfold, args.kfold / 2)
        combinations_of_2_subsets_lis = [combo for combo in combinations_of_2_subsets]

        train_data_split = [
            torch.utils.data.ConcatDataset([combo[0], combo[1]])
            for combo in combinations_of_2_subsets_lis
        ]

    print(f"Total dataset size: {len(train_data)}")
    print("Returning Datasets - sizes:")
    for dat in train_data_split:
        print(len(dat))

    return {
        "train": [
            torch.utils.data.DataLoader(
                train_data_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
            )
            for train_data_subset in train_data_split
        ],
        "test": torch.utils.data.DataLoader(
            test_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
        ),
    }


def get_cifar100_data_loader(num_models, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = datasets.CIFAR100(
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
    )

    test_data = datasets.CIFAR100(
        root="./data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    if args.kfold < 3:
        fraction = 1 / num_models
        train_data_split = (
            torch.utils.data.random_split(train_data, [fraction] * num_models)
            if not args.diff_weight_init
            else [train_data] * num_models
        )
    else:
        import itertools

        fraction = 1 / args.kfold
        train_data_kfold = torch.utils.data.random_split(train_data, [fraction] * args.kfold)
        # for args.kfold==4, we want the (4 choose 2) = 6 possibly pairings of the data subsets
        # each model always gets half the data, since picking args.kfold/2 per set and kfold%2==0
        combinations_of_2_subsets = itertools.combinations(train_data_kfold, args.kfold / 2)
        combinations_of_2_subsets_lis = [combo for combo in combinations_of_2_subsets]

        train_data_split = [
            torch.utils.data.ConcatDataset([combo[0], combo[1]])
            for combo in combinations_of_2_subsets_lis
        ]

    print(f"Total dataset size: {len(train_data)}")
    print("Returning Datasets - sizes:")
    for dat in train_data_split:
        print(len(dat))

    return {
        "train": [
            torch.utils.data.DataLoader(
                train_data_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
            )
            for train_data_subset in train_data_split
        ],
        "test": torch.utils.data.DataLoader(
            test_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
        ),
    }


import os


def get_imagenet_data_loader():
    traindir = os.path.join("/local/home/stuff/imagenet", "train")
    valdir = os.path.join("/local/home/stuff/imagenet", "val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    fraction = 1 / num_models
    original_seed = torch.initial_seed()
    torch.manual_seed(99)
    train_data_split = (
        torch.utils.data.random_split(train_dataset, [fraction] * num_models)
        if not args.diff_weight_init
        else [train_dataset] * num_models
    )
    torch.manual_seed(original_seed)

    if False:
        # can turn this on
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = [
        torch.utils.data.DataLoader(
            train_data_subset,
            batch_size=256,
            shuffle=(train_sampler is None),
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler,
        )
        for train_data_subset in train_data_split
    ]

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler,
    )

    return {"train": train_loader, "test": val_loader}


def train(model, loaders, num_epochs, gpu_id):
    """
    Has to be a function that loads a dataset.
    A given model and an amount of epochs of training will be given.
    """

    loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr = 0.05)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    model.train()

    # Train the model
    total_step = len(loaders["train"])

    val_acc_per_epoch = []
    best_model = model
    best_model_accuracy = -1
    lr = 0.05
    for epoch in range(num_epochs):
        optimizer = optim.SGD(model.parameters(), lr=lr * (0.5 ** (epoch // 30)), momentum=0.9)
        for i, (images, labels) in enumerate(loaders["train"]):
            if next(model.parameters()).is_cuda:
                images, labels = images.cuda(), labels.cuda()
            # gives batch data, normalize x when iterate train_loader

            predictions = model(images)
            loss = loss_func(predictions, labels)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

        this_epoch_acc = test(model, loaders=loaders, gpu_id=gpu_id)
        val_acc_per_epoch.append(this_epoch_acc)
        if this_epoch_acc > best_model_accuracy:
            best_model = copy.deepcopy(model)
            best_model_accuracy = this_epoch_acc

    return best_model, best_model_accuracy


# define the testing function
def test(model, loaders, gpu_id):
    model.eval()

    accuracy_accumulated = 0
    total = 0
    with torch.no_grad():
        for images, labels in loaders["test"]:
            if gpu_id != -1:
                images, labels = images.cuda(), labels.cuda()
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))

            accuracy_accumulated += accuracy
            total += 1
    return accuracy_accumulated / total


# actually execute the training and testing
if __name__ == "__main__":
    args = get_train_parameters()
    num_models = args.num_models
    if args.kfold % 2 == 1:
        args.kfold = 0
        print("args.kfold is not even - need even so can give every model 50% of the data.")
        print("Reverting to default: args.kfold=0")
    loaders = None
    if args.dataset == "cifar10":
        loaders = get_cifar_data_loader(num_models, args)
    elif args.dataset == "cifar100":
        print("went in here!!")
        loaders = get_cifar100_data_loader(num_models, args)
    else:
        loaders = get_mnist_data_loader(num_models, args)

    model_parent = get_model(args.model_name, output_dim=100 if args.dataset == "cifar100" else 10)
    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in enumerate(
        zip(model_parent.named_parameters(), model_parent.named_parameters())
    ):
        print(f"{layer0_name} : {fc_layer0_weight.shape}")

    for idx in range(num_models):
        model = (
            copy.deepcopy(model_parent)
            if not args.diff_weight_init
            else get_model(args.model_name, output_dim=100 if args.dataset == "cifar100" else 10)
        )
        if args.gpu_id != -1:
            model = model.cuda(args.gpu_id)
        model, _ = train(
            model,
            {"train": loaders["train"][idx], "test": loaders["test"]},
            args.num_epochs,
            args.gpu_id,
        )
        accuracy = test(model, loaders, args.gpu_id)

        print("Test Accuracy of the model %d: %.2f" % (idx, accuracy))
        # store the trained model and performance
        if args.kfold > 2:
            kfold_text = args.kfold
        else:
            kfold_text = False
        torch.save(
            model,
            "models/models_{}/{}_diffWeightInit{}_kfold{}_nrModels{}_{}_eps{}_{}.pth".format(
                args.model_name,
                args.model_name,
                args.diff_weight_init,
                kfold_text,
                args.num_models,
                args.dataset,
                args.num_epochs,
                idx,
            ),
        )
