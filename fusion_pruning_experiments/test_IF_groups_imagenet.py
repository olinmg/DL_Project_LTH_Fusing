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

import torchvision.models as model_archs
def get_pretrained_resnet50(model_file_path, gpu_id):
    model = model_archs.__dict__["resnet50"]()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(f"{model_file_path}.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.module.to("cpu")
    model = model.cuda(gpu_id) if gpu_id != -1 else model.to("cpu")
    return model

import os
def get_imagenet_data_loader():
    folder = "/local/home/gaf/coolvenv/ICLR_env/DL_Project_LTH_Fusing/fusion_pruning_experiments/fake_imagenet" # "/local/home/stuff/imagenet"
    traindir = os.path.join(folder, "train")
    valdir = os.path.join(folder, "val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    '''train_dataset = datasets.ImageFolder(
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
    '''
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

    if False:
        # can turn this on
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True
        )
    else:
        train_sampler = None
        val_sampler = None

    '''train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )'''

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler,
    )

    return {"train": None, "test": val_loader}

import torch.distributed as dist
from enum import Enum
class Summary(Enum):
    # Only used to work with imagenet
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    # Only used to work with imagenet

    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)

import torch.nn as nn
import time
def accuracy(output, target, topk=(1,)):
    # only used for imagenet/resnet50
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ProgressMeter(object):
    # Only used to work with imagenet
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

from torch.utils.data import Subset
def evaluate_performance_imagenet(
    input_model, loaders, gpu_id, prune=True, distributed=False, world_size=1
):
    if gpu_id != -1:
        input_model = input_model.cuda(gpu_id)

    model = input_model
    val_loader = loaders["test"]

    # def validate(val_loader, model, criterion, args): # from main.py in https://github.com/pytorch/examples/tree/main/imagenet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)

    # Only used to work with imagenet
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if gpu_id is not None and torch.cuda.is_available():
                    images = images.cuda(gpu_id, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to("mps")
                    target = target.to("mps")
                if torch.cuda.is_available():
                    target = target.cuda(gpu_id, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                """
                if i % args.print_freq == 0:
                    progress.display(i + 1)
                """

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader)
        + (distributed and (len(val_loader.sampler) * world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if distributed:
        top1.all_reduce()
        top5.all_reduce()

    if distributed and (len(val_loader.sampler) * world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * world_size, len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    if prune:
        model.cpu()
    return top1.avg

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


import json
import torchvision.models as tvmodels

if __name__ == '__main__':
    
    example_inputs = torch.randn(1, 3, 224, 224)
    out_features = 1000
    gpu_id = 0
    backward_pruning = True
    model_name = "resnet50"
    dataset = "Imagenet"


    loaders = get_imagenet_data_loader()

    # using the pretrained model from torchvision
    model_original = tvmodels.resnet50(pretrained=True)

    # model_original = get_pretrained_resnet50("./trained_models/resnet50_imagenet/seed_0/resnet50_imagenet_pretrained_0", gpu_id)

    print(evaluate_performance_imagenet(model_original, loaders, gpu_id=gpu_id))

    DG = tp.DependencyGraph().build_dependency(model_original, example_inputs=example_inputs)

    ignored_layers = find_ignored_layers(model_original=model_original, out_features=out_features)
    num_groups = 0
    for group in DG.get_all_groups_in_order(ignored_layers=ignored_layers):
        num_groups += 1

    output_file_idx = 0
    output_file_name = f"{model_name}_{dataset}_{backward_pruning}_{output_file_idx}.json"

    while os.path.isfile(output_file_name):
        output_file_idx += 1
        output_file_name = f"{model_name}_{dataset}_{backward_pruning}_{output_file_idx}.json"

    
    ot = OptimalTransport(gpu_id=gpu_id)
    meta_pruning_types = [None, ot]
    sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    prune_types = ["l1"]
    groups = [i for i in range(num_groups)]

    dict = {}
    dict["backward_pruning"] = backward_pruning # Important to know what indice of group means

    for prune_type in prune_types:
        dict[prune_type] = {}
        for group_idx in groups:
            dict[prune_type][group_idx] = {}
            for sparsity in sparsities:
                dict[prune_type][group_idx][sparsity] = {}
                for meta_prune in meta_pruning_types:
                    meta_prune_type = "default" if meta_prune == None else "IF"
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
                        backward_pruning=backward_pruning,
                        group_idxs=[group_idx],
                        train_fct=None)
                    
                    for ((name_orig, module_orig), (name, module)) in list(zip(model_original.named_modules(), pruned_model.named_modules())):
                        if isinstance(module_orig, (torch.nn.Conv2d, torch.nn.Linear)):
                            print(f"{module_orig.weight.shape} -> {module.weight.shape}")
                    
                    dict[prune_type][group_idx][sparsity][meta_prune_type] = evaluate_performance_imagenet(pruned_model, loaders, gpu_id=gpu_id)
                    
                    print(f"{prune_type} : {group_idx} : {sparsity} : {meta_prune_type} : {dict[prune_type][group_idx][sparsity][meta_prune_type]}")


                    with open(output_file_name, "w") as file:
                        json.dump(dict, file, indent=4)


                    
                    










