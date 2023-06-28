import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-models", default=2, type=int, help="number of models to ensemble")
    parser.add_argument("--gpu-id", default=-1, type=int, help="GPU id to use")
    parser.add_argument(
        "--diff-weight-init", action="store_true", help="Initialize models with different weights"
    )
    parser.add_argument(
        "--model-name", type=str, default="cnn", help="Type of neural network model (cnn | mlp)"
    )
    return parser


def get_parameters():
    parser = get_parser()
    base_args = parser.parse_args()
    return base_args


def get_train_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-models", default=6, type=int, help="number of models to train")
    parser.add_argument("--gpu-id", default=-1, type=int, help="GPU id to use")
    parser.add_argument(
        "--diff-weight-init", action="store_true", help="Initialize models with different weights"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cnn",
        help="Type of neural network model (cnn | mlp | vgg | resnet)",
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset to use (mnist | cifar)"
    )
    parser.add_argument("--num-epochs", default=1, type=int, help="Epochs to train")
    parser.add_argument("--save-folder", type=str, default="models", help="Folder to save results")
    parser.add_argument(
        "--kfold",
        type=int,
        default=0,
        help="Split dataset into equal parts and train models on different halves of the data. Must be even and bigger than two.",
    )
    return parser.parse_args()
