import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-models', default=2, type=int, help='number of models to ensemble')
    parser.add_argument('--gpu-id', default=-1, type=int, help='GPU id to use')
    parser.add_argument('--diff-weight-init', action='store_true', help='Initialize models with different weights')
    parser.add_argument('--model-name', type=str, default='cnn', help='Type of neural network model (cnn | mlp)')
    return parser

def get_parameters():
    parser = get_parser()
    base_args = parser.parse_args()
    return base_args