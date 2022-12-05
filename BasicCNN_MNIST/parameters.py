import argparse


def get_parser_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-models', default=2, type=int, help='number of models to ensemble')
    parser.add_argument('--gpu-id', default=-1, type=int, help='GPU id to use')
    parser.add_argument('--model-name', type=str, default='cnn', help='Type of neural network model (cnn | mlp)')
    parser.add_argument('--save-result', action='store_true', help='Saving model weights of fused model in result folder')
    parser.add_argument('--no-bias', action='store_true', help='Create model without bias in every layer')
    return parser

def get_parser_models():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-models', default=2, type=int, help='number of models to ensemble')
    parser.add_argument('--gpu-id', default=-1, type=int, help='GPU id to use')
    parser.add_argument('--diff-weight-init', action='store_true', help='Initialize models with different weights')
    parser.add_argument('--model-name', type=str, default='cnn', help='Type of neural network model (cnn | mlp)')
    parser.add_argument('--no-bias', action='store_true', help='Create model without bias in every layer')
    return parser

def get_parameters_main():
    parser = get_parser_main()
    base_args = parser.parse_args()
    return base_args

def get_parameters_models():
    parser = get_parser_models()
    base_args = parser.parse_args()
    return base_args