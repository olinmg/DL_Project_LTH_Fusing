import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-models', default=2, type=int, help='number of models to ensemble')
    parser.add_argument('--gpu-id', default=3, type=int, help='GPU id to use')
    return parser

def get_parameters():
    parser = get_parser()
    base_args = parser.parse_args()
    
    print(base_args)
    print(type(base_args))
    return base_args