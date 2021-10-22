import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import sys
import argparse

Parser = argparse.ArgumentParser(description='Training code for NeuralODFs.')
Parser.add_argument('--arch', help='Architecture to use.', choices=['Standard'], default='Standard')

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

