import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from beacon import utils as butils
import sys, os
import argparse

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))
# sys.path.append(os.path.join(FileDirPath, '../'))

import odf_dataset
from odf_dataset import ODFDatasetLoader as ODFDL
from single_losses import SingleDepthBCELoss
from single_models import LF4DSingle

Parser = argparse.ArgumentParser(description='Training code for NeuralODFs.')
Parser.add_argument('--arch', help='Architecture to use.', choices=['standard'], default='standard')
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--infer-samples', help='Number of samples to use during testing.', default=30, type=int)
Parser.add_argument('--no-val', help='Choose to not perform validation during training.', action='store_true', required=False)
Parser.set_defaults(no_val=False)  # True for DEBUG only todo
Parser.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true', required=False)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    butils.seedRandom(Args.seed)

    usePosEnc = True

    if Args.arch == 'standard':
        NeuralODF = LF4DSingle(input_size=(120 if usePosEnc else 6), radius=odf_dataset.DEFAULT_RADIUS,
                     coord_type=Args.coord_type, pos_enc=usePosEnc)

    TrainDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TrainData = ODFDL(root=Args.data_dir, train=True, download=True, mode=Args.mode, n_samples=Args.nsamples)
    ValData = ODFDL(root=Args.data_dir, train=False, download=True, mode=Args.mode, n_samples=Args.nsamples)
    print('[ INFO ]: Data has', len(TrainData), 'samples.')
    TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=NeuralODF.Config.Args.batch_size, shuffle=True, num_workers=4)
    if Args.no_val == False:
        ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=NeuralODF.Config.Args.batch_size, shuffle=True, num_workers=4)
    else:
        print('[ WARN ]: Not validating during training. This should be used for debugging purposes only.')
        ValDataLoader = None

    NeuralODF.fit(TrainDataLoader, Objective=SingleDepthBCELoss(), TrainDevice=TrainDevice, ValDataLoader=ValDataLoader)