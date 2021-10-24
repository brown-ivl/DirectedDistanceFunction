import torch
from beacon import utils as butils
import sys, os
import argparse
import numpy as np
import math

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from odf_dataset import DEFAULT_RADIUS
from odf_dataset import ODFDatasetLoader as ODFDL
from single_losses import SingleDepthBCELoss
from single_models import LF4DSingle

def infer(Network, DataLoader, Objective, Device):
    Network.eval()  # switch to evaluation mode
    ValLosses = []
    Tic = butils.getCurrentEpochTime()
    Network.to(Device)
    # print('Val length:', len(ValDataLoader))
    for i, (Data, Targets) in enumerate(DataLoader, 0):  # Get each batch
        DataTD = butils.sendToDevice(Data, Device)
        TargetsTD = butils.sendToDevice(Targets, Device)

        Output = Network(DataTD)
        Loss = Objective(Output, TargetsTD)
        ValLosses.append(Loss.item())

        # Print stats
        Toc = butils.getCurrentEpochTime()
        Elapsed = math.floor((Toc - Tic) * 1e-6)
        done = int(50 * (i + 1) / len(ValDataLoader))
        sys.stdout.write(('\r[{}>{}] val loss - {:.8f}, elapsed - {}')
                         .format('+' * done, '-' * (50 - done), np.mean(np.asarray(ValLosses)),
                                 butils.getTimeDur(Elapsed)))
        sys.stdout.flush()
    sys.stdout.write('\n')

    return ValLosses

Parser = argparse.ArgumentParser(description='Inference code for NeuralODFs.')
Parser.add_argument('--arch', help='Architecture to use.', choices=['standard'], default='standard')
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--rays-per-shape', help='Number of ray samples per object shape.', default=1000, type=int)
Parser.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true', required=False)
Parser.set_defaults(force_test_on_train=False)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.set_defaults(no_posenc=False)

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    butils.seedRandom(Args.seed)

    usePosEnc = not Args.no_posenc
    if Args.arch == 'standard':
        NeuralODF = LF4DSingle(input_size=(120 if usePosEnc else 6), radius=DEFAULT_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc)


    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NeuralODF.setupCheckpoint(Device)
    if Args.force_test_on_train:
        print('[ WARN ]: VALIDATING ON TRAINING DATA.')
    ValData = ODFDL(root=NeuralODF.Config.Args.input_dir, train=Args.force_test_on_train, download=True, n_samples=Args.rays_per_shape, usePositionalEncoding=usePosEnc)
    print('[ INFO ]: Validation data has {} shapes and {} rays per sample.'.format(int(len(ValData) / Args.rays_per_shape), Args.rays_per_shape))

    ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=NeuralODF.Config.Args.batch_size, shuffle=True, num_workers=0) # DEBUG, TODO: More workers not working

    infer(NeuralODF, ValDataLoader, SingleDepthBCELoss(), Device)