import torch
from beacon import utils as butils
import argparse
import math

from PyQt5.QtWidgets import QApplication
import numpy as np
from tk3dv.pyEasel import *
from Easel import Easel
from tqdm import tqdm

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from odf_dataset import ODFDatasetLiveVisualizer
from pc_sampler import PC_SAMPLER_RADIUS
from single_losses import SingleDepthBCELoss
from single_models import LF4DSingle
from pc_odf_dataset import PCODFDatasetLoader as PCDL

def infer(Network, DataLoader, Objective, Device):
    Network.eval()  # switch to evaluation mode
    ValLosses = []
    Tic = butils.getCurrentEpochTime()
    Network.to(Device)
    Rays = []
    Intersects = []
    Depths = []
    # print('Val length:', len(ValDataLoader))
    for i, (Data, Targets) in enumerate(DataLoader, 0):  # Get each batch
        Rays.append(Data)
        DataTD = butils.sendToDevice(Data, Device)
        TargetsTD = butils.sendToDevice(Targets, Device)

        Output = Network(DataTD)
        Loss = Objective(Output, TargetsTD)
        ValLosses.append(Loss.item())

        Intersects.append(Output[0] > 0.5)
        Depths.append(Output[1].detach().cpu())

        # Print stats
        Toc = butils.getCurrentEpochTime()
        Elapsed = math.floor((Toc - Tic) * 1e-6)
        done = int(50 * (i + 1) / len(ValDataLoader))
        sys.stdout.write(('\r[{}>{}] val loss - {:.8f}, elapsed - {}')
                         .format('+' * done, '-' * (50 - done), np.mean(np.asarray(ValLosses)),
                                 butils.getTimeDur(Elapsed)))
        sys.stdout.flush()
    sys.stdout.write('\n')

    Rays = torch.squeeze(torch.cat(Rays))
    Intersects = torch.squeeze(torch.cat(Intersects))
    Depths = torch.squeeze(torch.cat(Depths))

    return ValLosses, Rays, Intersects, Depths

Parser = argparse.ArgumentParser(description='Inference code for NeuralODFs.')
Parser.add_argument('--arch', help='Architecture to use.', choices=['standard'], default='standard')
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--rays-per-shape', help='Number of ray samples per object shape.', default=1000, type=int)
Parser.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true', required=False)
Parser.set_defaults(force_test_on_train=False)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.set_defaults(no_posenc=True) # Debug, fix this
Parser.add_argument('-v', '--viz-limit', help='Limit visualizations to these many rays.', required=False, type=int, default=1000)

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    butils.seedRandom(Args.seed)

    usePosEnc = not Args.no_posenc
    print('[ INFO ]: Using positional encoding:', usePosEnc)
    if Args.arch == 'standard':
        NeuralODF = LF4DSingle(input_size=(120 if usePosEnc else 6), radius=PC_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc)

    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NeuralODF.setupCheckpoint(Device)
    if Args.force_test_on_train:
        print('[ WARN ]: VALIDATING ON TRAINING DATA.')
    ValData = PCDL(root=NeuralODF.Config.Args.input_dir, train=Args.force_test_on_train, download=True, target_samples=Args.rays_per_shape, usePositionalEncoding=usePosEnc)
    print('[ INFO ]: Validation data has {} shapes and {} rays per sample.'.format(len(ValData), Args.rays_per_shape))

    ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=NeuralODF.Config.Args.batch_size, shuffle=False, num_workers=0) # DEBUG, TODO: More workers not working

    ValLosses, Rays, Intersects, Depths = infer(NeuralODF, ValDataLoader, SingleDepthBCELoss(), Device)
    # if usePosEnc:
    #     Rays = []
    #     print('[ INFO]: Converting from positional encoding to normal...')
    #     for Idx in tqdm(range(len(ValData))):
    #         Rays.append(ValData.__getitem__(Idx, PosEnc=False)[0])
    #     Rays = torch.cat(Rays, dim=0)

    app = QApplication(sys.argv)
    LoadedData = ValData[0]

    GTViz = ODFDatasetLiveVisualizer(coord_type='direction', rays=LoadedData[0].cpu(),
                             intersects=LoadedData[1][0].cpu(), depths=LoadedData[1][1].cpu(),
                             DataLimit=Args.viz_limit, Offset=[-1, 0, 0])
    PredViz =ODFDatasetLiveVisualizer(coord_type='direction', rays=Rays,
                             intersects=Intersects, depths=Depths,
                             DataLimit=Args.viz_limit, Offset=[1, 0, 0])
    CompareViz = Easel([GTViz, PredViz], sys.argv[1:])
    CompareViz.show()
    sys.exit(app.exec_())
