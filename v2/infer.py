import torch
from beacon import utils as butils
import argparse
import math

from PyQt5.QtWidgets import QApplication
import numpy as np
from tk3dv.pyEasel import *
from Easel import Easel
from tqdm import tqdm
import multiprocessing as mp

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from odf_dataset import ODFDatasetLiveVisualizer, ODFDatasetVisualizer
# from pc_sampler import PC_SAMPLER_RADIUS
from depth_sampler import DEPTH_SAMPLER_RADIUS
from single_losses import SingleDepthBCELoss, SINGLE_MASK_THRESH
from single_models import LF4DSingle
# from pc_odf_dataset import PCODFDatasetLoader as PCDL
from depth_odf_dataset import DepthODFDatasetLoader as DDL
from odf_dataset import ODFDatasetLoader as ODL
import odf_v2_utils as o2utils

def infer(Network, ValDataLoader, Objective, Device, Limit, UsePosEnc):
    Network.eval()  # switch to evaluation mode
    ValLosses = []
    Tic = butils.getCurrentEpochTime()
    Network.to(Device)
    # print('Val length:', len(ValDataLoader))
    Coords = []
    PredIntersects = []
    PredDepths = []
    GTIntersects = []
    GTDepths = []
    for i, (Data, Targets) in enumerate(ValDataLoader, 0):  # Get each batch
        if i >= ValLimit:
            break
        DataPosEnc = Data.copy()
        if UsePosEnc:
            for Idx, BD in enumerate(DataPosEnc):
                DataPosEnc[Idx] = torch.from_numpy(o2utils.get_positional_enc(BD.numpy())).to(torch.float32)
        DataTD = butils.sendToDevice(DataPosEnc, Device)
        TargetsTD = butils.sendToDevice(Targets, Device)

        Output = Network.forward(DataTD, {})
        Loss = Objective(Output, TargetsTD)
        ValLosses.append(Loss.item())

        for b in range(len(Output)):
            Coords.append(Data[b])
            GTIntersects.append(Targets[b][0])
            GTDepths.append(Targets[b][1])
            PredIntersects.append(Output[b][0].detach().cpu())
            PredDepths.append(Output[b][1].detach().cpu())

        # Print stats
        Toc = butils.getCurrentEpochTime()
        Elapsed = math.floor((Toc - Tic) * 1e-6)
        done = int(50 * (i + 1) / len(ValDataLoader))
        sys.stdout.write(('\r[{}>{}] val loss - {:.8f}, elapsed - {}')
                         .format('+' * done, '-' * (50 - done), np.mean(np.asarray(ValLosses)),
                                 butils.getTimeDur(Elapsed)))
        sys.stdout.flush()
    sys.stdout.write('\n')

    return ValLosses, Coords, GTIntersects, GTDepths, PredIntersects, PredDepths

Parser = argparse.ArgumentParser(description='Inference code for NeuralODFs.')
Parser.add_argument('--arch', help='Architecture to use.', choices=['standard'], default='standard')
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--rays-per-shape', help='Number of ray samples per object shape.', default=1000, type=int)
Parser.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true', required=False)
Parser.set_defaults(force_test_on_train=False)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.set_defaults(no_posenc=False)
Parser.add_argument('-v', '--viz-limit', help='Limit visualizations to these many rays.', required=False, type=int, default=1000)
Parser.add_argument('-l', '--val-limit', help='Limit validation samples.', required=False, type=int, default=-1)

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    butils.seedRandom(Args.seed)
    nCores = 0#mp.cpu_count()

    usePosEnc = not Args.no_posenc
    ValLimit = Args.val_limit
    print('[ INFO ]: Using positional encoding:', usePosEnc)
    if Args.arch == 'standard':
        NeuralODF = LF4DSingle(input_size=(120 if usePosEnc else 6), radius=DEPTH_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc)

    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NeuralODF.setupCheckpoint(Device)
    if Args.force_test_on_train:
        print('[ WARN ]: VALIDATING ON TRAINING DATA.')
    ValData = DDL(root=NeuralODF.Config.Args.input_dir, train=Args.force_test_on_train, download=True, target_samples=Args.rays_per_shape, usePositionalEncoding=False) # NOTE: We pass the positional encoding flag to infer function
    if ValLimit < 0:
        ValLimit = len(ValData)
    ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=NeuralODF.Config.Args.batch_size, shuffle=True, num_workers=nCores, collate_fn=DDL.collate_fn)

    print('[ INFO ]: Validation data has {} shapes and {} rays per sample.'.format(len(ValData), Args.rays_per_shape))

    ValLosses, Coords, GTIntersects, GTDepths, PredIntersects, PredDepths = infer(NeuralODF, ValDataLoader, SingleDepthBCELoss(), Device, ValLimit, usePosEnc)

    # if usePosEnc:
    #     Rays = []
    #     print('[ INFO]: Converting from positional encoding to normal...')
    #     for Idx in tqdm(range(len(ValData))):
    #         Rays.append(ValData.__getitem__(Idx, PosEnc=False)[0])
    #     Rays = torch.cat(Rays, dim=0)

    app = QApplication(sys.argv)
    VizIdx = [0, 1, 2, 3, 4]

    GTViz = []
    PredViz = []
    for vidx in VizIdx:
        GTViz.append(ODFDatasetLiveVisualizer(coord_type='direction', rays=Coords[vidx],
                                 intersects=GTIntersects[vidx], depths=GTDepths[vidx],
                                 DataLimit=Args.viz_limit, Offset=[-1, 0, 0]))
        PredViz.append(ODFDatasetLiveVisualizer(coord_type='direction', rays=Coords[vidx],
                                 intersects=PredIntersects[vidx], depths=PredDepths[vidx],
                                 DataLimit=Args.viz_limit, Offset=[1, 0, 0]))
    CompareViz = Easel(GTViz + PredViz, sys.argv[1:])
    CompareViz.show()
    sys.exit(app.exec_())
