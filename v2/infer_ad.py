from numpy.lib.npyio import load
import torch
from beacon import utils as butils
from beacon.supernet import SuperLoss
import argparse
import math

from PyQt5.QtWidgets import QApplication
import numpy as np
from tk3dv.pyEasel import *
from Easel import Easel
from tqdm import tqdm
import multiprocessing as mp

from odf_v2_utils import load_latent_vectors

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from odf_dataset import ODFDatasetLiveVisualizer, ODFDatasetVisualizer
from pc_sampler import PC_SAMPLER_RADIUS
from single_losses import SingleDepthBCELoss, SINGLE_MASK_THRESH, ADPredLoss, ADRegLoss
from single_models import LF4DSingleAutoDecoder
from pc_odf_dataset import PCODFDatasetLoader as PCDL
from odf_dataset import ODFDatasetLoader as ODL

def infer(Network, ValDataLoader, Objective, Device, Limit, OtherParameters):
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
        DataTD = butils.sendToDevice(Data, Device)
        TargetsTD = butils.sendToDevice(Targets, Device)

        print("LEN of data in infere ***********************")
        print(len(DataTD))
        Output = Network.forward(DataTD, OtherParameters)
        Loss = Objective(Output, TargetsTD)
        ValLosses.append(Loss.item())

        for b in range(len(Output)):
            Coords.append(Data[b][0].detach().cpu())
            GTIntersects.append(Targets[b][0])
            GTDepths.append(Targets[b][1])
            PredIntersects.append(Output[b][0][0].detach().cpu())
            PredDepths.append(Output[b][0][1].detach().cpu())

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

def infer_instance(Network, ValData, Objective, Device, Limit, OtherParameters, idx=0):
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
    # for i, (Data, Targets) in enumerate(ValDataLoader, 0):  # Get each batch
    Data, Targets = ValData[idx]
    # TODO: issue using batch size of one. Probably a squeeze somewhere
    Data = [Data, Data]
    Targets = [Targets, Targets]

    DataTD = butils.sendToDevice(Data, Device)
    TargetsTD = butils.sendToDevice(Targets, Device)

    print("LEN of data in infere_instance ***********************")
    print(len(DataTD))

    Output = Network.forward(DataTD, OtherParameters)
    Loss = Objective(Output, TargetsTD)
    ValLosses.append(Loss.item())

    for b in range(len(Output)):
        Coords.append(Data[b][0].detach().cpu())
        GTIntersects.append(Targets[b][0])
        GTDepths.append(Targets[b][1])
        PredIntersects.append(Output[b][0][0].detach().cpu())
        PredDepths.append(Output[b][0][1].detach().cpu())

    # Print stats
    Toc = butils.getCurrentEpochTime()
    Elapsed = math.floor((Toc - Tic) * 1e-6)
    # done = int(50 * (i + 1) / len(ValDataLoader))
    # sys.stdout.write(('\r[{}>{}] val loss - {:.8f}, elapsed - {}')
    #                     .format('+' * done, '-' * (50 - done), np.mean(np.asarray(ValLosses)),
    #                             butils.getTimeDur(Elapsed)))
    sys.stdout.flush()
    sys.stdout.write('\n')

    return ValLosses, Coords, GTIntersects, GTDepths, PredIntersects, PredDepths

Parser = argparse.ArgumentParser(description='Inference code for NeuralODFs autodecoder.')
Parser.add_argument('--arch', help='Architecture to use.', choices=['standard'], default='standard')
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--rays-per-shape', help='Number of ray samples per object shape.', default=1000, type=int)
Parser.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true', required=False)
Parser.set_defaults(force_test_on_train=False)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.set_defaults(no_posenc=True) # Debug, fix this
Parser.add_argument('-v', '--viz-limit', help='Limit visualizations to these many rays.', required=False, type=int, default=1000)
Parser.add_argument('-l', '--val-limit', help='Limit validation samples.', required=False, type=int, default=-1)
Parser.add_argument('--latent-size', type=int, default=256, help="The size of the latent vector for the autodecoder")
Parser.add_argument('--latent-stdev', type=float, default=0.001**2, help="The standard deviation of the zero mean gaussian used to initialize latent vectors")
Parser.add_argument('--use_l2', action="store_true", help="Use L2 loss instead of L1 loss")
Parser.add_argument('--viz-idx', type=int, default=0, help="The dataset index to visualize")


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
        NeuralODF = LF4DSingleAutoDecoder(input_size=(120 if usePosEnc else 6), radius=PC_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc, latent_size=Args.latent_size)


    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
    if Args.force_test_on_train:
        print('[ WARN ]: VALIDATING ON TRAINING DATA.')
    ValData = PCDL(root=NeuralODF.Config.Args.input_dir, train=Args.force_test_on_train, download=True, target_samples=Args.rays_per_shape, usePositionalEncoding=usePosEnc, ad=True)

    # lat_vecs = torch.nn.Embedding(len(ValData.LoadedOBJs), Args.latent_size, max_norm=8*Args.latent_stdev)
    lat_vecs = torch.nn.Embedding(len(ValData.LoadedOBJs), Args.latent_size)
    lat_vecs = lat_vecs.to(Device)
    OtherParameters = [lat_vecs]
    OtherParameterNames = ["Latent Vectors"]
    OtherParamDict = {OtherParameterNames[i]: OtherParameters[i] for i in range(len(OtherParameters))}
    NeuralODF.setupCheckpoint(Device, OtherParameters=OtherParameters, OtherParameterNames=OtherParameterNames)

    if ValLimit < 0:
        ValLimit = len(ValData)
    print(f"LEN ValData: {len(ValData)}")
    # print(ValData[0])
    ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=NeuralODF.Config.Args.batch_size, shuffle=False, num_workers=nCores, collate_fn=PCDL.collate_fn)

    print('[ INFO ]: Validation data has {} shapes and {} rays per sample.'.format(len(ValData), Args.rays_per_shape))

    loss = SuperLoss(Losses=[ADPredLoss(use_l2=Args.use_l2), ADRegLoss()], Weights=[1.0,1.0])

    # ValLosses, Coords, GTIntersects, GTDepths, PredIntersects, PredDepths = infer(NeuralODF, ValDataLoader, loss, Device, ValLimit, OtherParamDict)
    ValLosses, Coords, GTIntersects, GTDepths, PredIntersects, PredDepths = infer_instance(NeuralODF, ValData, loss, Device, ValLimit, OtherParamDict, idx=Args.viz_idx)

    # if usePosEnc:
    #     Rays = []
    #     print('[ INFO]: Converting from positional encoding to normal...')
    #     for Idx in tqdm(range(len(ValData))):
    #         Rays.append(ValData.__getitem__(Idx, PosEnc=False)[0])
    #     Rays = torch.cat(Rays, dim=0)

    app = QApplication(sys.argv)

    VizIdx = 0

    GTViz = ODFDatasetLiveVisualizer(coord_type='direction', rays=Coords[VizIdx],
                            intersects=GTIntersects[VizIdx], depths=GTDepths[VizIdx],
                            DataLimit=Args.viz_limit, Offset=[-1, 0, 0])
    PredViz = ODFDatasetLiveVisualizer(coord_type='direction', rays=Coords[VizIdx],
                            intersects=PredIntersects[VizIdx], depths=PredDepths[VizIdx],
                            DataLimit=Args.viz_limit, Offset=[1, 0, 0])
    CompareViz = Easel([GTViz, PredViz], sys.argv[1:])
    CompareViz.show()
    sys.exit(app.exec_())
