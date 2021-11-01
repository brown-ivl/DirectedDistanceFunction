import torch
from beacon import utils as butils
import sys, os
import argparse

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from pc_sampler import PC_SAMPLER_RADIUS
from single_losses import SingleDepthBCELoss
from single_models import LF4DSingle
from pc_odf_dataset import PCODFDatasetLoader as PCDL
from odf_dataset import ODFDatasetLoader as ODL

Parser = argparse.ArgumentParser(description='Training code for NeuralODFs.')
Parser.add_argument('--arch', help='Architecture to use.', choices=['standard'], default='standard')
Parser.add_argument('--loader', help='Which data loader to use.', choices=['mesh', 'pc'], default='pc')
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--train-rays-per-shape', help='Number of samples to use during testing.', default=1000, type=int)
Parser.add_argument('--val-rays-per-shape', help='Number of ray samples per object shape for validation.', default=10, type=int)
Parser.add_argument('--no-val', help='Choose to not perform validation during training.', action='store_true', required=False)
Parser.set_defaults(no_val=False)  # True for DEBUG only todo
Parser.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true', required=False)
Parser.set_defaults(force_test_on_train=False)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.set_defaults(no_posenc=True) # DEBUG. todo: fix this

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    butils.seedRandom(Args.seed)

    usePosEnc = not Args.no_posenc
    if Args.arch == 'standard':
        NeuralODF = LF4DSingle(input_size=(120 if usePosEnc else 6), radius=PC_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc)

    TrainDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if Args.loader == 'mesh':
        TrainData = ODL(root=NeuralODF.Config.Args.input_dir, train=True, download=True, n_samples=(Args.train_rays_per_shape), usePositionalEncoding=usePosEnc)
    elif Args.loader == 'pc':
        TrainData = PCDL(root=NeuralODF.Config.Args.input_dir, train=True, download=True, target_samples=Args.train_rays_per_shape, usePositionalEncoding=usePosEnc)

    if Args.force_test_on_train:
        print('[ WARN ]: VALIDATING ON TRAINING DATA.')
    if Args.loader == 'mesh':
        ValData = ODL(root=NeuralODF.Config.Args.input_dir, train=Args.force_test_on_train, download=True, n_samples=(Args.val_rays_per_shape), usePositionalEncoding=usePosEnc)
    elif Args.loader == 'pc':
        ValData = PCDL(root=NeuralODF.Config.Args.input_dir, train=Args.force_test_on_train, download=True,
                       target_samples=Args.val_rays_per_shape, usePositionalEncoding=usePosEnc)
    print('[ INFO ]: Training data has {} shapes and {} rays per sample.'.format(len(TrainData), Args.train_rays_per_shape))
    print('[ INFO ]: Validation data has {} shapes and {} rays per sample.'.format(len(ValData), Args.val_rays_per_shape))

    TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=NeuralODF.Config.Args.batch_size, shuffle=False, num_workers=0) # DEBUG, TODO: More workers not working
    if Args.no_val == False:
        ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=NeuralODF.Config.Args.batch_size, shuffle=False, num_workers=0)
    else:
        print('[ WARN ]: Not validating during training. This should be used for debugging purposes only.')
        ValDataLoader = None

    NeuralODF.fit(TrainDataLoader, Objective=SingleDepthBCELoss(), TrainDevice=TrainDevice, ValDataLoader=ValDataLoader)