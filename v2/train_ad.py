import torch
from beacon import utils as butils
from beacon.supernet import SuperLoss
import sys, os
import argparse
import multiprocessing as mp


FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from pc_sampler import PC_SAMPLER_RADIUS
from single_losses import SingleDepthBCELoss, ADPredLoss, ADRegLoss, ADCombinedLoss

from single_models import LF4DSingleAutoDecoder
from pc_odf_dataset import PCODFDatasetLoader as PCDL

Parser = argparse.ArgumentParser(description='Training code for NeuralODF autodecoder.')
Parser.add_argument('--coord-type', help='Type of coordinates to use, valid options are points | direction | pluecker.', choices=['points', 'direction', 'pluecker'], default='direction')
Parser.add_argument('--rays-per-shape', help='Number of samples to use during testing.', default=1000, type=int)
Parser.add_argument('--val-rays-per-shape', help='Number of ray samples per object shape for validation.', default=10, type=int)
Parser.add_argument('--no-val', help='Choose to not perform validation during training.', action='store_true', required=False)
Parser.set_defaults(no_val=False)  # True for DEBUG only todo
Parser.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true', required=False)
Parser.set_defaults(force_test_on_train=False)
Parser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)
Parser.add_argument('--no-posenc', help='Choose not to use positional encoding.', action='store_true', required=False)
Parser.set_defaults(no_posenc=True) # DEBUG. todo: fix this
Parser.add_argument('--latent-size', type=int, default=256, help="The size of the latent vector for the autodecoder")
Parser.add_argument('--latent-stdev', type=float, default=0.001**2, help="The standard deviation of the zero mean gaussian used to initialize latent vectors")
Parser.add_argument('--lr-decoder', type=float, default=0.00001, help="The baseline learning rate for the decoder weights")
Parser.add_argument('--lr-latvecs', type=float, default=0.001, help="The learning rate for the latent vectors")


if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    butils.seedRandom(Args.seed)
    nCores = 0#mp.cpu_count()

    usePosEnc = not Args.no_posenc
    NeuralODF = LF4DSingleAutoDecoder(input_size=(120 if usePosEnc else 6), radius=PC_SAMPLER_RADIUS, coord_type=Args.coord_type, pos_enc=usePosEnc, latent_size=Args.latent_size)

    TrainDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # TrainDevice = "cpu"
    TrainData = PCDL(root=NeuralODF.Config.Args.input_dir, train=True, download=True, target_samples=Args.rays_per_shape, usePositionalEncoding=usePosEnc)
    if Args.force_test_on_train:
        print('[ WARN ]: VALIDATING ON TRAINING DATA.')
    ValData = PCDL(root=NeuralODF.Config.Args.input_dir, train=Args.force_test_on_train, download=True, target_samples=Args.val_rays_per_shape, usePositionalEncoding=usePosEnc)
    print('[ INFO ]: Training data has {} shapes and {} rays per sample.'.format(len(TrainData), Args.rays_per_shape))
    print('[ INFO ]: Validation data has {} shapes and {} rays per sample.'.format(len(ValData), Args.val_rays_per_shape))

     # Initialize embeddings for the training examples
    lat_vecs = torch.nn.Embedding(len(TrainData.LoadedOBJs), Args.latent_size, max_norm=8*Args.latent_stdev)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        # get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
        Args.latent_stdev
    )
    TrainData.addEmbeddings(lat_vecs)
    ValData.addEmbeddings(lat_vecs)

    #TODO: Figure out how to propagate embedding gradients with multiple workers
    TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=NeuralODF.Config.Args.batch_size, shuffle=True, num_workers=nCores, collate_fn=PCDL.collate_fn)
    if Args.no_val == False:
        ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=NeuralODF.Config.Args.batch_size, shuffle=True, num_workers=nCores, collate_fn=PCDL.collate_fn)
    else:
        print('[ WARN ]: Not validating during training. This should be used for debugging purposes only.')
        ValDataLoader = None

    

    # Create optimizer for both the network weights and the latent vectors
    optimizer_all = torch.optim.Adam(
        [
            {
                "params": NeuralODF.parameters(),
                "lr": Args.lr_decoder,
            },
            {
                "params": lat_vecs.parameters(),
                "lr": Args.lr_latvecs,
            },
        ]
    )
    # print(f"Loader Shape: {next(iter(TrainDataLoader))}")

    # loss = SuperLoss(Losses=[ADPredLoss, ADRegLoss], Weights=[1.0,1.0])
    loss = ADCombinedLoss

    NeuralODF.fit(TrainDataLoader, Objective=loss, TrainDevice=TrainDevice, ValDataLoader=ValDataLoader)