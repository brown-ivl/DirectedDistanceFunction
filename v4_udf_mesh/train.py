import torch
import sys, os
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import wandb
from torch.optim.lr_scheduler import StepLR
import trimesh


FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from data import DepthData
import odf_utils
import sampling
from depth_sampler_5d import DEPTH_SAMPLER_RADIUS
from losses import DepthLoss, IntersectionLoss, DepthFieldRegularizingLoss, ConstantRegularizingLoss, MultiViewRayLoss
from odf_models import ODFSingleV3, ODFSingleV3Constant, ODFSingleV3SH, ODFSingleV3ConstantSH
from depth_odf_dataset_5d import DepthODFDatasetLoader as DDL
import v3_utils
from infer import infer


def train(save_dir, name, model, optimizer, train_loader, val_loader, loss_history, hyperparameters, device, scheduler):
    
    epochs = hyperparameters["epochs"]
    arch = hyperparameters["architecture"]
    previous_epochs = 0

    if "train" in loss_history:
        previous_epochs = len(loss_history["train"])
    else:
        all_losses = ["train", "val", "train_depth", "val_depth", "train_intersection", "val_intersection", "ssl"]
        if "constant" in arch:
            all_losses += ["train_dfr", "val_dfr", "train_cr", "val_cr"]
        for loss in all_losses:
            loss_history[loss] = []


    # loss functions
    depth_loss_fn = DepthLoss()
    intersection_loss_fn = IntersectionLoss()
    dfr_loss_fn = DepthFieldRegularizingLoss()
    cr_loss_fn = ConstantRegularizingLoss()
    ssl_loss_fn = MultiViewRayLoss()
    maxShift = [(i+1)/10 for i in range(13)]+[1.3]*10
    cp = [0.5 for i in range(30)]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    for e in range(epochs):
        print(f"\n----------------------------- EPOCH: {e+previous_epochs+1}/{epochs+previous_epochs} -----------------------------")
        # Train on training batches
        print("Training...")
        all_train_losses = []
        all_train_depth_losses = []
        all_train_intersection_losses = []
        all_train_dfr_losses = []
        all_train_cr_losses = []
        all_train_ssl_losses = []
        model.train()
        for batch in tqdm(train_loader):
            data, targets = batch
            data = v3_utils.sendToDevice(data, device)
            targets = v3_utils.sendToDevice(targets, device)
            optimizer.zero_grad()

            # #########   LOSSES   #########
            output = model(data)
            train_depth_loss = depth_loss_fn(output, targets, cp[(e+previous_epochs)//20000])
            all_train_depth_losses.append(train_depth_loss.detach().cpu().numpy())
            train_intersection_loss = intersection_loss_fn(output, targets)
            all_train_intersection_losses.append(train_intersection_loss.detach().cpu().numpy())
            train_ssl_loss = ssl_loss_fn(model, data, targets, maxShift[(e+previous_epochs)//10000])
            all_train_ssl_losses.append(train_ssl_loss.detach().cpu().numpy())
            train_loss = train_depth_loss + train_intersection_loss
            #if e>0:
            #    train_loss += train_ssl_loss
            if 'constant' in arch:
                train_dfr_loss = dfr_loss_fn(model, data)
                all_train_dfr_losses.append(train_dfr_loss.detach().cpu().numpy())
                train_cr_loss = cr_loss_fn(model, data)
                all_train_cr_losses.append(train_cr_loss.detach().cpu().numpy())
                train_loss += train_dfr_loss + train_cr_loss
                #train_loss += train_cr_loss
            all_train_losses.append(train_loss.detach().cpu().numpy())
            train_loss.backward()
            optimizer.step()
        if e>0:
            scheduler.step()
        print(f"Training Loss: {np.mean(np.asarray(all_train_losses)):.5f}\n")


        # Validate on validation batches
        print("Validation...")
        all_val_losses = []
        all_val_depth_losses = []
        all_val_intersection_losses = []
        all_val_dfr_losses = []
        all_val_cr_losses = []
        model.eval()
        for batch in tqdm(val_loader):
            data, targets = batch
            data = v3_utils.sendToDevice(data, device)
            targets = v3_utils.sendToDevice(targets, device)

            # #########   LOSSES   #########
            output = model(data)
            val_depth_loss = depth_loss_fn(output, targets, cp[e//20000])
            all_val_depth_losses.append(val_depth_loss.detach().cpu().numpy())
            val_intersection_loss = intersection_loss_fn(output, targets)
            all_val_intersection_losses.append(val_intersection_loss.detach().cpu().numpy())
            val_loss = val_depth_loss + val_intersection_loss
            if "constant" in arch:
                val_dfr_loss = dfr_loss_fn(model, data)
                all_val_dfr_losses.append(val_dfr_loss.detach().cpu().numpy())
                val_cr_loss = cr_loss_fn(model, data)
                all_val_cr_losses.append(val_cr_loss.detach().cpu().numpy())
                val_loss += val_dfr_loss + val_cr_loss
            all_val_losses.append(val_loss.detach().cpu().numpy())
        print(f"Validation Loss: {np.mean(np.asarray(all_val_losses)):.5f}\n")

        # track manually for matplotlib visualization
        loss_history["train"].append(np.mean(np.asarray(all_train_losses)))
        loss_history["val"].append(np.mean(np.asarray(all_val_losses)))
        loss_history["train_depth"].append(np.mean(np.asarray(all_train_depth_losses)))
        loss_history["val_depth"].append(np.mean(np.asarray(all_val_depth_losses)))
        loss_history["train_intersection"].append(np.mean(np.asarray(all_train_intersection_losses)))
        loss_history["ssl"].append(np.mean(np.asarray(all_train_ssl_losses)))
        loss_history["val_intersection"].append(np.mean(np.asarray(all_val_intersection_losses)))
        if "constant" in arch:
            loss_history["train_dfr"].append(np.mean(np.asarray(all_train_dfr_losses)))
            loss_history["val_dfr"].append(np.mean(np.asarray(all_val_dfr_losses)))
            loss_history["train_cr"].append(np.mean(np.asarray(all_train_cr_losses)))
            loss_history["val_cr"].append(np.mean(np.asarray(all_val_cr_losses)))

        # track using weights and biases
        loss_dict = {"train_loss": np.mean(np.asarray(all_train_losses)),
                    "val_loss": np.mean(np.asarray(all_val_losses)),
                    "train_depth_loss": np.mean(np.asarray(all_train_depth_losses)),
                    "val_depth_loss": np.mean(np.asarray(all_val_depth_losses)),
                    "train_intersection_loss": np.mean(np.asarray(all_train_intersection_losses)),
                    "ssl_loss": np.mean(np.asarray(all_train_ssl_losses)),
                    "val_intersection_loss": np.mean(np.asarray(all_val_intersection_losses)),
                    }
        if "constant" in arch:
            loss_dict.update({"train_dfr_loss": np.mean(np.asarray(all_train_dfr_losses)),
                    "val_dfr_loss": np.mean(np.asarray(all_val_dfr_losses)),
                    "train_cr_loss": np.mean(np.asarray(all_train_cr_losses)),
                    "val_cr_loss": np.mean(np.asarray(all_val_cr_losses))
            })
        wandb.log(loss_dict)

        if e%100==0:
            # save checkpoint
            v3_utils.checkpoint(model, save_dir, name, previous_epochs+e, optimizer, loss_history)
        #v3_utils.plotLosses(loss_history, save_dir, name)
        # wandb.watch(model)



import faulthandler; faulthandler.enable()

if __name__ == '__main__':
    Parser = v3_utils.BaselineParser
    Parser.add_argument('--epochs', help='Number of epochs to train for', type=int, default=10)
    Parser.add_argument('--batch-size', help='Choose mini-batch size.', required=False, default=8, type=int)
    Parser.add_argument('--learning-rate', help='Choose the learning rate.', default=0.001, type=float)
    Parser.add_argument('--n-layers', help="Number of layers in the network backbone", default=7, type=int)
    Parser.add_argument('--rays-per-shape', help='Number of samples to use during testing.', default=100, type=int)
    Parser.add_argument('--val-rays-per-shape', help='Number of ray samples per object shape for validation.', default=10, type=int)
    Parser.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true', required=False)
    Parser.add_argument('--additional-intersections', type=int, default=0, help="The number of addtional intersecting rays to generate per surface point")
    Parser.add_argument('--near-surface-threshold', type=float, default=-1., help="Sample an additional near-surface (within threshold) point for each intersecting ray. No sampling if negative.")
    Parser.add_argument('--tangent-rays-ratio', type=float, default=0., help="The proportion of sampled rays that should be roughly tangent to the object.")
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    #wandb.init(project=Args.dataset, entity="neural-odf")
    wandb.init(project="shirt", entity="neural-odf")
    wandb.run.name = Args.expt_name
    wandb.run.save()

    v3_utils.seedRandom(Args.seed)

    nCores = 4#mp.cpu_count()

    if Args.arch == 'standard':
        NeuralODF = ODFSingleV3(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers)
    elif Args.arch == 'constant':
        NeuralODF = ODFSingleV3Constant(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers)
    elif Args.arch == 'SH':
        NeuralODF = ODFSingleV3SH(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers, degrees=Args.degrees)
        print('[ INFO ]: Degrees {}'.format(Args.degrees))
    elif Args.arch == 'SH_constant':
        NeuralODF = ODFSingleV3ConstantSH(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers, degrees=Args.degrees)
        print('[ INFO ]: Degrees {}'.format(Args.degrees))


    Device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    #mesh = trimesh.load(os.path.join(Args.input_dir, Args.dataset+'.obj'))
    verts, faces, _ = v3_utils.load_object(Args.dataset, Args.input_dir)
    #faces = mesh.faces
    #verts = mesh.vertices
    #verts = odf_utils.mesh_normalize(verts)
    vert_noise = 0.01
    tan_noise = 0.01
    radius_ = 1.25
    sampling_methods = [sampling.sample_uniform_ray_space, 
                        sampling.sampling_preset_noise(sampling.sample_vertex_noise, vert_noise),
                        sampling.sampling_preset_noise(sampling.sample_vertex_all_directions, vert_noise),
                        sampling.sampling_preset_noise(sampling.sample_vertex_tangential, tan_noise)]
    sampling_frequency = [0.5, 0.2, 0.2, 0.1]
    test_sampling_frequency = [1., 0., 0., 0.]

    TrainData = DepthData(faces,verts,radius_,sampling_methods,sampling_frequency,size=Args.rays_per_shape*nCores*Args.batch_size)
    ValData = DepthData(faces,verts,radius_,sampling_methods,test_sampling_frequency,size=Args.val_rays_per_shape*nCores*Args.batch_size)


    #TrainData = DDL(root=Args.input_dir, name=Args.dataset, train=True, target_samples=Args.rays_per_shape, usePositionalEncoding=Args.use_posenc, sampling_frequency=[1.0, 0.0, 0.0, 0.0])
    #print(f"DATA SIZE: {len(TrainData)}")
    if Args.force_test_on_train:
        print('[ WARN ]: VALIDATING ON TRAINING DATA.')
    #ValData = DDL(root=Args.input_dir, name=Args.dataset, train=Args.force_test_on_train, target_samples=Args.val_rays_per_shape, usePositionalEncoding=Args.use_posenc, sampling_frequency=[1.0, 0.0, 0.0, 0.0])
    #print('[ INFO ]: Training data has {} shapes and {} rays per sample.'.format(len(TrainData), Args.rays_per_shape))
    #print('[ INFO ]: Validation data has {} shapes and {} rays per sample.'.format(len(ValData), Args.val_rays_per_shape))

    TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=Args.batch_size*Args.rays_per_shape, shuffle=True, num_workers=nCores, collate_fn=DepthData.collate_fn)
    ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=Args.batch_size*Args.val_rays_per_shape, shuffle=True, num_workers=nCores, collate_fn=DepthData.collate_fn)

    hyperparameters = {
        "learning_rate": Args.learning_rate,
        "epochs": Args.epochs,
        "batch_size": Args.batch_size,
        "architecture": Args.arch,
        "dataset": Args.dataset
    }

    wandb.config = hyperparameters

    loss_history = {}
    previous_epochs = 0
    if os.path.exists(os.path.join(Args.output_dir, Args.expt_name, "checkpoints")):
        checkpoint_dict = v3_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=False)
        NeuralODF.load_state_dict(checkpoint_dict['model_state_dict'])
        NeuralODF = NeuralODF.to(Device)
        optimizer = torch.optim.Adam(NeuralODF.parameters(), lr=Args.learning_rate, weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        loss_history = checkpoint_dict['loss_history']
        loss_history["ssl"] = []
        # TODO: load scheduler
        scheduler = StepLR(optimizer, step_size=10000, gamma=0.25)
    else:
        NeuralODF = NeuralODF.to(Device)
        optimizer = torch.optim.Adam(NeuralODF.parameters(), lr=Args.learning_rate, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=10000, gamma=0.25)


    train(Args.output_dir, Args.expt_name, NeuralODF, optimizer, TrainDataLoader, ValDataLoader, loss_history, hyperparameters, Device, scheduler)


    # Now load the best checkpoint for evaluation
    checkpoint_dict = v3_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=True)
    NeuralODF.load_state_dict(checkpoint_dict['model_state_dict'])
    NeuralODF.to(Device)

    losses, depth_error, precision, recall, accuracy, f1 = infer(Args.expt_name, NeuralODF, ValDataLoader, hyperparameters, Device)


    all_losses = ["train", "val", "train_depth", "val_depth", "train_intersection", "val_intersection", "ssl"]
    if Args.arch == "constant":
        all_losses += ["train_dfr", "val_dfr", "train_cr", "val_cr"]

    for loss in all_losses:
        del wandb.run.summary[loss+"_loss"]

    for loss in losses:
        wandb.run.summary[loss+"_loss"] = losses[loss]
    wandb.run.summary["depth_error"] = depth_error
    wandb.run.summary["mask precision"] = precision
    wandb.run.summary["mask recall"] = recall
    wandb.run.summary["mask f1"] = f1
    wandb.run.summary["mask accuracy"] = accuracy
