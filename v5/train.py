import torch
import sys, os
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import wandb
from torch.optim.lr_scheduler import StepLR


FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from depth_sampler_5d import DEPTH_SAMPLER_RADIUS
from losses import DepthLoss, IntersectionLoss, DepthFieldRegularizingLoss, ConstantRegularizingLoss, MaskLoss, ODFLoss, MaskLossV2
from odf_models import ODFSingleV3, ODFSingleV3Constant, ODFSingleV3SH, ODFSingleV3ConstantSH, ODFV5, IntersectionMask3D, IntersectionMask3DV2, IntersectionMask3DMLP
from depth_odf_dataset_5d import DepthODFDatasetLoader as DDL
from occnet_loader import OccNetLoader as ONL
import v5_utils
from infer import infer

torch.autograd.set_detect_anomaly(True)


def train(save_dir, name, odf_model, mask_model, optimizer, train_loader, val_loader, loss_history, hyperparameters, device, scheduler):
    
    epochs = hyperparameters["epochs"]
    arch = hyperparameters["architecture"]
    previous_epochs = 0

    # odf_loss = ODFLoss()
    # mask_loss = MaskLossV2()

    depth_loss = DepthLoss()
    mask_loss = MaskLoss()

    if "train" in loss_history:
        previous_epochs = len(loss_history["train"])
    else:
        all_losses = ["train", "val", "train_depth", "val_depth", "train_mask", "val_mask"]
        for loss in all_losses:
            loss_history[loss] = []


    # loss functions
    

    for e in range(epochs):
        print(f"\n----------------------------- EPOCH: {e+previous_epochs+1}/{epochs+previous_epochs} -----------------------------")
        # Train on training batches
        print("Training...")
        all_train_losses = []
        all_train_depth_losses = []
        all_train_mask_losses = []
        odf_model.train()
        mask_model.train()
        for batch in tqdm(train_loader):
            coords, targets = batch

            # odf_coords = v5_utils.sendToDevice(odf_coords, device)
            # odf_targets = v5_utils.sendToDevice(odf_targets, device)
            # mask_coords = v5_utils.sendToDevice(mask_coords, device)
            # mask_targets = v5_utils.sendToDevice(mask_targets, device)
            coords = v5_utils.sendToDevice(coords, device)
            targets = v5_utils.sendToDevice(targets, device)
            optimizer.zero_grad()

            # #########   LOSSES   #########
            odf_output = odf_model(coords)

            projected_points = [coords[i][:,:3] + coords[i][:,3:]*torch.hstack([odf_output[i],]*3) for i in range(len(coords))]
            mask_output = mask_model(projected_points)
            # print(len(mask_output))
            # print([torch.mean(mask_output[i]) for i in range(len(mask_output))])
            # print(f"Mean Mask: {sum([torch.mean(mask_output[i]) for i in range(len(mask_output))])/len(mask_output)}")

            batch_depth_loss = depth_loss(odf_output, targets)
            batch_mask_loss = mask_loss(mask_output, targets)

            all_train_depth_losses.append(batch_depth_loss.detach().cpu().numpy())
            all_train_mask_losses.append(batch_mask_loss.detach().cpu().numpy())
            train_loss = batch_depth_loss + batch_mask_loss
            # train_loss = batch_mask_loss
            all_train_losses.append(train_loss.detach().cpu().numpy())
            train_loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Training Loss: {np.mean(np.asarray(all_train_losses)):.5f}\n")


        # Validate on validation batches
        print("Validation...")
        all_val_losses = []
        all_val_depth_losses = []
        all_val_mask_losses = []
        odf_model.eval()
        mask_model.eval()
        for batch in tqdm(val_loader):
            coords, targets = batch

            # odf_coords = v5_utils.sendToDevice(odf_coords, device)
            # odf_targets = v5_utils.sendToDevice(odf_targets, device)
            # mask_coords = v5_utils.sendToDevice(mask_coords, device)
            # mask_targets = v5_utils.sendToDevice(mask_targets, device)
            coords = v5_utils.sendToDevice(coords, device)
            targets = v5_utils.sendToDevice(targets, device)

            # #########   LOSSES   #########
            odf_output = odf_model(coords)
            projected_points = [coords[i][:,:3] + coords[i][:,3:]*torch.hstack([odf_output[i],]*3) for i in range(len(coords))]
            mask_output = mask_model(projected_points)

            batch_depth_loss = depth_loss(odf_output, targets)
            batch_mask_loss = mask_loss(mask_output, targets)


            all_val_depth_losses.append(batch_depth_loss.detach().cpu().numpy())
            all_val_mask_losses.append(batch_mask_loss.detach().cpu().numpy())
            val_loss = batch_depth_loss + batch_mask_loss
            # val_loss = batch_mask_loss
            all_val_losses.append(val_loss.detach().cpu().numpy())
        print(f"Validation Loss: {np.mean(np.asarray(all_val_losses)):.5f}\n")

        # track manually for matplotlib visualization
        loss_history["train"].append(np.mean(np.asarray(all_train_losses)))
        loss_history["val"].append(np.mean(np.asarray(all_val_losses)))
        loss_history["train_depth"].append(np.mean(np.asarray(all_train_depth_losses)))
        loss_history["val_depth"].append(np.mean(np.asarray(all_val_depth_losses)))
        loss_history["train_mask"].append(np.mean(np.asarray(all_train_mask_losses)))
        loss_history["val_mask"].append(np.mean(np.asarray(all_val_mask_losses)))

        # track using weights and biases
        loss_dict = {"train_loss": np.mean(np.asarray(all_train_losses)),
                    "val_loss": np.mean(np.asarray(all_val_losses)),
                    "train_depth_loss": np.mean(np.asarray(all_train_depth_losses)),
                    "val_depth_loss": np.mean(np.asarray(all_val_depth_losses)),
                    "train_mask_loss": np.mean(np.asarray(all_train_mask_losses)),
                    "val_mask_loss": np.mean(np.asarray(all_val_mask_losses)),
                    }
        wandb.log(loss_dict)


        # save checkpoint
        v5_utils.checkpoint(odf_model, mask_model, save_dir, name, previous_epochs+e, optimizer, scheduler, loss_history)
        v5_utils.plotLosses(loss_history, save_dir, name)
        # wandb.watch(model)



import faulthandler; faulthandler.enable()

if __name__ == '__main__':
    Parser = v5_utils.BaselineParser
    Parser.add_argument('--epochs', help='Number of epochs to train for', type=int, default=10)
    Parser.add_argument('--batch-size', help='Choose mini-batch size.', required=False, default=16, type=int)
    Parser.add_argument('--learning-rate', help='Choose the learning rate.', default=0.001, type=float)
    Parser.add_argument('--n-layers', help="Number of layers in the network backbone", default=7, type=int)
    Parser.add_argument('--rays-per-shape', help='Number of samples to use during testing.', default=1000, type=int)
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
    wandb.init(project="bunny", entity="neural-odf")
    wandb.run.name = Args.expt_name
    wandb.run.save()

    v5_utils.seedRandom(Args.seed)

    nCores = 0#mp.cpu_count()

    if Args.arch == 'standard':
        NeuralODF = ODFV5(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers)
        # Mask3D = IntersectionMask3DV2(dim=3, hidden_size=256)
        Mask3D = IntersectionMask3DMLP(n_layers=3, pos_enc=False)
    # elif Args.arch == 'constant':
    #     NeuralODF = ODFSingleV3Constant(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers)
    # elif Args.arch == 'SH':
    #     NeuralODF = ODFSingleV3SH(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers, degrees=Args.degrees)
    #     print('[ INFO ]: Degrees {}'.format(Args.degrees))
    # elif Args.arch == 'SH_constant':
    #     NeuralODF = ODFSingleV3ConstantSH(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers, degrees=Args.degrees)
    #     print('[ INFO ]: Degrees {}'.format(Args.degrees))


    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TrainData = DDL(root=Args.input_dir, name=Args.dataset, train=True, download=False, target_samples=Args.rays_per_shape, usePositionalEncoding=Args.use_posenc)
    # TrainData = ONL(root=Args.input_dir, name=Args.dataset, train=True, download=False, target_samples=Args.rays_per_shape)
    print(f"DATA SIZE: {len(TrainData)}")
    if Args.force_test_on_train:
        print('[ WARN ]: VALIDATING ON TRAINING DATA.')
    ValData = DDL(root=Args.input_dir, name=Args.dataset, train=Args.force_test_on_train, download=True, target_samples=Args.val_rays_per_shape, usePositionalEncoding=Args.use_posenc)
    # ValData = ONL(root=Args.input_dir, name=Args.dataset, train=Args.force_test_on_train, download=False, target_samples=Args.rays_per_shape)

    print('[ INFO ]: Training data has {} shapes and {} rays per sample.'.format(len(TrainData), Args.rays_per_shape))
    print('[ INFO ]: Validation data has {} shapes and {} rays per sample.'.format(len(ValData), Args.val_rays_per_shape))

    TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=Args.batch_size, shuffle=True, num_workers=nCores, collate_fn=DDL.collate_fn)
    ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=Args.batch_size, shuffle=True, num_workers=nCores, collate_fn=DDL.collate_fn)

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
        checkpoint_dict = v5_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=False)
        NeuralODF.load_state_dict(checkpoint_dict['odf_model_state_dict'])
        NeuralODF = NeuralODF.to(Device)
        Mask3D.load_state_dict(checkpoint_dict['mask_model_state_dict'])
        Mask3D = Mask3D.to(Device)
        optimizer = torch.optim.Adam(list(NeuralODF.parameters()) + list(Mask3D.parameters()), lr=Args.learning_rate, weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        loss_history = checkpoint_dict['loss_history']
        # TODO: load scheduler
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.25)
        scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
    else:
        NeuralODF = NeuralODF.to(Device)
        Mask3D = Mask3D.to(Device)
        optimizer = torch.optim.Adam(list(NeuralODF.parameters()) + list(Mask3D.parameters()), lr=Args.learning_rate, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.25)


    train(Args.output_dir, Args.expt_name, NeuralODF, Mask3D, optimizer, TrainDataLoader, ValDataLoader, loss_history, hyperparameters, Device, scheduler)


    # Now load the best checkpoint for evaluation
    checkpoint_dict = v5_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=True)
    NeuralODF.load_state_dict(checkpoint_dict['odf_model_state_dict'])
    NeuralODF.to(Device)
    Mask3D.load_state_dict(checkpoint_dict['mask_model_state_dict'])
    Mask3D.to(Device)

    # losses, depth_error, precision, recall, accuracy, f1 = infer(Args.expt_name, NeuralODF, ValDataLoader, hyperparameters, Device)


    # all_losses = ["train", "val", "train_depth", "val_depth", "train_intersection", "val_intersection"]
    # if Args.arch == "constant":
    #     all_losses += ["train_dfr", "val_dfr", "train_cr", "val_cr"]

    # for loss in all_losses:
    #     del wandb.run.summary[loss+"_loss"]

    # for loss in losses:
    #     wandb.run.summary[loss+"_loss"] = losses[loss]
    # wandb.run.summary["depth_error"] = depth_error
    # wandb.run.summary["mask precision"] = precision
    # wandb.run.summary["mask recall"] = recall
    # wandb.run.summary["mask f1"] = f1
    # wandb.run.summary["mask accuracy"] = accuracy
