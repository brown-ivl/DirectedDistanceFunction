import torch
import sys, os
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import wandb


FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from depth_sampler_5d import DEPTH_SAMPLER_RADIUS
from losses import DepthLoss, IntersectionLoss, DepthFieldRegularizingLoss, ConstantRegularizingLoss
from odf_models import ODFSingleV3, ODFSingleV3Constant, ODFSingleV3SH, ODFSingleV3ConstantSH
from depth_odf_dataset_5d import DepthODFDatasetLoader as DDL
import v3_utils
from infer import infer

def train(save_dir, name, model, optimizer, train_loader, val_loader, loss_history, hyperparameters, device):
    
    epochs = hyperparameters["epochs"]
    arch = hyperparameters["architecture"]
    previous_epochs = 0

    if "train" in loss_history:
        previous_epochs = len(loss_history["train"])
    else:
        all_losses = ["train", "val", "train_depth", "val_depth", "train_intersection", "val_intersection"]
        if arch == "constant":
            all_losses += ["train_dfr", "val_dfr", "train_cr", "val_cr"]
        for loss in all_losses:
            loss_history[loss] = []


    # loss functions
    depth_loss_fn = DepthLoss()
    intersection_loss_fn = IntersectionLoss()
    dfr_loss_fn = DepthFieldRegularizingLoss()
    cr_loss_fn = ConstantRegularizingLoss()
    

    for e in range(epochs):
        print(f"\n----------------------------- EPOCH: {e+previous_epochs+1}/{epochs+previous_epochs} -----------------------------")
        # Train on training batches
        print("Training...")
        all_train_losses = []
        all_train_depth_losses = []
        all_train_intersection_losses = []
        all_train_dfr_losses = []
        all_train_cr_losses = []
        model.train()
        for batch in tqdm(train_loader):
            data, targets = batch
            data = v3_utils.sendToDevice(data, device)
            targets = v3_utils.sendToDevice(targets, device)
            optimizer.zero_grad()

            # #########   LOSSES   #########
            output = model(data)
            train_depth_loss = depth_loss_fn(output, targets)
            all_train_depth_losses.append(train_depth_loss.detach().cpu().numpy())
            train_intersection_loss = intersection_loss_fn(output, targets)
            all_train_intersection_losses.append(train_intersection_loss.detach().cpu().numpy())
            train_loss = train_depth_loss + train_intersection_loss
            if arch == "constant":
                train_dfr_loss = dfr_loss_fn(model, data)
                all_train_dfr_losses.append(train_dfr_loss.detach().cpu().numpy())
                train_cr_loss = cr_loss_fn(model, data)
                all_train_cr_losses.append(train_cr_loss.detach().cpu().numpy())
                train_loss += train_dfr_loss + train_cr_loss
            all_train_losses.append(train_loss.detach().cpu().numpy())
            train_loss.backward()
            optimizer.step()
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
            val_depth_loss = depth_loss_fn(output, targets)
            all_val_depth_losses.append(val_depth_loss.detach().cpu().numpy())
            val_intersection_loss = intersection_loss_fn(output, targets)
            all_val_intersection_losses.append(val_intersection_loss.detach().cpu().numpy())
            val_loss = val_depth_loss + val_intersection_loss
            if arch == "constant":
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
        loss_history["val_intersection"].append(np.mean(np.asarray(all_val_intersection_losses)))
        if arch == "constant":
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
                    "val_intersection_loss": np.mean(np.asarray(all_val_intersection_losses)),
                    }
        if arch == "constant":
            loss_dict.update({"train_dfr_loss": np.mean(np.asarray(all_train_dfr_losses)),
                    "val_dfr_loss": np.mean(np.asarray(all_val_dfr_losses)),
                    "train_cr_loss": np.mean(np.asarray(all_train_cr_losses)),
                    "val_cr_loss": np.mean(np.asarray(all_val_cr_losses))
            })
        wandb.log(loss_dict)


        # save checkpoint
        v3_utils.checkpoint(model, save_dir, name, previous_epochs+e, optimizer, loss_history)
        v3_utils.plotLosses(loss_history, save_dir, name)
        # wandb.watch(model)



import faulthandler; faulthandler.enable()

if __name__ == '__main__':
    Parser = v3_utils.BaselineParser
    Parser.add_argument('--epochs', help='Number of epochs to train for', type=int, default=10)
    Parser.add_argument('--batch-size', help='Choose mini-batch size.', required=False, default=16, type=int)
    Parser.add_argument('--learning-rate', help='Choose the learning rate.', default=0.001, type=float)
    Parser.add_argument('--rays-per-shape', help='Number of samples to use during testing.', default=1000, type=int)
    Parser.add_argument('--val-rays-per-shape', help='Number of ray samples per object shape for validation.', default=10, type=int)
    Parser.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true', required=False)
    Parser.add_argument('--additional-intersections', type=int, help="The number of addtional intersecting rays to generate per surface point")
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    wandb.init(project=Args.dataset, entity="neural-odf")
    wandb.run.name = Args.expt_name
    wandb.run.save()

    v3_utils.seedRandom(Args.seed)

    nCores = 0#mp.cpu_count()

    if Args.arch == 'standard':
        NeuralODF = ODFSingleV3(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=10)
    elif Args.arch == 'constant':
        NeuralODF = ODFSingleV3Constant(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=10)
    elif Args.arch == 'SH':
        NeuralODF = ODFSingleV3SH(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=10, degrees=Args.degrees)
        print('[ INFO ]: Degrees {}'.format(Args.degrees))
    elif Args.arch == 'SH_constant':
        NeuralODF = ODFSingleV3ConstantSH(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=10, degrees=Args.degrees)
        print('[ INFO ]: Degrees {}'.format(Args.degrees))


    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TrainData = DDL(root=Args.input_dir, name=Args.dataset, train=True, download=False, target_samples=Args.rays_per_shape, usePositionalEncoding=Args.use_posenc, additional_intersections=Args.additional_intersections)
    print(f"DATA SIZE: {len(TrainData)}")
    if Args.force_test_on_train:
        print('[ WARN ]: VALIDATING ON TRAINING DATA.')
    ValData = DDL(root=Args.input_dir, name=Args.dataset, train=Args.force_test_on_train, download=True, target_samples=Args.val_rays_per_shape, usePositionalEncoding=Args.use_posenc)
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
        checkpoint_dict = v3_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=False)
        NeuralODF.load_state_dict(checkpoint_dict['model_state_dict'])
        NeuralODF = NeuralODF.to(Device)
        optimizer = torch.optim.Adam(NeuralODF.parameters(), lr=Args.learning_rate, weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        loss_history = checkpoint_dict['loss_history']
    else:
        NeuralODF = NeuralODF.to(Device)
        optimizer = torch.optim.Adam(NeuralODF.parameters(), lr=Args.learning_rate, weight_decay=1e-5)

    train(Args.output_dir, Args.expt_name, NeuralODF, optimizer, TrainDataLoader, ValDataLoader, loss_history, hyperparameters, Device)


    # Now load the best checkpoint for evaluation
    checkpoint_dict = v3_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=True)
    NeuralODF.load_state_dict(checkpoint_dict['model_state_dict'])
    NeuralODF.to(Device)

    losses, depth_error, precision, recall, accuracy, f1 = infer(Args.expt_name, NeuralODF, ValDataLoader, hyperparameters, Device)


    all_losses = ["train", "val", "train_depth", "val_depth", "train_intersection", "val_intersection"]
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
