from cv2 import DescriptorMatcher
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
from losses import DepthLoss, IntersectionLoss, DepthFieldRegularizingLoss, ConstantRegularizingLoss, LatentPriorLoss
from odf_models import ODFSingleV3, ODFADV3, ODFSingleV3Constant, ODFSingleV3SH, ODFSingleV3ConstantSH
from depth_odf_dataset_5d import DepthODFDatasetLoader as DDL
import v3_utils
from infer import infer, infer_ad


def train(save_dir, name, model, embeddings, optimizer, train_loader, val_loader, loss_history, hyperparameters, device, scheduler):
    
    epochs = hyperparameters["epochs"]
    arch = hyperparameters["architecture"]
    previous_epochs = 0

    if "train" in loss_history:
        previous_epochs = len(loss_history["train"])
    else:
        all_losses = ["train", "val", "train_depth", "val_depth", "train_intersection", "val_intersection", "train_prior", "val_prior"]
        for loss in all_losses:
            loss_history[loss] = []


    # loss functions
    depth_loss_fn = DepthLoss()
    intersection_loss_fn = IntersectionLoss()
    prior_loss_fn = LatentPriorLoss()
    # dfr_loss_fn = DepthFieldRegularizingLoss()
    # cr_loss_fn = ConstantRegularizingLoss()
    

    for e in range(epochs):
        print(f"\n----------------------------- EPOCH: {e+previous_epochs+1}/{epochs+previous_epochs} -----------------------------")
        # Train on training batches
        print("Training...")
        all_train_losses = []
        all_train_depth_losses = []
        all_train_intersection_losses = []
        all_train_prior_losses = []
        model.train()
        for batch in tqdm(train_loader):
            data, targets = batch
            data = v3_utils.sendToDevice(data, device)
            targets = v3_utils.sendToDevice(targets, device)
            optimizer.zero_grad()

            # #########   LOSSES   #########
            output = model(data, embeddings)
            train_depth_loss = depth_loss_fn(output, targets)
            all_train_depth_losses.append(train_depth_loss.detach().cpu().numpy())
            # print(f"Train Depth Loss: {train_depth_loss.detach().cpu().numpy():.3f}")
            train_intersection_loss = intersection_loss_fn(output, targets)
            all_train_intersection_losses.append(train_intersection_loss.detach().cpu().numpy())
            train_prior_loss = prior_loss_fn(embeddings, data)
            all_train_prior_losses.append(train_prior_loss.detach().cpu().numpy())
            train_loss = train_depth_loss + train_intersection_loss + train_prior_loss
            all_train_losses.append(train_loss.detach().cpu().numpy())
            train_loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Training Loss: {np.mean(np.asarray(all_train_losses)):.5f}\n")


        # Validate on validation batches
        print("Validation...")
        all_val_losses = []
        all_val_depth_losses = []
        all_val_intersection_losses = []
        all_val_prior_losses = []
        model.eval()
        for batch in tqdm(val_loader):
            data, targets = batch
            data = v3_utils.sendToDevice(data, device)
            targets = v3_utils.sendToDevice(targets, device)

            # #########   LOSSES   #########
            output = model(data, embeddings)
            val_depth_loss = depth_loss_fn(output, targets)
            all_val_depth_losses.append(val_depth_loss.detach().cpu().numpy())
            val_intersection_loss = intersection_loss_fn(output, targets)
            all_val_intersection_losses.append(val_intersection_loss.detach().cpu().numpy())
            #val_loss = val_depth_loss + val_intersection_loss
            val_prior_loss = prior_loss_fn(embeddings, data)
            val_loss = val_depth_loss + val_intersection_loss + val_prior_loss
            all_val_prior_losses.append(val_prior_loss.detach().cpu().numpy())
            all_val_losses.append(val_loss.detach().cpu().numpy())
        print(f"Validation Loss: {np.mean(np.asarray(all_val_losses)):.5f}\n")

        # track manually for matplotlib visualization
        loss_history["train"].append(np.mean(np.asarray(all_train_losses)))
        loss_history["val"].append(np.mean(np.asarray(all_val_losses)))
        loss_history["train_depth"].append(np.mean(np.asarray(all_train_depth_losses)))
        loss_history["val_depth"].append(np.mean(np.asarray(all_val_depth_losses)))
        loss_history["train_intersection"].append(np.mean(np.asarray(all_train_intersection_losses)))
        loss_history["val_intersection"].append(np.mean(np.asarray(all_val_intersection_losses)))
        loss_history["train_prior"].append(np.mean(np.asarray(all_train_prior_losses)))
        loss_history["val_prior"].append(np.mean(np.asarray(all_val_prior_losses)))

        # track using weights and biases
        loss_dict = {"train_loss": np.mean(np.asarray(all_train_losses)),
                    "val_loss": np.mean(np.asarray(all_val_losses)),
                    "train_depth_loss": np.mean(np.asarray(all_train_depth_losses)),
                    "val_depth_loss": np.mean(np.asarray(all_val_depth_losses)),
                    "train_intersection_loss": np.mean(np.asarray(all_train_intersection_losses)),
                    "val_intersection_loss": np.mean(np.asarray(all_val_intersection_losses)),
                    "train_prior_loss": np.mean(np.asarray(all_train_prior_losses)),
                    "val_prior_loss": np.mean(np.asarray(all_val_prior_losses)),
                    }
        wandb.log(loss_dict)


        # save checkpoint
        if e%1000==0:
            v3_utils.checkpoint(model, save_dir, name, previous_epochs+e, optimizer, loss_history, latent_vectors=lat_vecs)
        #v3_utils.plotLosses(loss_history, save_dir, name)
        # wandb.watch(model)



import faulthandler; faulthandler.enable()

if __name__ == '__main__':
    Parser = v3_utils.BaselineParser
    Parser.add_argument('--epochs', help='Number of epochs to train for', type=int, default=10)
    Parser.add_argument('--batch-size', help='Choose mini-batch size.', required=False, default=100, type=int)
    Parser.add_argument('--learning-rate', help='Choose the learning rate.', default=0.0005, type=float)
    Parser.add_argument('--lr-latvecs', help='Choose the learning rate for the latent vectors.', default=0.0005, type=float)
    Parser.add_argument('--n-layers', help="Number of layers in the network backbone", default=7, type=int)
    Parser.add_argument('--rays-per-shape', help='Number of samples to use during testing.', default=1000, type=int)
    Parser.add_argument('--val-rays-per-shape', help='Number of ray samples per object shape for validation.', default=10, type=int)
    Parser.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true', required=False)
    Parser.add_argument('--additional-intersections', type=int, default=0, help="The number of addtional intersecting rays to generate per surface point")
    Parser.add_argument('--near-surface-threshold', type=float, default=-1., help="Sample an additional near-surface (within threshold) point for each intersecting ray. No sampling if negative.")
    Parser.add_argument('--tangent-rays-ratio', type=float, default=0., help="The proportion of sampled rays that should be roughly tangent to the object.")
    Parser.add_argument('--latent-size', type=int, default=256, help="Size of latent vectors in autodecoder")
    Parser.add_argument('--latent-stdev', type=float, default=0.001**2, help="The standard deviation of the zero mean gaussian used to initialize latent vectors")
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    wandb.init(project=Args.dataset, entity="neural-odf")
    #wandb.init(project="torus", entity="neural-odf")
    wandb.run.name = Args.expt_name
    wandb.run.save()

    v3_utils.seedRandom(Args.seed)

    nCores = 0#mp.cpu_count()

    if Args.arch == 'standard':
        NeuralODF = ODFADV3(input_size=(120 if Args.use_posenc else 6), latent_size=Args.latent_size, radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=Args.n_layers)

    # load instance mappings if they have already been dumped   
    instance_index_map = v3_utils.read_instance_index_map(Args.output_dir, Args.expt_name)

    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TrainData = DDL(root=Args.input_dir, name=Args.dataset, train=True, download=False, ad=True, target_samples=Args.rays_per_shape, usePositionalEncoding=Args.use_posenc, instance_index_map=instance_index_map)
    print(f"DATA SIZE: {len(TrainData)}")
    if Args.force_test_on_train:
        print('[ WARN ]: VALIDATING ON TRAINING DATA.')
    ValData = DDL(root=Args.input_dir, name=Args.dataset, train=Args.force_test_on_train, download=True, ad=True, target_samples=Args.val_rays_per_shape, usePositionalEncoding=Args.use_posenc, instance_index_map=instance_index_map)
    print('[ INFO ]: Training data has {} shapes and {} rays per sample.'.format(len(TrainData), Args.rays_per_shape))
    print('[ INFO ]: Validation data has {} shapes and {} rays per sample.'.format(len(ValData), Args.val_rays_per_shape))

    if instance_index_map is None:
        v3_utils.save_instance_index_map(Args.output_dir, Args.expt_name, TrainData.instance_index_map)   

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

     # Initialize embeddings for the training examples
    lat_vecs = torch.nn.Embedding(TrainData.n_instances, Args.latent_size)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        # get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
        Args.latent_stdev
    )


    loss_history = {}
    previous_epochs = 0
    if os.path.exists(os.path.join(Args.output_dir, Args.expt_name, "checkpoints")):
        checkpoint_dict = v3_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=False)
        NeuralODF.load_state_dict(checkpoint_dict['model_state_dict'])
        NeuralODF = NeuralODF.to(Device)
        lat_vecs.load_state_dict(checkpoint_dict["latent_vectors"])
        lat_vecs = lat_vecs.to(Device)
        # optimizer = torch.optim.Adam(NeuralODF.parameters(), lr=Args.learning_rate, weight_decay=1e-5)
        optimizer = torch.optim.Adam(
        [
            {
                "params": NeuralODF.parameters(),
                "lr": Args.learning_rate,
                "weight_decay": 1e-5,
            },
            {
                "params": lat_vecs.parameters(),
                "lr": Args.lr_latvecs,
            },
        ]
        )
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        loss_history = checkpoint_dict['loss_history']
        # TODO: load scheduler
        scheduler = StepLR(optimizer, step_size=2500, gamma=0.25)
    else:
        NeuralODF = NeuralODF.to(Device)
        lat_vecs = lat_vecs.to(Device)
        # optimizer = torch.optim.Adam(NeuralODF.parameters(), lr=Args.learning_rate, weight_decay=1e-5)
        optimizer = torch.optim.Adam(
        [
            {
                "params": NeuralODF.parameters(),
                "lr": Args.learning_rate,
                "weight_decay": 1e-5,
            },
            {
                "params": lat_vecs.parameters(),
                "lr": Args.lr_latvecs,
            },
        ]
        )
        scheduler = StepLR(optimizer, step_size=2500, gamma=0.25)


    train(Args.output_dir, Args.expt_name, NeuralODF, lat_vecs, optimizer, TrainDataLoader, ValDataLoader, loss_history, hyperparameters, Device, scheduler)


    # Now load the best checkpoint for evaluation
    checkpoint_dict = v3_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=True)
    NeuralODF.load_state_dict(checkpoint_dict['model_state_dict'])
    NeuralODF.to(Device)

    losses, depth_error, precision, recall, accuracy, f1 = infer_ad(Args.expt_name, NeuralODF, lat_vecs, ValDataLoader, hyperparameters, Device)


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
