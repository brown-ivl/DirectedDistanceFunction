import torch
import argparse
import math
import sys, os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from sklearn.metrics import confusion_matrix


FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, 'loaders'))
sys.path.append(os.path.join(FileDirPath, 'losses'))
sys.path.append(os.path.join(FileDirPath, 'models'))

from depth_sampler_5d import DEPTH_SAMPLER_RADIUS
from losses import DepthLoss, IntersectionLoss, DepthFieldRegularizingLoss, ConstantRegularizingLoss
from odf_models import ODFSingleV3, ODFSingleV3Constant
from depth_odf_dataset_5d import DepthODFDatasetLoader as DDL
import v3_utils

def praf1(output, target):
    '''
    Returns the Precision, Recall, Accuracy, and F1 Score for the intersection mask
    '''
    assert isinstance(output, list)
    B = len(target)

    GTMasks = torch.cat([x[0] for x in target]).detach().cpu().numpy()
    MaskPredictions = torch.cat([x[0] for x in output]).to(torch.bool).detach().cpu().numpy()

    mask_confusion_mat = confusion_matrix(GTMasks, MaskPredictions)
    mask_tn = mask_confusion_mat[0][0]
    mask_fp = mask_confusion_mat[0][1]
    mask_fn = mask_confusion_mat[1][0]
    mask_tp = mask_confusion_mat[1][1]

    recall = mask_tp / (mask_tp + mask_fn)
    precision = mask_tp / (mask_tp + mask_fp)
    f1 = 2*(precision*recall) / (precision + recall)
    accuracy = (mask_tp + mask_tn) / (mask_tp + mask_tn + mask_fp + mask_fn)

    return precision, recall, accuracy, f1


def avg_depth_error(output, target):
    assert isinstance(output, list)
    Sigmoid = torch.nn.Sigmoid()
    B = len(target)

    total_depth_error = 0.
    total_depths = 0.
    for b in range(B):
        GTMask, GTDepth = target[b]

        if len(output[b]) == 2:
            PredMaskConf, PredDepth = output[b]
        else:
            PredMaskConf, PredDepth, PredMaskConst, PredConst = output[b]
            PredDepth += Sigmoid(PredMaskConst)*PredConst

        PredMaskConfSig = Sigmoid(PredMaskConf)
        PredMaskMaxConfVal = PredMaskConfSig
        ValidRaysIdx = PredMaskMaxConfVal > v3_utils.INTERSECTION_MASK_THRESHOLD  # Use predicted mask
        ValidRaysIdx = torch.logical_and(ValidRaysIdx, GTMask)
        
        # compute mean depth only when it is defined by both prediction and ground truth
        if torch.max(ValidRaysIdx) > 0.:
            total_depth_error += torch.sum(torch.abs(GTDepth[ValidRaysIdx] - PredDepth[ValidRaysIdx]))
            total_depths += torch.sum(ValidRaysIdx)
    return total_depth_error / total_depths


def infer(name, model, val_loader, hyperparameters, device):

    model.eval()
    arch = hyperparameters["architecture"]


    # loss functions
    depth_loss_fn = DepthLoss()
    intersection_loss_fn = IntersectionLoss()
    dfr_loss_fn = DepthFieldRegularizingLoss()
    cr_loss_fn = ConstantRegularizingLoss()

    losses = {}
    loss_names = ["val", "val_depth", "val_intersection"]
    if arch == "constant":
        loss_names += ["val_dfr", "val_cr"]
    for loss in loss_names:
        losses[loss] = []

    all_val_losses = []
    all_val_depth_losses = []
    all_val_intersection_losses = []
    all_val_dfr_losses = []
    all_val_cr_losses = []

    all_targets = []
    all_outputs = []

    print(f"Evaluating model {name}...")
    for batch in tqdm(val_loader):
        data, targets = batch
        data = v3_utils.sendToDevice(data, device)
        targets = v3_utils.sendToDevice(targets, device)
        all_targets += targets
        
        # #########   LOSSES   #########
        output = model(data)
        all_outputs += output
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

    losses = {}
    losses["val"] = np.mean(np.asarray(all_val_losses))
    losses["val_depth"] = np.mean(np.asarray(all_val_depth_losses))
    losses["val_intersection"] = np.mean(np.asarray(all_val_intersection_losses))
    if arch == "constant":
        losses["val_dfr"] = np.mean(np.asarray(all_val_dfr_losses))
        losses["val_cr"] =  np.mean(np.asarray(all_val_cr_losses))

    depth_error = avg_depth_error(all_outputs, all_targets)
    precision, recall, accuracy, f1 = praf1(all_outputs, all_targets)

    return losses, depth_error, precision, recall, accuracy, f1





if __name__ == '__main__':
    Parser = v3_utils.BaselineParser
    Parser.add_argument('--val-rays-per-shape', help='Number of ray samples per object shape for validation.', default=10, type=int)
    Parser.add_argument('--force-test-on-train', help='Choose to test on the training data. CAUTION: Use this for debugging only.', action='store_true', required=False)
    Parser.add_argument('--batch-size', help='Choose mini-batch size.', required=False, default=16, type=int)
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    hyperparameters = {
        "batch_size": Args.batch_size,
        "architecture": Args.arch,
        "dataset": Args.dataset
    }

    v3_utils.seedRandom(Args.seed)
    nCores=0#mp.cpu_count()
    Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {Device}")

    # Load Data
    if Args.force_test_on_train:
        print('[ WARN ]: VALIDATING ON TRAINING DATA.')
    ValData = DDL(root=Args.input_dir, name=Args.dataset, train=Args.force_test_on_train, download=True, target_samples=Args.val_rays_per_shape, usePositionalEncoding=Args.use_posenc)
    ValDataLoader = torch.utils.data.DataLoader(ValData, batch_size=Args.batch_size, shuffle=True, num_workers=nCores, collate_fn=DDL.collate_fn)


    if Args.arch == 'standard':
        NeuralODF = ODFSingleV3(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=10)
    elif Args.arch == 'constant':
        NeuralODF = ODFSingleV3Constant(input_size=(120 if Args.use_posenc else 6), radius=DEPTH_SAMPLER_RADIUS, pos_enc=Args.use_posenc, n_layers=10)


    # check to see if we have a model checkpoint
    if os.path.exists(os.path.join(Args.output_dir, Args.expt_name, "checkpoints")):
        checkpoint_dict = v3_utils.load_checkpoint(Args.output_dir, Args.expt_name, device=Device, load_best=True)
        NeuralODF.load_state_dict(checkpoint_dict['model_state_dict'])
        NeuralODF.to(Device)
    else:
        print(f"Unable to perform inference for model {Args.expt_name} because no checkpoints were found")


    # run inference
    losses, depth_error, precision, recall, accuracy, f1 = infer(Args.output_dir, Args.expt_name, NeuralODF, ValDataLoader, hyperparameters, Device)

    print("-"*20 + Args.expt_name + " Evaluation" + "-"*20)
    print("Losses:")
    for loss in losses:
        print(f"\t{loss}{' '*(20-len(loss))}: {losses[loss]:.3f}")
    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"Depth Error - {depth_error:.4f}")
