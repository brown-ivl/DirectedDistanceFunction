import argparse
from ftplib import all_errors
import torch
import numpy as np
import os
import glob
import sys
import requests
import matplotlib.pyplot as plt
import math
import trimesh
import random

#####################################################
###################### SETUP ########################
#####################################################

INTERSECTION_MASK_THRESHOLD = 0.5

BaselineParser = argparse.ArgumentParser(description='Parser for NeuralODFs.')
BaselineParser.add_argument('--expt-name', help='Provide a name for this experiment.')
BaselineParser.add_argument('--input-dir', help='Provide the input directory where datasets are stored.')
BaselineParser.add_argument('--dataset', help='The dataset')
BaselineParser.add_argument('--output-dir', help='Provide the *absolute* output directory where checkpoints, logs, and other output will be stored (under expt_name).')
BaselineParser.add_argument('--arch', help='Architecture to use.', choices=['standard', 'constant'], default='standard')
BaselineParser.add_argument('--use-posenc', help='Choose to use positional encoding.', action='store_true', required=False)
BaselineParser.add_argument('-s', '--seed', help='Random seed.', required=False, type=int, default=42)

def seedRandom(seed):
    # NOTE: This gets us very close to deterministic but there are some small differences in 1e-4 and smaller scales
    print('[ INFO ]: Seeding RNGs with {}'.format(seed))
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#####################################################
################# LOADING/SAVING ####################
#####################################################
    

def checkpoint_filename(name, epoch):
    '''
    Returns the filename of a checkpoint for a specific epoch
    '''
    filename = f"{name}_checkpoint_{epoch:06}.tar"
    return filename

def load_checkpoint(save_dir, name, device="cpu", load_best=False):
    '''
    Load a model checkpoint
    if load_best is True, the best model checkpoint will be loaded instead of 
    '''
    if not load_best:
        checkpoint_dir = os.path.join(save_dir, name, "checkpoints")
        all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.tar"))
        if len(all_checkpoints) == 0:
            raise Exception(f"There are no saved checkpoints for {name}")
        all_checkpoints.sort()
        print(f"Loading checkpoint {os.path.basename(all_checkpoints[-1])}")
        return torch.load(all_checkpoints[-1], map_location=device)
    else:
        checkpoint_dir = os.path.join(save_dir, name, "checkpoints")
        best_file = os.path.join(save_dir, name, "best_checkpoint.txt")
        if not os.path.exists(best_file):
            raise Exception(f"Could not identify the best checkpoint. {best_file} was not found.")
        f = open(best_file, "r")
        best_epoch = int(f.read().split("$")[0])
        f.close()
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_filename(name, best_epoch))
        print(f"Loading checkpoint {os.path.basename(checkpoint_file)}")
        return torch.load(checkpoint_file, map_location=device)

def save_checkpoint(save_dir, checkpoint_dict):
    assert("name" in checkpoint_dict)
    assert("epoch" in checkpoint_dict)
    name = checkpoint_dict["name"]
    epoch = checkpoint_dict["epoch"]
    if not os.path.exists(os.path.join(save_dir, name)):
        os.mkdir(os.path.join(save_dir, name))
    if not os.path.exists(os.path.join(save_dir, name, "checkpoints")):
        os.mkdir(os.path.join(save_dir, name, "checkpoints"))
    out_file = os.path.join(save_dir, name, "checkpoints", checkpoint_filename(name, epoch))
    torch.save(checkpoint_dict, out_file)

    # check if this is the best checkpointt
    best_file = os.path.join(save_dir, name, "best_checkpoint.txt")
    is_best = False
    if "val" in checkpoint_dict["loss_history"]:
        if not os.path.exists(best_file):
            is_best = True
        else:
            f = open(best_file, "r+")
            best_val = float(f.read().split("$")[1])
            if checkpoint_dict["loss_history"]["val"][-1] < best_val:
                is_best = True
            f.close()
    if is_best:
        f = open(best_file, "w")
        f.write(f"{epoch}${checkpoint_dict['loss_history']['val'][-1]}")
        f.close()

def build_checkpoint(odf_model, mask_model, name, epoch, optimizer, scheduler, loss_history):
    checkpoint_dict = {
        'name': name,
        'epoch': epoch,
        'odf_model_state_dict': odf_model.state_dict(),
        'mask_model_state_dict': mask_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history,
    }
    return checkpoint_dict

def checkpoint(odf_model, mask_model, save_dir, name, epoch, optimizer, scheduler, loss_history):
    checkpoint_dict = build_checkpoint(odf_model, mask_model, name, epoch, optimizer, scheduler, loss_history)
    save_checkpoint(save_dir, checkpoint_dict)

def expandTilde(Path):
    if '~' == Path[0]:
        return os.path.expanduser(Path)

    return Path

def downloadFile(url, filename, verify=True):
    with open(expandTilde(filename), 'wb') as f:
        response = requests.get(url, stream=True, verify=verify)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}>{}]'.format('=' * done, '-' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')


#####################################################
#################### SAMPLING #######################
#####################################################

def sphere_interior_sampler(num_points, radius=1.25):
    '''
    Uniform sampling of sphere volume: https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    '''
    phi = np.random.uniform(0,2*np.pi,size=num_points)
    costheta = np.random.uniform(-1,1,size=num_points)
    theta = np.arccos(costheta)
    u = np.random.uniform(0,1,size=num_points)
    r = radius * np.cbrt(u)

    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )
    return np.concatenate((x[:,None], y[:,None], z[:,None]), axis=-1)

def sphere_surface_sampler(num_points, radius=1.00):
    '''
    uniform sampling from sphere surface by drawing from normal distribution
    '''
    normal_samples = np.random.normal(size=(num_points,3))
    surface_samples = normal_samples / np.stack((np.linalg.norm(normal_samples,axis=1),)*3, axis=1) * radius
    return surface_samples

def odf_domain_sampler(n_points, radius=1.25):
    '''
    Samples points uniformly at random from an ODF input domain
    '''
    # sample viewpoints on sphere
    sampled_positions = sphere_interior_sampler(n_points, radius=radius)
    sampled_directions = sphere_surface_sampler(n_points)

    coords = np.concatenate([sampled_positions, sampled_directions], axis=-1)
    return coords

#####################################################
###################### MESHES #######################
#####################################################

def load_object(obj_name, data_path):
    obj_file = os.path.join(data_path, f"{obj_name}.obj")

    obj_mesh = trimesh.load(obj_file)
    # obj_mesh.show()

    ## deepsdf normalization
    mesh_vertices = obj_mesh.vertices
    mesh_faces = obj_mesh.faces
    center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0))/2.0
    max_dist = np.linalg.norm(mesh_vertices - center, axis=1).max()
    max_dist = max_dist * 1.03
    mesh_vertices = (mesh_vertices - center) / max_dist
    obj_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    
    return mesh_vertices, mesh_faces, obj_mesh

#####################################################
###################### OTHER ########################
#####################################################

def positional_encoding(val, L=10):
    '''
    val - the value to apply the encoding to
    L   - controls the size of the encoding (size = 2*L  - see paper for details)
    Implements the positional encoding described in section 5.1 of NeRF
    https://arxiv.org/pdf/2003.08934.pdf
    '''
    return [x for i in range(L) for x in [math.sin(2**(i)*math.pi*val), math.cos(2**(i)*math.pi*val)]]


def positional_encoding_tensor(coords, L=10):
    assert(len(coords.shape)==2)
    columns = []
    for i in range(coords.shape[1]):
        columns += [coords[:,i]]+[x for j in range(L) for x in [torch.sin(2**i*coords[:,i]), torch.cos(2**(i)*coords[:,i])]]
    pos_encodings = torch.stack(columns, dim=-1)
    return pos_encodings

def sendToDevice(TupleOrTensor, Device):
    '''
    Send tensor or tuple to specified device
    '''
    if isinstance(TupleOrTensor, torch.Tensor):
        TupleOrTensorD = TupleOrTensor.to(Device)
    else:
        TupleOrTensorD = [None]*len(TupleOrTensor)
        for Ctr in range(len(TupleOrTensor)):
            TupleOrTensorD[Ctr] = sendToDevice(TupleOrTensor[Ctr], Device)
        if isinstance(TupleOrTensor, tuple):
            TupleOrTensorD = tuple(TupleOrTensorD)

    return TupleOrTensorD

def plotLosses(loss_history, save_dir, name):
    plt.clf()
    for loss in loss_history:
        if loss == "train" or loss == "val":
            plt.plot(loss_history[loss], linestyle='-', label=loss)
        else:
            plt.plot(loss_history[loss], linestyle="--", label=loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(name)
    plt.savefig(os.path.join(save_dir, name, "losses_curve.png"))
