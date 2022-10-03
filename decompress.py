import os
from random import sample
import time
import argparse
from glob import glob

import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as Data
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points
import torchac

import AE
import pc_kit
import pc_io

parser = argparse.ArgumentParser(
    prog='decompress.py',
    description='Decompress Point Clouds',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('model_save_folder', help='Directory where to save/load trained models.')
parser.add_argument('compressed_path', help='Compressed file saving directory.')
parser.add_argument('decompressed_path', help='Decompressed point clouds saving directory.')
parser.add_argument('--load_scale', type=int, help='Input/Decompressed point cloud coordinate scale. [0, load_scale]', default=1)

args = parser.parse_args()

# CREATE DECOMPRESSED_PATH PATH
if not os.path.exists(args.decompressed_path):
    os.makedirs(args.decompressed_path)

# GET FILENAME FROM COMPRESSED PATH
files = glob(os.path.join(args.compressed_path, '*.ply.s.bin'))
filenames = [os.path.split(x[:-6])[-1] for x in files]

MODEL_FILE = os.path.join(args.model_save_folder, 'ae.pkl')
ae = torch.load(MODEL_FILE).cuda().eval()

def KNN_Patching(batch_x, sampled_xyz, K):
    dist, group_idx, grouped_xyz = knn_points(sampled_xyz, batch_x, K=K, return_nn=True)
    grouped_xyz -= sampled_xyz.view(1, -1, 1, 3)
    x_patches = grouped_xyz.view(-1, K, 3)
    return x_patches

def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    #print(cdf.shape)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    # On GPU, softmax followed by cumsum can lead to the final value being 
    # slightly bigger than 1, so we clamp.
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    return cdf_with_0

pmf = ae.get_pmf('cuda')
cdf = pmf_to_cdf(pmf)

start_time = time.time()
for i in tqdm(range(len(filenames))):
    #loaded = np.load(os.path.join(args.compressed_path, filenames[i] + '.s.npz'))
    #sampled_xyz = torch.Tensor(loaded['sampled'])#.cuda()
    sampled_xyz = np.fromfile(os.path.join(args.compressed_path, filenames[i] + '.s.bin'), dtype=np.float16)
    sampled_xyz = torch.Tensor(sampled_xyz)
    # print(sampled_xyz.shape)
    #latent = torch.Tensor(loaded['latent'])#.cuda()
    sampled_xyz = sampled_xyz.unsqueeze(0)
    S = np.fromfile(os.path.join(args.compressed_path, filenames[i] + '.h.bin'), dtype=np.uint16)[0]

    with open(os.path.join(args.compressed_path, filenames[i] + '.p.bin'), 'rb') as fin:
        byte_stream = fin.read()
    latent = torchac.decode_float_cdf(cdf.repeat((S, 1, 1)).cpu(), byte_stream)
    latent = latent.float() - 99

    new_xyz = ae.cpu().decoder(latent)
    new_xyz = new_xyz.reshape(S, -1, 3)

    pc = (new_xyz.view(1, S, -1, 3) + sampled_xyz.view(1, S, 1, 3)).reshape(-1, 3)
    pc = pc * args.load_scale
    pc_io.save_pc(pc.detach().cpu().numpy(), filenames[i] + '.dec.ply', path=args.decompressed_path)


t = time.time() - start_time
n = t / len(filenames)
print(f"Done! Execution time: {round(t, 3)}s, i.e., {round(n, 5)}s per point cloud.")
