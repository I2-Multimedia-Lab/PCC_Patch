import os
import time
import argparse
from glob import glob

import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as Data
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

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
files = glob(os.path.join(args.compressed_path, '*.ply.npz'))
filenames = [x[len(args.compressed_path):-4] for x in files]
filenames = [os.path.split(x[:-4])[-1] for x in files]

MODEL_FILE = os.path.join(args.model_save_folder, 'ae.pkl')
ae = torch.load(MODEL_FILE).cuda().eval()

start_time = time.time()
for i in tqdm(range(len(filenames))):
    loaded = np.load(os.path.join(args.compressed_path, filenames[i] + '.npz'))
    sampled_xyz = torch.Tensor(loaded['sampled']).cuda()
    latent = torch.Tensor(loaded['latent']).cuda()

    sampled_xyz = sampled_xyz.unsqueeze(0)#.float() / 255
    latent = latent.float()

    S = sampled_xyz.shape[1]

    new_xyz = ae.decoder(latent)
    new_xyz = new_xyz.reshape(S, -1, 3)

    pc = (new_xyz.view(1, S, -1, 3) + sampled_xyz.view(1, S, 1, 3)).reshape(-1, 3)
    pc = pc * args.load_scale
    pc_io.save_pc(pc.detach().cpu().numpy(), filenames[i] + '.dec.ply', path=args.decompressed_path)

t = time.time() - start_time
n = t / len(filenames)
print(f"Done! Execution time: {round(t, 3)}s, i.e., {round(n, 5)}s per point cloud.")
