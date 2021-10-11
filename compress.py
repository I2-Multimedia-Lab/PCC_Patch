import os
import argparse
import subprocess

import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.utils.data as Data
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

import AE
import pc_kit
import pc_io

parser = argparse.ArgumentParser(
    prog='compress.py',
    description='Compress Point Clouds',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('model_save_folder', help='Directory where to save/load trained models.')
parser.add_argument('glob_input_path', help='Point clouds glob pattern to be compressed.')
parser.add_argument('compressed_path', help='Compressed file saving directory.')
parser.add_argument('--load_scale', type=int, help='Input point cloud coordinate scale. [0, load_scale]', default=1)
parser.add_argument('--ALPHA', type=int, help='The factor of patch coverage ratio.', default=2)
#parser.add_argument('--K', type=int, help='Number of points in each patch.', default=128)
#parser.add_argument('--d', type=int, help='Bottleneck size.', default=16)

args = parser.parse_args()

# CREATE COMPRESSED PATH
if not os.path.exists(args.compressed_path):
    os.makedirs(args.compressed_path)

# READ INPUT FILES
p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(args.load_scale)
files = pc_io.get_files(args.glob_input_path)
filenames = np.array([os.path.split(x)[1] for x in files])

MODEL_FILE = os.path.join(args.model_save_folder, 'ae.pkl')
ae = torch.load(MODEL_FILE).cuda().eval()

def KNN_Patching(batch_x, sampled_xyz, K):
    dist, group_idx, grouped_xyz = knn_points(sampled_xyz, batch_x, K=K, return_nn=True)
    grouped_xyz -= sampled_xyz.view(1, -1, 1, 3)
    x_patches = grouped_xyz.view(-1, K, 3)
    return x_patches

# DO THE COMPRESS
with torch.no_grad():
    for i in tqdm(range(filenames.shape[0])):
        # GET 1 POINT CLOUD
        pc = pc_io.load_points([files[i]], p_min, p_max, processbar=False)
        pc = torch.Tensor(pc).cuda()
        pc = pc / args.load_scale

        N = pc.shape[1]
        K = ae.k * args.ALPHA // 1
        k = ae.k
        S = N * args.ALPHA // K
        d = ae.d
        
        # SAMPLING
        sampled_xyz = pc_kit.index_points(pc, pc_kit.farthest_point_sample_batch(pc, S))
         # DIVIDE PATCH BY KNN
        x_patches = KNN_Patching(pc, sampled_xyz, K)
        # DO THE ANALYSIS TRANSFORM FOR PATCHES
        x_patches = x_patches.transpose(1, 2)
        # FEED X_PATCHES ONE BY ONE
        patch_features = []

        for j in range(S):
            patch_feature = ae.sa(x_patches[j].view(1, 3, K))
            patch_features.append(patch_feature)
        patch_features = torch.cat(patch_features)
        latent = ae.pn(torch.cat((x_patches, patch_features), dim=1))
        latent_quantized = torch.round(latent)
        
        # SAVE AS FILE
        #sampled_xyz = np.round(sampled_xyz.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        sampled_xyz = (sampled_xyz.squeeze(0).cpu().numpy()).astype(np.float16)
        latent_quantized = latent_quantized.cpu().numpy().astype(np.int8)
        
        np.savez_compressed(os.path.join(args.compressed_path, filenames[i]), sampled=sampled_xyz, latent=latent_quantized)
        subprocess.call('xz -k -9 ' + os.path.join(args.compressed_path, filenames[i] + '.npz'), shell=True)

