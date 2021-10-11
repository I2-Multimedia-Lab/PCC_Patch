import os
import argparse

import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as Data
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

import AE
from bitEstimator import BitEstimator
import pc_kit
import pc_io

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser(
    prog='train_ae.py',
    description='Train autoencoder using point cloud patches',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('train_glob', help='Point clouds glob pattern for training.')
parser.add_argument('model_save_folder', help='Directory where to save/load trained models.')
parser.add_argument('--N', type=int, help='Point cloud resolution.', default=8192)
parser.add_argument('--ALPHA', type=int, help='The factor of patch coverage ratio.', default=2)
parser.add_argument('--K', type=int, help='Number of points in each patch.', default=128)
parser.add_argument('--d', type=int, help='Bottleneck size.', default=16)
parser.add_argument('--lr', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--batch_size', type=int, help='number of patches in a batch.', default=16)
parser.add_argument('--lamda', type=float, help='Lambda for rate-distortion tradeoff.', default=1e-06)
parser.add_argument('--rate_loss_enable_step', type=int, help='Apply rate-distortion tradeoff at x steps.', default=40000)
parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
parser.add_argument('--lr_decay_steps', type=int, help='Decays the learning rate every x steps.', default=60000)
parser.add_argument('--max_steps', type=int, help='Train up to this number of steps..', default=80000)

args = parser.parse_args()

S = args.N * args.ALPHA // args.K
k = args.K // args.ALPHA

if not os.path.exists(args.model_save_folder):
    os.makedirs(args.model_save_folder)

p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(1)
files = pc_io.get_files(args.train_glob)
points = pc_io.load_points(files, p_min, p_max)

points = torch.Tensor(points)
print(f'Point train samples: {points.shape[0]}, corrdinate range: [{points.min()}, {points.max()}]')

# PATCH DIVISION
print('Dividing ModelNet point clouds to patches...')
patches = []
for i in tqdm(range(points.shape[0])):
    pc = points[i].unsqueeze(0)
    sampled = pc_kit.index_points(pc, pc_kit.farthest_point_sample_batch(pc, S))
    dist, group_idx, grouped_xyz = knn_points(sampled, pc, K=args.K, return_nn=True)
    grouped_xyz -= sampled.view(1, S, 1, 3)
    patches.append(grouped_xyz.view(-1, args.K, 3))
patches = torch.cat(patches, dim=0)
print('We get patches:', patches.shape)

loader = Data.DataLoader(
    dataset = patches,
    batch_size = args.batch_size,
    shuffle = True,
)

ae = AE.get_model(k=k, d=args.d).cuda().train()
criterion = AE.get_loss().cuda()

optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr)

global_step = 0
bpps, losses = [], []

for epoch in range(9999):
    for step, (batch_x) in enumerate(loader):
        batch_x = batch_x.cuda()
        
        batch_x_pred, bpp = ae(batch_x)
        if global_step < args.rate_loss_enable_step:
            loss = criterion(batch_x_pred, batch_x, bpp, 0)
        else:
            loss = criterion(batch_x_pred, batch_x, bpp, args.lamda)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        # PRINT
        losses.append(loss.item())
        bpps.append(bpp.item())
        if global_step % 500 == 0:
            print(f'Epoch:{epoch} | Step:{global_step} | Estimated bpp:{round(np.array(bpps).mean(), 5)} | Loss:{round(np.array(losses).mean(), 5)}')
            losses, bpps = [], []

        if global_step % args.lr_decay_steps == 0:
            args.lr = args.lr * args.lr_decay
            for g in optimizer.param_groups:
                g['lr'] = args.lr
            print(f'Learning rate decay triggered at step {global_step}, LR is setting to{args.lr}.')
        
        # SAVE MODEL
        if global_step % 500 == 0:
            torch.save(ae, os.path.join(args.model_save_folder, 'ae.pkl'))


        if global_step >= args.max_steps:
            break
    if global_step >= args.max_steps:
        break