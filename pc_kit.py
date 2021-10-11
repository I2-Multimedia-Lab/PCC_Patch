import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    if N < npoint:
        idxes = np.hstack((np.tile(np.arange(N), npoint//N), np.random.randint(N, size=npoint%N)))
        return point[idxes, :]

    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def farthest_point_sample_batch(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    #print('points size:', points.size(), 'idx size:', idx.size())
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    # view_shape == [B, S, K]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    # view_shape == [B, 1, 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # repeat_shape == [1, S, K]
    #print('points:', points.size(), ', idx:', idx.size(), ', view_shape:', view_shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # batch_indices == tensor[0, 1, ..., B-1]
    #print('batch_indices:', batch_indices.size())
    batch_indices = batch_indices.view(view_shape)
    # batch_indices size == [B, 1, 1]
    #print('after view batch_indices:', batch_indices.size())
    batch_indices = batch_indices.repeat(repeat_shape)
    # batch_indices size == [B, S, K]
    new_points = points[batch_indices, idx.long(), :]
    return new_points

# POINTNET
class PointNet(nn.Module):
    def __init__(self, in_channel, mlp, relu, bn):
        super(PointNet, self).__init__()

        mlp.insert(0, in_channel)
        self.mlp_Modules = nn.ModuleList()
        for i in range(len(mlp) - 1):
            if relu[i]:
                if bn:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlp[i], mlp[i+1], 1),
                        nn.BatchNorm2d(mlp[i+1]),
                        nn.ReLU(),
                        )
                else:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlp[i], mlp[i+1], 1),
                        nn.ReLU(),
                        )
            else:
                mlp_Module = nn.Sequential(
                    nn.Conv2d(mlp[i], mlp[i+1], 1),
                    )
            self.mlp_Modules.append(mlp_Module)


    def forward(self, points):
        """
        Input:
            points: input points position data, [B, C, N]
        Return:
            points: feature data, [B, D]
        """
        
        points = points.unsqueeze(-1) # [B, C, N, 1]
        
        for m in self.mlp_Modules:
            points = m(points)
        # [B, D, N, 1]
        
        #points_np = points.detach().cpu().numpy()
        #np.save('./npys/ae_pn_feature.npy', points_np)

        points = torch.max(points, 2)[0]    # [B, D, 1]
        points = points.squeeze(-1) # [B, D] 

        return points

class SAPP(nn.Module):
    def __init__(self, feature_region, in_channel, mlp, bn=False):
        super(SAPP, self).__init__()
        self.K = feature_region
        self.bn = bn

        if self.bn:
            self.bn0 = nn.BatchNorm2d(mlp[0])
            self.bn1 = nn.BatchNorm2d(mlp[1])
            self.bn2 = nn.BatchNorm2d(mlp[2])

        self.conv0 = nn.Conv2d(in_channel, mlp[0], 1)
        self.conv1 = nn.Conv2d(mlp[0], mlp[1], 1)
        self.conv2 = nn.Conv2d(mlp[1], mlp[2], 1)


    def forward(self, xyz):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # 转置
        xyz = xyz.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = N
        K = self.K
    
        # 使用farthest point sample从点列中采样出S个点
        new_xyz = xyz
        #dist, group_idx = self.knn(xyz, new_xyz)
        
        #print('group_idx:', group_idx.size())
        #print(group_idx)
        #grouped_xyz = index_points(xyz, group_idx)
        dists, idx, grouped_xyz = knn_points(new_xyz, xyz, K=self.K, return_nn=True)
        grouped_xyz -= new_xyz.view(B, S, 1, C)

        # 接下来将分组过后的点集计算特征值
        grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]

        grouped_points = F.relu(self.bn0(self.conv0(grouped_points))) if self.bn else F.relu(self.conv0(grouped_points))
        grouped_points = F.relu(self.bn1(self.conv1(grouped_points))) if self.bn else F.relu(self.conv1(grouped_points))
        grouped_points = F.relu(self.bn2(self.conv2(grouped_points))) if self.bn else F.relu(self.conv2(grouped_points))

        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]

        #new_xyz = new_xyz.permute(0, 2, 1)

        return new_points

