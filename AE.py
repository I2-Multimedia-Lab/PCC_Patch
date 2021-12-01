import math
import torch.nn as nn
import torch
import torch.nn.functional as F

from pc_kit import PointNet, SAPP
from bitEstimator import BitEstimator
from pytorch3d.loss import chamfer_distance


class get_model(nn.Module):
    def __init__(self, k, d):
        super(get_model, self).__init__()
        self.k = k
        self.d = d
        
        self.sa = SAPP(in_channel=3, feature_region=k//4, mlp=[32, 64, 128], bn=False)
        self.pn = PointNet(in_channel=3+128, mlp=[256, 512, 1024, d], relu=[True, True, True, False], bn=False)

        self.decoder = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, k*3),
        )

        self.be = BitEstimator(channel=d)

    def forward(self, xyz):
        B, K, C = xyz.shape

        # encode
        xyz = xyz.transpose(1, 2)
        feature = self.sa(xyz)
        feature = self.pn(torch.cat((xyz, feature), dim=1))

        # quantization
        if self.training:
            quantizated_feature = feature + torch.nn.init.uniform_(torch.zeros(feature.size()), -0.5, 0.5).cuda()
        else:
            quantizated_feature = torch.round(feature)
        bottleneck = quantizated_feature

        # decode
        new_xyz = self.decoder(bottleneck)
        new_xyz = new_xyz.reshape(B, -1, 3)

        # BITRATE ESTIMATION
        prob = self.be(bottleneck + 0.5) - self.be(bottleneck - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
        #print(total_bits)
        bpp = total_bits / K / B

        return new_xyz, bpp

    def get_pmf(self, device='cuda'):
        L = 99  # get cdf [-L, ..., L-1], total L*2 numbers
        pmf = torch.zeros(1, self.d, L*2).to(device)
        for l in range(-L, L):
            z = torch.ones((1, self.d)).to(device) * l
            pmf[0, :, l+L] = (self.be(z + 0.5) - self.be(z - 0.5))[0, :]
        #print(pmf.shape)
        return pmf


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, bpp, lamda):
        '''
        Input:
            pred: reconstructed point cloud (B, N, 3)
            target: origin point cloud (B, CxN, 3)
            bottleneck: 
        '''
        d, d_normals = chamfer_distance(pred, target)
        loss = d + lamda * bpp

        return loss