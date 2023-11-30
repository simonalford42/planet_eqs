import torch
import torch.nn as nn
from utils import assert_equal
import os
import pysr
import json
import einops

class MaskLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(n_features))

    def forward(self, x):
        return x * self.mask

    def l1_cost(self):
        return torch.sum(torch.abs(self.mask))


class Cyborg(nn.Module):
    def __init__(self, nn1, nn2, out_n, nn1_ixs=None, nn2_ixs=None):
        super().__init__()
        self.nn1 = nn1
        self.nn2 = nn2

        if nn1_ixs is None:
            assert nn2_ixs is not None
            nn1_ixs = [i for i in range(out_n) if i not in nn2_ixs]
        if nn2_ixs is None:
            assert nn1_ixs is not None
            nn2_ixs = [i for i in range(out_n) if i not in nn1_ixs]
        assert sorted(nn1_ixs + nn2_ixs) == list(range(out_n)), 'invalid ixs provided'

        self.nn1_ixs = nn1_ixs
        self.nn2_ixs = nn2_ixs

    def forward(self, x):
        # [B, T, d]
        out1 = self.nn1(x)
        out2 = self.nn2(x)
        out = torch.cat([out1[..., self.nn1_ixs], out2[..., self.nn2_ixs]], dim=-1)
        assert_equal(out.shape, out1.shape)
        return out


class PySRFeatureNN(torch.nn.Module):
    def __init__(self, filepath='sr_results/hall_of_fame_1278_1_0.pkl', model_selection='best'):
        super().__init__()
        # something like 'sr_results/hall_of_fame_7955_1.pkl'
        self.filepath = filepath
        assert os.path.exists(filepath), f'filepath does not exist: {filepath}'
        indices_path = filepath[:-4] + '_indices.json'
        self.module_list = nn.ModuleList(pysr.PySRRegressor.from_file(filepath, model_selection=model_selection).pytorch())
        with open(indices_path, 'r') as f:
            self.included_indices = json.load(f)

        print('PySR NN equations:')
        for m in self.module_list:
            print(m)

    def forward(self, x):
        B, T, d = x.shape
        x = x[..., self.included_indices]
        # input: [B, T, d]
        # output: [B, n, d]
        # the learned features expect a single batch axis as input
        x = einops.rearrange(x, 'B T d -> (B T) d')
        # list of length n_features
        x = [module(x) for module in self.module_list]
        x = einops.rearrange(x, 'n (B T) -> B T n', B=B, T=T)
        return x


class RandomFeatureNN(torch.nn.Module):
    def __init__(self, in_n, out_n):
        super().__init__()
        self.in_n = in_n
        self.out_n = out_n
        # not learnable!
        self.random_projection = nn.Parameter(torch.rand(in_n, out_n) * 2 - 1, requires_grad=False)

    def forward(self, x):
        B, T, d = x.shape
        assert_equal(d, self.in_n)
        # basically double batch matrix multiply
        out = torch.einsum('ijk, kl', x, self.random_projection)
        assert_equal(out.shape, (B, T, self.out_n))
        return out


class ZeroNN(torch.nn.Module):
    def __init__(self, in_n, out_n):
        super().__init__()
        self.in_n = in_n
        self.out_n = out_n
        assert out_n < in_n

    def forward(self, x):
        B, T, d = x.shape
        return torch.zeros_like(x)[:, :, :self.out_n]
