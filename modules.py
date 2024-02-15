import torch
import torch.nn as nn
from utils import assert_equal
import os
import pysr
import json
import einops
import spock_reg_model
import numpy as np
import torch.nn.functional as F


# instead of sigmoid and sum, use a softmax.
# sum over predicates is 1, decrease temperature over training so it specializes
# make it not worry about std - set to constant, etc
# lower lr, train longer



class IfThenNN(nn.Module):
    def __init__(self, n_preds, in_dim, out_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_preds = n_preds
        self.mlp = mlp(in_dim, n_preds * (1 + out_dim), hidden_dim, n_layers)

    def forward(self, x):
        out = self.mlp(x)
        preds = out[..., :self.n_preds]
        bodies = out[..., self.n_preds:]
        bodies = einops.rearrange(bodies, '... (n o) -> ... n o', n=self.n_preds)
        preds = torch.sigmoid(preds)
        return torch.einsum('... n, ... n o -> ... o', preds, bodies)


class IfThenNN2(nn.Module):
    def __init__(self, n_preds, in_dim, out_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_preds = n_preds
        self.mlp = mlp(in_dim, n_preds, hidden_dim, n_layers)
        self.bodies = nn.Parameter(torch.randn(n_preds, out_dim))

    def forward(self, x):
        preds = self.mlp(x)
        preds = torch.sigmoid(preds)
        return torch.einsum('... n, n o -> ... o', preds, self.bodies)


def mlp(in_n, out_n, hidden, layers, act='relu'):
    '''
        layers is the number of "hidden" layers,
        aka layers besides the input layer (which reduces to hidden dim)
        and output layer (which changes hidden dim to out dim)
        so:
        if layers = -1, returns Linear(in, out)
        if layers =  0, returns [Linear(in, h), Linear(h, out)
        if layers =  1, returns [Linear(in, h), Linear(h, h), Linear(h, out)]
        etc.
    '''
    if act == 'relu':
        act = nn.ReLU
    elif act == 'softplus':
        act = nn.Softplus
    else:
        raise NotImplementedError('act must be relu or softplus')

    if layers == -1:
        return nn.Linear(in_n, out_n)

    result = [nn.Linear(in_n, hidden),
             act()]
    for i in reversed(range(layers)):
        result.extend([
            nn.Linear(hidden, hidden),
            act()
            ])

    result.extend([nn.Linear(hidden, out_n)])
    return nn.Sequential(*result)

class ZeroFeatureAtIx(nn.Module):
    def __init__(self, n_features, ix):
        super().__init__()
        self.n_features = n_features
        self.ix = ix

    def forward(self, x):
        return torch.cat([x[..., :self.ix], torch.zeros_like(x[..., self.ix:self.ix+1]), x[..., self.ix+1:]], dim=-1)

class SpecialLinear(nn.Module):
    def __init__(self, n_inputs, n_features, init=False):
        super().__init__()
        self.n_inputs = n_inputs
        # number of features we're emulating
        self.n_features = n_features
        self.linear = nn.Linear(n_inputs + n_inputs * n_inputs, 2*n_features)
        if init:
            linear = spock_reg_model.load(total_steps=300000, seed=0, version=21101).feature_nn
            assert_equal(type(linear), nn.Linear)
            self.init_layer(linear)

    def forward(self, x):
        return self.linear(x)

    def from_linear(linear: nn.Linear):
        layer = SpecialLinear(n_inputs=linear.weight.shape[1], n_features=linear.weight.shape[0])
        layer.init_layer(linear)
        return layer

    def init_layer(self, layer: nn.Linear):
        # layer to emulate is a [n_inputs, n_features] linear layer.
        # originally we take the mean and variance of the features.
        # now we are trying to recreate those features by running a linear transformation the means and variances of the inputs
        w, b = layer.weight.data, layer.bias.data
        assert_equal(w.shape, (self.n_features, self.n_inputs))
        assert_equal(b.shape, (self.n_features,))

        with torch.no_grad():
            # values outside the mean/var-covar block diagonal are zero
            self.linear.weight.data[...] = 0
            # calculating the means from the means.
            self.linear.weight.data[:self.n_features, :self.n_inputs] = w
            # calculating the variances from the covariances
            w2 = torch.einsum('ij, ik -> ijk', w, w)
            w2 = einops.rearrange(w2, 'i j k -> i (j k)')
            self.linear.weight[self.n_features:, self.n_inputs:] = w2
            self.linear.bias.data[:self.n_features] = b
            self.linear.bias.data[self.n_features:] = 0


class MaskLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(n_features))

    def forward(self, x):
        return x * self.mask

    def l1_cost(self):
        return torch.sum(torch.abs(self.mask))


class MaskedLinear(nn.Module):
    def __init__(self, linear, mask, debug='none'):
        super().__init__()
        self.linear = linear
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.debug = debug

    def forward(self, x):
        if self.debug == '1':
            assert torch.all(self.mask == 1)
        elif self.debug == '2':
            weight = self.linear.weight * self.mask
            assert torch.all(weight == self.linear.weight)
        elif self.debug == '3':
            weight = self.linear.weight
        elif self.debug == '4':
            return self.linear(x)
        else:
            weight = self.linear.weight * self.mask

        return F.linear(x, weight, self.linear.bias)


def pruned_linear(linear: nn.Linear, top_k=None, threshold=None, debug=None):
    if top_k is not None:
        mask = torch.zeros_like(linear.weight)
        for r in range(linear.weight.shape[0]):
            _, ixs = torch.topk(linear.weight[r].abs(), k=top_k)
            mask[r][ixs] = 1
    elif threshold is not None:
        mask = torch.zeros_like(linear.weight)
        mask[linear.weight.abs() > threshold] = 1

        if debug is not None:
            linear = nn.Linear(linear.weight.shape[1], linear.weight.shape[0])
            mask = torch.ones_like(mask)

    return MaskedLinear(linear, mask, debug)


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


class SumModule(nn.Module):
    def __init__(self, module1, module2):
        super(SumModule, self).__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, x):
        return self.module1(x) + self.module2(x)


def load_pysr_module_list(filepath, model_selection):
    if model_selection in ['best', 'accuracy', 'score']:
        return nn.ModuleList(pysr.PySRRegressor.from_file(filepath, model_selection=model_selection).pytorch())
    else:
        reg = pysr.PySRRegressor.from_file(filepath)
        # find the ixs with closest complexity equal to model_selection
        ixs = []
        for i in range(reg.nout_):
            ix = np.argmin(np.abs(reg.equations_[i]['complexity'] - int(model_selection)))
            ixs.append(ix)

        print('PySR model selection ixs: ', ixs)
        return nn.ModuleList(reg.pytorch(index=ixs))


class PySRNet(nn.Module):
    def __init__(self, filepath='sr_results/hall_of_fame_21101_0_1.pkl', model_selection='best'):
        super().__init__()
        # something like 'sr_results/hall_of_fame_21101_0_1.pkl'
        self.filepath = filepath
        assert os.path.exists(filepath), f'filepath does not exist: {filepath}'
        self.module_list = load_pysr_module_list(filepath, model_selection)

    def forward(self, x):
        # input: [B, d]
        # output: [B, n]
        out = [module(x) for module in self.module_list]
        # if the pysr equation is a constant, it returns a scalar for some reason
        # repeat [,] to [B, ]
        if len(out[0].shape) == 0:
            out = einops.repeat(torch.tensor(out, device=x.device), 'n -> B n', B=x.shape[0])
        else:
            out = einops.rearrange(out, 'n B -> B n')
        return out


class PySRFeatureNN(torch.nn.Module):
    def __init__(self, filepath='sr_results/hall_of_fame_1278_1_0.pkl', model_selection='best'):
        super().__init__()
        # something like 'sr_results/hall_of_fame_7955_1.pkl'
        self.filepath = filepath
        assert os.path.exists(filepath), f'filepath does not exist: {filepath}'
        indices_path = filepath[:-4] + '_indices.json'

        self.module_list = load_pysr_module_list(filepath, model_selection)

        with open(indices_path, 'r') as f:
            self.included_indices = json.load(f)

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
