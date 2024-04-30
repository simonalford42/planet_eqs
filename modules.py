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
from petit20_survival_time import Tsurv

class Products2(nn.Module):
    def __init__(self):
        super().__init__()
        # [a1, e1], [a2, e2], [a3, e3]
        self.arg1s = [[8, 9], [17, 18], [26, 27]]
        # [sin/cos of angles for planet 1, planet 2, planet3]
        self.arg2s = [[11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25], [29, 30, 31, 32, 33, 34]]
        self.products = []
        for a, b in zip(self.arg1s, self.arg2s):
            for i in a:
                for j in b:
                    self.products.append((i, j))

        self.products = torch.tensor(self.products)

    def forward(self, x):
        # return all of the normal features, as well as the specified products
        products = x[..., self.products[:, 0]] * x[..., self.products[:, 1]]
        # x is [B, N, d] and products is [B, N, 36]. go to [B, N, d + 36]
        return torch.cat([x, products], dim=-1)


class Products(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [..., d]
        # returns: [..., d * d]
        x = torch.einsum('... i, ... j -> ... ij', x, x)
        x = einops.rearrange(x, '... i j -> ... (i j)')
        return x

# instead of sigmoid and sum, use a softmax.
# sum over predicates is 1, decrease temperature over training so it specializes
# make it not worry about std - set to constant, etc
# lower lr, train longer

class IfThenNN(nn.Module):
    # predicates are mlp's, bodies are mlp's
    def __init__(self, n_preds, in_dim, out_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_preds = n_preds
        self.mlp = mlp(in_dim, n_preds * (1 + out_dim), hidden_dim, n_layers)

    def forward(self, x, temperature=1):
        out = self.mlp(x)
        preds = out[..., :self.n_preds]
        bodies = out[..., self.n_preds:]
        bodies = einops.rearrange(bodies, '... (n o) -> ... n o', n=self.n_preds)
        preds = F.softmax(preds / temperature, dim=-1)
        return torch.einsum('... n, ... n o -> ... o', preds, bodies)


class IfThenNN2(nn.Module):
    # predicates are mlp's, bodies are constants
    def __init__(self, n_preds, in_dim, out_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_preds = n_preds
        self.mlp = mlp(in_dim, n_preds, hidden_dim, n_layers)
        self.bodies = nn.Parameter(torch.randn(n_preds, out_dim))

    def forward(self, x, temperature=1):
        preds = self.mlp(x)
        preds = F.softmax(preds / temperature, dim=-1)
        return torch.einsum('... n, n o -> ... o', preds, self.bodies)

    def preds(self, x, temperature=1):
        preds = self.mlp(x)
        return F.softmax(preds / temperature, dim=-1)


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
    def __init__(self, linear, mask):
        super().__init__()
        self.linear = linear
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        weight = self.linear.weight * self.mask
        return F.linear(x, weight, self.linear.bias)


def pruned_linear(linear: nn.Linear, top_k, top_n=None):
    '''
    top_k: for each feature, number of weights to keep
    top_n: number of features to keep
    '''

    mask = torch.zeros_like(linear.weight)
    for r in range(linear.weight.shape[0]):
        _, ixs = torch.topk(linear.weight[r].abs(), k=top_k)
        mask[r][ixs] = 1

    if top_n is not None:
        # mask is (n_features, n_inputs)
        _, ixs = torch.topk(mask.abs().sum(dim=-1), k=top_n)
        mask[~ixs] = 0

    return MaskedLinear(linear, mask)


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


class BioLinear(nn.Module):
    # BioLinear is just Linear, but each neuron comes with coordinates.
    def __init__(self, in_dim, out_dim, in_fold=1, out_fold=1):
        super(BioLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.in_fold = in_fold # in_fold is the number of folds applied to input vectors. It only affects coordinates, not computations.
        self.out_fold = out_fold # out_fold is the number of folds applied to output vectors. It only affects coordinates, not computations.
        assert in_dim % in_fold == 0
        assert out_dim % out_fold == 0
        #compute in_cor, shape: (in_dim)
        in_dim_fold = int(in_dim/in_fold)
        out_dim_fold = int(out_dim/out_fold)
        self.in_coordinates = torch.tensor(list(np.linspace(1/(2*in_dim_fold), 1-1/(2*in_dim_fold), num=in_dim_fold))*in_fold, dtype=torch.float) # place input neurons in 1D Euclidean space
        self.out_coordinates = torch.tensor(list(np.linspace(1/(2*out_dim_fold), 1-1/(2*out_dim_fold), num=out_dim_fold))*out_fold, dtype=torch.float) # place output neurons in 1D Euclidean space
        self.input = None
        self.output = None
        
    def forward(self, x):
        self.input = x.clone()
        self.output = self.linear(x).clone()
        return self.output
    
    
class BioMLP(nn.Module):
    # BioMLP is just MLP, but each neuron comes with coordinates.
    def __init__(self, in_dim=2, out_dim=2, w=2, depth=2, shp=None, token_embedding=False, embedding_size=None):
        super(BioMLP, self).__init__()
        if shp == None:
            shp = [in_dim] + [w]*(depth-1) + [out_dim]
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.depth = depth
                 
        else:
            self.in_dim = shp[0]
            self.out_dim = shp[-1]
            self.depth = len(shp) - 1

        linear_list = []
        for i in range(self.depth):
            if i == 0:
                linear_list.append(BioLinear(shp[i], shp[i+1], in_fold=1))
                
            else:
                linear_list.append(BioLinear(shp[i], shp[i+1]))
        self.linears = nn.ModuleList(linear_list)
        
        
        if token_embedding == True:
            # embedding size: number of tokens * embedding dimension
            self.embedding = torch.nn.Parameter(torch.normal(0,1,size=embedding_size))
        
        self.shp = shp
        # parameters for the bio-inspired trick
        self.l0 = 0.1 # distance between two nearby layers
        self.in_perm = torch.nn.Parameter(torch.tensor(np.arange(int(self.in_dim/self.linears[0].in_fold)), dtype=torch.float))
        self.out_perm = torch.nn.Parameter(torch.tensor(np.arange(int(self.out_dim/self.linears[-1].out_fold)), dtype=torch.float))
        self.top_k = 5 # the number of important neurons (used in Swaps)
        self.token_embedding = token_embedding
        self.n_parameters = sum(p.numel() for p in self.parameters())
        self.original_params = None

    def forward(self, x):
        rearranged = False
        if x.dim() > 2:
            rearranged = True
            B, T, d = x.shape
            x = einops.rearrange(x, 'B T d -> (B T) d')        

        shp = x.shape
        in_fold = self.linears[0].in_fold
        x = x.reshape(shp[0], in_fold, int(shp[1]/in_fold))
        x = x[:, :, self.in_perm.long()]
        x = x.reshape(shp[0], shp[1])
        # f = torch.nn.SiLU()
        f = torch.nn.ReLU()
        for i in range(self.depth - 1):
            x = f(self.linears[i](x))
        x = self.linears[-1](x)

        out_perm_inv = torch.zeros(self.out_dim, dtype=torch.long)
        out_perm_inv[self.out_perm.long()] = torch.arange(self.out_dim)
        x = x[:, out_perm_inv]
        
        if rearranged:
            x = einops.rearrange(x, '(B T) n -> B T n', B=B, T=T)

        return x

    def get_linear_layers(self):
        return self.linears
    
    def get_cc(self, weight_factor=1.0, bias_penalize=True, no_penalize_last=False):
        # compute connection cost
        # bias_penalize = True penalizes biases, otherwise doesn't penalize biases
        # no_penalize_last = True means do not penalize last linear layer, False means penalize last layer.
        cc = 0
        num_linear = len(self.linears)
        for i in range(num_linear):
            if i == num_linear - 1 and no_penalize_last:
                weight_factor = 0.
            biolinear = self.linears[i]
            dist = torch.abs(biolinear.out_coordinates.unsqueeze(dim=1) - biolinear.in_coordinates.unsqueeze(dim=0))
            cc += torch.sum(torch.abs(biolinear.linear.weight)*(weight_factor*dist+self.l0))
            if bias_penalize == True:
                cc += torch.sum(torch.abs(biolinear.linear.bias)*(self.l0))
        if self.token_embedding:
            cc += torch.sum(torch.abs(self.embedding)*(self.l0))
            #pass
        return cc
    
    def swap_weight(self, weights, j, k, swap_type="out"):
        # Given a weight matrix, swap the j^th and k^th neuron in inputs/outputs when swap_type = "in"/"out"
        with torch.no_grad():  
            if swap_type == "in":
                temp = weights[:,j].clone()
                weights[:,j] = weights[:,k].clone()
                weights[:,k] = temp
            elif swap_type == "out":
                temp = weights[j].clone()
                weights[j] = weights[k].clone()
                weights[k] = temp
            else:
                raise Exception("Swap type {} is not recognized!".format(swap_type))
            
    def swap_bias(self, biases, j, k):
        # Given a bias vector, swap the j^th and k^th neuron.
        with torch.no_grad():  
            temp = biases[j].clone()
            biases[j] = biases[k].clone()
            biases[k] = temp
    
    def swap(self, i, j, k):
        # in the ith layer (of neurons), swap the jth and the kth neuron. 
        # Note: n layers of weights means n+1 layers of neurons.
        linears = self.get_linear_layers()
        num_linear = len(linears)
        if i == 0:
            # input layer, only has outgoing weights; update in_perm
            weights = linears[i].linear.weight
            infold = linears[i].in_fold
            fold_dim = int(weights.shape[1]/infold)
            for l in range(infold):
                self.swap_weight(weights, j+fold_dim*l, k+fold_dim*l, swap_type="in")
            # change input_perm
            self.swap_bias(self.in_perm, j, k)
        elif i == num_linear:
            # output layer, only has incoming weights and biases; update out_perm
            weights = linears[i-1].linear.weight
            biases = linears[i-1].linear.bias
            self.swap_weight(weights, j, k, swap_type="out")
            self.swap_bias(biases, j, k)
            # change output_perm
            self.swap_bias(self.out_perm, j, k)
        else:
            # middle layer : incoming weights, outgoing weights, and biases
            weights_in = linears[i-1].linear.weight
            weights_out = linears[i].linear.weight
            biases = linears[i-1].linear.bias
            self.swap_weight(weights_in, j, k, swap_type="out")
            self.swap_weight(weights_out, j, k, swap_type="in")
            self.swap_bias(biases, j, k)

    def get_top_id(self, i, top_k=20):
        # in the ith layer (of neurons), get the top k important neurons (have large weight connections with other neurons)
        linears = self.get_linear_layers()
        num_linear = len(linears)
        if i == 0:
            # input layer
            weights = linears[i].linear.weight
            score = torch.sum(torch.abs(weights), dim=0)
            in_fold = linears[0].in_fold
            #print(score.shape)
            score = torch.sum(score.reshape(in_fold, int(score.shape[0]/in_fold)), dim=0)
        elif i == num_linear:
            # output layer
            weights = linears[i-1].linear.weight
            score = torch.sum(torch.abs(weights), dim=1)
        else:
            weights_in = linears[i-1].linear.weight
            weights_out = linears[i].linear.weight
            score = torch.sum(torch.abs(weights_out), dim=0) + torch.sum(torch.abs(weights_in), dim=1)
        #print(score.shape)
        top_index = torch.flip(torch.argsort(score),[0])[:top_k]
        return top_index
    
    def relocate_ij(self, i, j):
        # In the ith layer (of neurons), relocate the jth neuron
        linears = self.get_linear_layers()
        num_linear = len(linears)
        if i < num_linear:
            num_neuron = int(linears[i].linear.weight.shape[1]/linears[i].in_fold)
        else:
            num_neuron = linears[i-1].linear.weight.shape[0]
        ccs = []
        for k in range(num_neuron):
            self.swap(i,j,k)
            ccs.append(self.get_cc())
            self.swap(i,j,k)
        k = torch.argmin(torch.stack(ccs))
        self.swap(i,j,k)
            
    def relocate_i(self, i):
        # Relocate neurons in the ith layer
        top_id = self.get_top_id(i, top_k=self.top_k)
        for j in top_id:
            self.relocate_ij(i,j)
            
    def relocate(self):
        # Relocate neurons in the whole model
        linears = self.get_linear_layers()
        num_linear = len(linears)
        for i in range(num_linear+1):
            self.relocate_i(i)
            
    def plot(self):
        fig, ax = plt.subplots(figsize=(3,3))
        #ax = plt.gca()
        shp = self.shp
        s = 1/(2*max(shp))
        for j in range(len(shp)):
            N = shp[j]
            if j == 0:
                in_fold = self.linears[j].in_fold
                N = int(N/in_fold)
            for i in range(N):
                if j == 0:
                    for fold in range(in_fold):
                        circle = Ellipse((1/(2*N)+i/N, 0.1*j+0.02*fold-0.01), s, s/10*((len(shp)-1)+0.4), color='black')
                        ax.add_patch(circle)
                else:
                    for fold in range(in_fold):
                        circle = Ellipse((1/(2*N)+i/N, 0.1*j), s, s/10*((len(shp)-1)+0.4), color='black')
                        ax.add_patch(circle)


        plt.ylim(-0.02,0.1*(len(shp)-1)+0.02)
        plt.xlim(-0.02,1.02)

        linears = self.linears
        for ii in range(len(linears)):
            biolinear = linears[ii]
            p = biolinear.linear.weight
            p_shp = p.shape
            p = p/torch.abs(p).max()
            in_fold = biolinear.in_fold
            fold_num = int(p_shp[1]/in_fold)
            for i in range(p_shp[0]):
                if ii == 0:
                    for fold in range(in_fold):
                        for j in range(fold_num):
                            plt.plot([1/(2*p_shp[0])+i/p_shp[0], 1/(2*fold_num)+j/fold_num], [0.1*(ii+1),0.1*ii+0.02*fold-0.01], lw=1*np.abs(p[i,j].detach().numpy()), color="blue" if p[i,j]>0 else "red")
                else:
                    for j in range(fold_num):
                        plt.plot([1/(2*p_shp[0])+i/p_shp[0], 1/(2*fold_num)+j/fold_num], [0.1*(ii+1),0.1*ii], lw=0.5*np.abs(p[i,j].detach().numpy()), color="blue" if p[i,j]>0 else "red")
                    
        ax.axis('off')
        
    def thresholding(self, threshold, checkpoint = True):
        # snap too small weights (smaller than threshold) to zero. Useful for pruning.
        num = 0
        if checkpoint:
            self.original_params = [param.clone() for param in self.parameters()]
        with torch.no_grad():
            for param in self.parameters():
                num += torch.sum(torch.abs(param)>threshold)
                param.data = param*(torch.abs(param)>threshold)
        return num
                
    def intervening(self, i, pos, value, ptype="weight", checkpoint = True):
        if checkpoint:
            self.original_params = [param.clone() for param in self.parameters()]
        with torch.no_grad():
            if ptype == "weight":
                self.linears[i].linear.weight[pos] = value
            elif ptype == "bias":
                self.linears[i].linear.bias[pos] = value
                
    def revert(self):
        with torch.no_grad():
            for param, original_param in zip(self.parameters(), self.original_params):
                param.data.copy_(original_param.data)
