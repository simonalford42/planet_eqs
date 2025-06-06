import pickle as pkl
from copy import deepcopy as copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
import matplotlib as mpl
mpl.use('agg')
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import math
import pytorch_lightning as pl
import math
from functools import wraps
import warnings
from torch.optim.optimizer import Optimizer
from collections import OrderedDict
import einops
import utils
import modules
import glob
from matplotlib import pyplot as plt
import os



def load(version, seed=None):
    path = utils.ckpt_path(version, seed)

    f = path + '/version=0-v0.ckpt'
    if os.path.exists(f):
        return VarModel.load_from_checkpoint(f)
    f = path + '/version=0.ckpt'
    if os.path.exists(f):
        return VarModel.load_from_checkpoint(f)

    # try again one directory up, in case we're running from figures/ folder
    path = '../' + path
    f = path + '/version=0-v0.ckpt'
    if os.path.exists(f):
        return VarModel.load_from_checkpoint(f)
    f = path + '/version=0.ckpt'
    if os.path.exists(f):
        return VarModel.load_from_checkpoint(f)

    import pdb; pdb.set_trace()
    raise ValueError(f'Could not find model for version {version} and seed {seed}')


def load_with_pysr_f2(version, pysr_version, pysr_model_selection='accuracy', pysr_dir='sr_results/'):
    model = load(version)
    pysr_net = modules.get_pysr_regress_nn(pysr_version, pysr_model_selection, results_dir=pysr_dir)
    model.regress_nn = pysr_net
    return model
    # from modules import AddStdPredNN, Pred1StdNN
    # model2 = load(version)
    # model = AddStdPredNN(model.cuda(), Pred1StdNN().cuda())
    # return model


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
        f = torch.nn.SiLU()
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


class CustomOneCycleLR(torch.optim.lr_scheduler._LRScheduler):
    """Custom version of one-cycle learning rate to stop early"""
    def __init__(self,
                 optimizer,
                 max_lr,
                 swa_steps_start,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 last_epoch=-1):

        total_steps = swa_steps_start #Just fix afterwards.
        epochs = None
        steps_per_epoch = None
        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Validate total_steps
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError("Expected non-negative integer total_steps, but got {}".format(total_steps))
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError("Expected non-negative integer epochs, but got {}".format(epochs))
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError("Expected non-negative integer steps_per_epoch, but got {}".format(steps_per_epoch))
            self.total_steps = epochs * steps_per_epoch
        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))

        # Validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
        elif anneal_strategy == 'cos':
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear

        # Initialize learning rate variables
        max_lrs = self._format_param('max_lr', self.optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group['initial_lr'] = max_lrs[idx] / div_factor
                group['max_lr'] = max_lrs[idx]
                group['min_lr'] = group['initial_lr'] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if 'momentum' not in self.optimizer.defaults and 'betas' not in self.optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')
            self.use_beta1 = 'betas' in self.optimizer.defaults
            max_momentums = self._format_param('max_momentum', optimizer, max_momentum)
            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (m_momentum, beta2)
                    else:
                        group['momentum'] = m_momentum
                    group['max_momentum'] = m_momentum
                    group['base_momentum'] = b_momentum

        super(CustomOneCycleLR, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        if pct >= 1.0:
            return end
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        if pct >= 1.0:
            return end
        return (end - start) * pct + start

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        lrs = []
        step_num = self.last_epoch
        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))
        for group in self.optimizer.param_groups:
            if step_num <= self.step_size_up:
                computed_lr = self.anneal_func(group['initial_lr'], group['max_lr'], step_num / self.step_size_up)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['max_momentum'], group['base_momentum'],
                                                         step_num / self.step_size_up)
            else:
                down_step_num = step_num - self.step_size_up
                computed_lr = self.anneal_func(group['max_lr'], group['min_lr'], down_step_num / self.step_size_down)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['base_momentum'], group['max_momentum'],
                                                         down_step_num / self.step_size_down)
            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (computed_momentum, beta2)
                else:
                    group['momentum'] = computed_momentum
        return lrs

def get_data(
        ssX=None,
        batch_size=32,
        train=True,
        **kwargs):
    """
    inputs:
        batch_size: int

    return:
        (dataloader, test_dataloader)
    """
    plot_random = False if 'plot_random' not in kwargs else kwargs['plot_random']
    plot_resonant = not plot_random
    train_all = False if 'train_all' not in kwargs else kwargs['train_all']
    plot = False if 'plot' not in kwargs else kwargs['plot']
    if not train_all and ssX is None:
        plot_resonant = True
        plot_random = False

    if train_all:
        filename = 'data/combined.pkl'
    elif plot_resonant:
        filename = 'data/resonant_dataset.pkl'
    elif plot_random:
        filename = 'data/random_dataset.pkl'

    # These are generated by data_from_pkl.py
    loaded_data = pkl.load(
        open(filename, 'rb')
    )

    train_ssX = (ssX is None)

    fullX, fully = loaded_data['X'], loaded_data['y']

    if train_all:
        len_random = 17082 #Number of valid random examples (others have NaNs)
        random_data = np.arange(len(fullX)) >= (len(fullX) - len_random)


    # Differentiate megno
    if 'fix_megno' in kwargs and kwargs['fix_megno']:
        idx = [i for i, lab in enumerate(loaded_data['labels']) if 'megno' in lab][0]
        fullX[:, 1:, idx] -= fullX[:, :-1, idx]

    if 'include_derivatives' in kwargs and kwargs['include_derivatives']:
        derivative = fullX[:, 1:, :] - fullX[:, :-1, :]
        derivative = np.concatenate((
            derivative[:, [0], :],
            derivative), axis=1)
        fullX = np.concatenate((
            fullX, derivative),
            axis=2)


    # Hide fraction of test
    # MAKE SURE WE DO COPIES AFTER!!!!
    if train:
        if train_all:
            remy, finaly, remX, finalX, rem_random, final_random = train_test_split(fully, fullX, random_data, shuffle=True, test_size=1./10, random_state=0)
            trainy, testy, trainX, testX, train_random, test_random = train_test_split(remy, remX, rem_random, shuffle=True, test_size=1./10, random_state=1)
        else:
            remy, finaly, remX, finalX = train_test_split(fully, fullX, shuffle=True, test_size=1./10, random_state=0)
            trainy, testy, trainX, testX = train_test_split(remy, remX, shuffle=True, test_size=1./10, random_state=1)
    else:
        assert not train_all
        remy = fully
        finaly = fully
        testy = fully
        trainy = fully
        remX = fullX
        finalX = fullX
        testX = fullX
        trainX = fullX

    if plot:
        # Use test dataset for plotting, so put it in validation part:
        testX = finalX
        testy = finaly

    if train_ssX:
        if 'power_transform' in kwargs and kwargs['power_transform']:
            ssX = PowerTransformer(method='yeo-johnson') #Power is best
        else:
            ssX = StandardScaler() #Power is best

    n_t = trainX.shape[1]
    n_features = trainX.shape[2]

    needs_training = train_ssX or 'train_ssX' in kwargs and kwargs['train_ssX']
    if needs_training:
        ssX.fit(trainX.reshape(-1, n_features)[::1539])

    ttrainy = trainy
    ttesty = testy
    ttrainX = ssX.transform(trainX.reshape(-1, n_features)).reshape(-1, n_t, n_features)
    ttestX = ssX.transform(testX.reshape(-1, n_features)).reshape(-1, n_t, n_features)
    if train_all:
        ttest_random = test_random
        ttrain_random = train_random

    tremX = ssX.transform(remX.reshape(-1, n_features)).reshape(-1, n_t, n_features)
    tremy = remy

    train_len = ttrainX.shape[0]
    X = Variable(torch.from_numpy(np.concatenate((ttrainX, ttestX))).type(torch.FloatTensor))
    y = Variable(torch.from_numpy(np.concatenate((ttrainy, ttesty))).type(torch.FloatTensor))
    if train_all:
        r = Variable(torch.from_numpy(np.concatenate((ttrain_random, ttest_random))).type(torch.BoolTensor))

    Xrem = Variable(torch.from_numpy(tremX).type(torch.FloatTensor))
    yrem = Variable(torch.from_numpy(tremy).type(torch.FloatTensor))

    idxes = np.s_[:]
    dataset = torch.utils.data.TensorDataset(X[:train_len, :, idxes], y[:train_len])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

    # Cut up dataset into only the random or resonant parts.
    # Only needed if plotting OR
    if (not plot) or (not train_all):
        test_dataset = torch.utils.data.TensorDataset(X[train_len:, :, idxes], y[train_len:])
    else:
        if plot_random: mask =  r
        else:           mask = ~r
        print(f'Plotting with {mask.sum()} total elements, when plot_random={plot_random}')
        test_dataset = torch.utils.data.TensorDataset(X[train_len:][r[train_len:]][:, :, idxes], y[train_len:][r[train_len:]])

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=3000, shuffle=False, pin_memory=True, num_workers=8)

    kwargs['model'].ssX = copy(ssX)

    return dataloader, test_dataloader


def soft_clamp(x, lo, high):
    return 0.5*(torch.tanh(x)+1)*(high-lo) + lo

def hard_clamp(x, lo, high):
    return torch.max(torch.min(x, torch.tensor(high)), torch.tensor(lo))


def safe_log_erf(x):
    base_mask = x < -1
    value_giving_zero = torch.zeros_like(x, device=x.device)
    x_under = torch.where(base_mask, x, value_giving_zero)
    x_over = torch.where(~base_mask, x, value_giving_zero)

    f_under = lambda x: (
         0.485660082730562*x + 0.643278438654541*torch.exp(x) +
         0.00200084619923262*x**3 - 0.643250926022749 - 0.955350621183745*x**2
    )
    f_over = lambda x: torch.log(1.0+torch.erf(x))

    return f_under(x_under) + f_over(x_over)

EPSILON = 1e-5

class VarModel(pl.LightningModule):
    """Bayesian Neural Network model for predicting instability time"""
    def __init__(self, hparams):
        # so we can load old runs before the variable names were changed
        if 'hidden_dim' not in hparams:
            hparams['hidden_dim'] = hparams['hidden']
        if 'f2_depth' not in hparams:
            hparams['f2_depth'] = hparams['out']
        if 'f1_depth' not in hparams:
            hparams['f1_depth'] = hparams['in']

        super().__init__()
        if 'seed' not in hparams: hparams['seed'] = 0
        pl.seed_everything(hparams['seed'])

        hparams['include_derivatives'] = False if 'include_derivatives' not in hparams else hparams['include_derivatives']

        if 'time_series_features' not in hparams:
            hparams['time_series_features'] = 38+3

        if hparams['time_series_features'] == 82:
            hparams['time_series_features'] = 41

        self.fix_megno = False if 'fix_megno' not in hparams else hparams['fix_megno']
        self.fix_megno2 = False if 'fix_megno2' not in hparams else hparams['fix_megno2']
        self.include_angles = False if 'include_angles' not in hparams else hparams['include_angles']

        self.n_features = hparams['time_series_features'] * (1 + int(hparams['include_derivatives']))

        if 'combined_mass_feature' in hparams and hparams['combined_mass_feature']:
            self.n_features += 1

        self.l1_reg_inputs = 'l1_reg' in hparams and hparams['l1_reg'] == 'inputs'
        self.l1_reg_weights = 'l1_reg' in hparams and hparams['l1_reg'] in ['weights', 'both_weights']
        self.l1_reg_f2_weights = 'l1_reg' in hparams and hparams['l1_reg'] in ['f2_weights', 'both_weights']

        # self.feature_nn_out_dim is usually hparams['latent'], unless we're using a weird f1 variant (such as taking products of input features)
        self.feature_nn, self.feature_nn_out_dim = self.get_feature_nn(hparams)

        summary_dim = self.feature_nn_out_dim * 2

        if (('no_std' in hparams and hparams['no_std'])
            or ('no_mean' in hparams and hparams['no_mean'])):
            summary_dim = self.feature_nn_out_dim

        if self.fix_megno:
            summary_dim += 2

        # sometimes the f2 config changes the summary dim
        self.regress_nn, self.summary_dim = self.get_regress_nn(hparams, summary_dim)

        self.input_noise_logvar = nn.Parameter(torch.zeros(self.n_features)-2)
        self.summary_noise_logvar = nn.Parameter(torch.zeros(summary_dim) - 2) # add to summaries, not direct latents

        self.lowest = 0.5
        if 'lower_std' in hparams and hparams['lower_std']:
            self.lowest = 0.1

        self.latents = None
        self.beta_in = 1 if 'beta_in' not in hparams else hparams['beta_in']
        self.beta_out = 1 if 'beta_out' not in hparams else hparams['beta_out']
        self.megno_location = 7
        self.mmr_location = [3, 6]
        self.nan_location = [38, 39, 40]
        self.eplusminus_location = [1, 2, 4, 5]

        # SWA params
        hparams['scheduler_choice'] = 'swa' #'cycle' if 'scheduler_choice' not in hparams else hparams['scheduler_choice']
        hparams['save_freq'] = 25 if 'save_freq' not in hparams else hparams['save_freq']
        hparams['eval_freq'] = 5 if 'eval_freq' not in hparams else hparams['eval_freq']
        hparams['momentum'] = 0.9 if 'momentum' not in hparams else hparams['momentum']
        hparams['weight_decay'] = 1e-4 if 'weight_decay' not in hparams else hparams['weight_decay']
        hparams['noisy_val'] = True if 'noisy_val' not in hparams else hparams['noisy_val']

        # self.hparams = hparams # this should be automatically done by the next line
        self.save_hyperparameters(hparams)
        self.steps = hparams['steps']
        self.batch_size = hparams['batch_size']
        self.lr = hparams['lr'] #init_lr
        self._dataloader = None
        self._val_dataloader = None
        self.random_sample = False if 'random_sample' not in hparams else hparams['random_sample']
        self.train_len = 78660
        self.test_len = 8740
        self._summary_kl = 0.0
        self.include_mmr = hparams['include_mmr']
        self.include_nan = hparams['include_nan']
        self.include_eplusminus = True if 'include_eplusminus' not in hparams else hparams['include_eplusminus']
        self.train_all = False if 'train_all' not in hparams else hparams['train_all']
        self.mse = hparams['mse_loss'] if 'mse_loss' in hparams else False

        self._cur_summary = None

        self.ssX = None
        self.ssy = None

        if 'eval' in hparams and hparams['eval']:
            self.disable_optimization()

        if 'freeze_f1' in hparams and hparams['freeze_f1']:
            utils.freeze_module(self.feature_nn)
            self.input_noise_logvar.requires_grad = False
            self.summary_noise_logvar.requires_grad = False

        if 'freeze_f2' in hparams and hparams['freeze_f2']:
            utils.freeze_module(self.regress_nn)

        if 'predict_eq_uncertainty' in hparams and hparams['predict_eq_uncertainty']:
            # load old model + pysr equations, but hide it so it's not a submodule
            model = load_with_pysr_f2(version=24880, pysr_version=11003, pysr_model_selection='accuracy')
            model = model.cuda()
            self.eq_model = [model]
            for param in self.eq_model[0].parameters():
                param.requires_grad = False


    def path(self):
        version = self.hparams['version']
        if 'eval' in self.hparams and self.hparams['eval']:
            if 'load_f1' in self.hparams and self.hparams['load_f1']:
                version =  self.hparams['load_f1']
            if 'load_f1_f2' in self.hparams and self.hparams['load_f1_f2']:
                version =  self.hparams['load_f1_f2']

        path = f'{version}_{self.hparams["seed"]}'

        if self.hparams['pysr_f2'] or type(self.regress_nn) == modules.PySRNet:
            # go from 'sr_results/11003.pkl' to 11003 by extracting the number from it
            pysr_version = int(''.join(filter(str.isdigit, self.regress_nn.filepath)))
            path += f'_pysr_f2_v={pysr_version}'
            path += f'_ms={self.regress_nn.model_selection}'

        return path


    def get_regress_nn(self, hparams, summary_dim):
        '''
        summary dim is
            - the input dim of the regress nn
            - the dimension of the summary statistics
            - typically two times the output dim of the f1 network
        '''

        if 'f2_variant' not in hparams:
            hparams['f2_variant'] = 'mlp'

        load_version = None
        if 'load_f2' in hparams and hparams['load_f2']:
            load_version = hparams['load_f2']
        elif 'load_f1_f2' in hparams and hparams['load_f1_f2']:
            load_version = hparams['load_f1_f2']
        if load_version is not None:
            model = load(load_version)
            regress_nn = model.regress_nn
            summary_dim = model.summary_dim

            if 'prune_f2_topk' in hparams and hparams['prune_f2_topk'] is not None:
                assert isinstance(regress_nn, nn.Linear)
                regress_nn = modules.pruned_linear(regress_nn, top_k=hparams['prune_f2_topk'])

        elif hparams['f1_variant'] == 'mean_cov':
            i = self.n_features
            if 'mean_var' in hparams and hparams['mean_var']:
                summary_dim = i + i
                regress_nn = modules.mlp(summary_dim, 2, hparams['hidden_dim'], hparams['f2_depth'])
            else:
                summary_dim = i + i*i
                # in: n_inputs + n_inputs * n_inputs, out: 2 * n_features
                special_linear = modules.SpecialLinear(n_inputs=self.n_features, n_features=hparams['latent'],
                                                       init=hparams['init_special'])
                utils.freeze_module(special_linear)
                regress_nn = nn.Sequential(special_linear,
                                     nn.BatchNorm1d(summary_dim),
                                     modules.mlp(summary_dim, 2, hparams['hidden_dim'], hparams['f2_depth']))
        elif hparams['f2_variant'] == 'ifthen':
            regress_nn = modules.IfThenNN(hparams['n_predicates'], summary_dim, 2, hparams['hidden_dim'], hparams['f2_depth'])
        elif hparams['f2_variant'] == 'ifthen2':
            regress_nn = modules.IfThenNN2(hparams['n_predicates'], summary_dim, 2, hparams['hidden_dim'], hparams['f2_depth'])
        elif hparams['f2_variant'] == 'linear':
            regress_nn = nn.Linear(summary_dim, 2)
        elif hparams['f2_variant'] == 'bimt':
            regress_nn = BioMLP(in_dim=summary_dim, depth=hparams['f2_depth']+2, w=hparams['hidden_dim'], out_dim=2)
        elif 'pysr_f2' in hparams and hparams['pysr_f2']:
            regress_nn = modules.PySRNet(hparams['pysr_f2'], hparams['pysr_f2_model_selection'])
            if len(regress_nn.module_list) == 1:
                # pysr only predicts mean. predict std using NN loaded for f1, or a new network
                print('PySR only predicts mean. Adding a new network to predict std.')
                if 'load_f1' in hparams and hparams['load_f1']:
                    print('Using --load_f1 network regress_nn to predict std')
                    assert 0, 'plz debug this before using it, something seems off'
                    base_f2 = load(hparams['load_f1']).regress_nn
                    regress_nn = base_f2
                    # base_f2 = modules.PySRNet('sr_results/29741.pkl', 'best')
                    utils.freeze_module(base_f2)
                else:
                    print('Initializing new network to predict std')
                    base_f2 = modules.mlp(summary_dim, 2, hparams['hidden_dim'], hparams['f2_depth'])
                assert 0, 'need to debug this part too; does addstdprednn do the shapes right?'
                regress_nn = modules.AddStdPredNN(regress_nn, base_f2)

            if 'nn_pred_std' in hparams and hparams['nn_pred_std']:
                print('Using a neural network to predict std')
                std_nn = modules.mlp(summary_dim, 2, hparams['hidden_dim'], hparams['f2_depth'])
                # regress_nn is the pysr module
                utils.freeze_module(regress_nn)
                regress_nn = modules.AddStdPredNN(regress_nn, std_nn)
        else:
            regress_nn = modules.mlp(summary_dim, 2, hparams['hidden_dim'], hparams['f2_depth'])

        def calculate_additional_features(summary_stats, previous_sr_model_path):
            # Move the tensor to CPU before converting to numpy
            summary_stats_np = summary_stats.cpu().detach().numpy()

            with open(previous_sr_model_path, 'rb') as f:
                previous_sr_model = pkl.load(f)

            additional_features = []

            for equation_set in previous_sr_model.equations_:
                for index, equation in equation_set.iterrows():
                    lambda_func = equation['lambda_format']
                    evaluated_result = lambda_func(summary_stats_np)
                    # Ensure the result is reshaped to match the batch size
                    evaluated_result = evaluated_result.reshape(-1, 1)
                    additional_features.append(evaluated_result)

            additional_features = np.hstack(additional_features)
            additional_features_tensor = torch.tensor(additional_features, dtype=summary_stats.dtype, device=summary_stats.device)
            return torch.cat([summary_stats, additional_features_tensor], dim=-1)

        if 'f2_residual' in hparams and hparams['f2_residual'] or 'pysr_f2_residual' in hparams and hparams['pysr_f2_residual']:
            if 'pysr_f2' in hparams and hparams['pysr_f2']:
                utils.freeze_module(regress_nn)

            if hparams['f2_residual'] == 'mlp':
                residual_net = modules.mlp(summary_dim, 2, hparams['hidden_dim'], hparams['f2_depth'])
            else:
                # assert hparams['f2_residual'] == 'pysr'
                residual_net = modules.PySRNet(hparams['pysr_f2_residual'], hparams['pysr_f2_residual_model_selection'])

                if len(residual_net.module_list) == 1:
                    # pysr only predicts mean. predict std using NN loaded for f1, or a new network
                    print('PySR only predicts mean. Adding a new network to predict std.')
                    if 'load_f1' in hparams and hparams['load_f1']:
                        print('Using --load_f1 network regress_nn to predict std')
                        base_f2 = load(hparams['load_f1']).regress_nn
                    else:
                        print('Initializing new network to predict std')
                        base_f2 = modules.mlp(summary_dim, 2, hparams['hidden_dim'], hparams['f2_depth'])
                    assert 0, 'need to fix this commented out code'
                    # residual_net = modules.PySRRegressNN(residual_net, base_f2)

            if 'pysr_f2_residual' in hparams and hparams['pysr_f2_residual']:
                def combined_predict_instability(summary_stats):
                    summary_stats_with_additional = calculate_additional_features(summary_stats, hparams['pysr_f2_residual'])
                    return residual_net(summary_stats_with_additional)

                regress_nn = modules.SumModule(regress_nn, combined_predict_instability)
            else:
                regress_nn = modules.SumModule(regress_nn, residual_net)

        return regress_nn, summary_dim


    def get_feature_nn(self, hparams):
        feature_nn = None
        out_dim = hparams['latent'] # certain f1 variants might change this

        if 'load_f1_feature_nn' in hparams and hparams['load_f1_feature_nn']:
            feature_nn = torch.load(hparams['load_f1_feature_nn'], seed=0)
            out_dim = feature_nn.linear.weight.shape[0]
        elif any(p in hparams and hparams[p] for p in ['load_f1', 'load_f1_f2']):
            load_version = None
            if 'load_f1' in hparams and hparams['load_f1']:
                load_version = hparams['load_f1']
            else:
                assert 'load_f1_f2' in hparams and hparams['load_f1_f2']
                load_version = hparams['load_f1_f2']
            model = load(load_version)
            feature_nn = model.feature_nn
            out_dim = model.feature_nn_out_dim
            if 'prune_f1_topk' in hparams and hparams['prune_f1_topk'] is not None:
                # hack in case f1 was a weird variant, like products2
                if isinstance(feature_nn, nn.Sequential):
                    if isinstance(feature_nn[1], nn.Linear):
                        feature_nn = modules.pruned_linear(feature_nn, top_k=hparams['prune_f1_topk'])
                    elif isinstance(feature_nn[1], nn.Identity):
                        feature_nn = nn.Sequential(
                            modules.pruned_input_mask(feature_nn[0], top_k=hparams['prune_f1_topk']),
                            nn.Identity())

                elif isinstance(feature_nn, nn.Linear):
                    feature_nn = modules.pruned_linear(feature_nn, top_k=hparams['prune_f1_topk'])

        elif 'pysr_f1' in hparams and hparams['pysr_f1']:
            # constants can still be optimized with SGD
            feature_nn = modules.PySRFeatureNN(hparams['pysr_f1'], model_selection=hparams['pysr_f1_model_selection'])
            out_dim = len(feature_nn.module_list)
        elif hparams['f1_variant'] == 'random_features':
            feature_nn = modules.RandomFeatureNN(in_n=self.n_features, out_n=hparams['latent'])
        elif hparams['f1_variant'] == 'identity':
            feature_nn = torch.nn.Identity()
            out_dim = self.n_features
        elif hparams['f1_variant'] == 'zero':
            feature_nn = modules.ZeroNN(in_n=self.n_features, out_n=hparams['latent'])
        elif hparams['f1_variant'] == 'linear':
            feature_nn = nn.Linear(self.n_features, hparams['latent'], bias=False)
            # feature_nn = nn.Linear(self.n_features, hparams['latent'], bias=True) # if loading 21101
        elif hparams['f1_variant'] == 'bimt':
            feature_nn = BioMLP(in_dim=self.n_features, out_dim=hparams['latent'])
        elif hparams['f1_variant'] == 'biolinear':
            feature_nn = BioLinear(in_dim=self.n_features, out_dim=hparams['latent'])
        elif hparams['f1_variant'] == 'mean_cov':
            feature_nn = None  # calc handled without feature nn
        elif hparams['f1_variant'] == 'products':
            linear = nn.Linear(self.n_features * self.n_features, hparams['latent'],
                               bias='no_bias' not in hparams or not hparams['no_bias'])
            feature_nn = nn.Sequential(modules.Products(), linear)
        elif hparams['f1_variant'] == 'products2':
            products_net = modules.Products2()
            linear = nn.Linear(self.n_features + len(products_net.products), hparams['latent'],
                               bias='no_bias' not in hparams or not hparams['no_bias'])
            feature_nn = nn.Sequential(products_net, linear)
        elif hparams['f1_variant'] == 'products3':
            products_net = modules.Products3()
            linear = nn.Linear(self.n_features + len(products_net.products), hparams['latent'],
                               bias='no_bias' not in hparams or not hparams['no_bias'])
            feature_nn = nn.Sequential(products_net, linear)
        elif hparams['f1_variant'] == 'random':
            feature_nn = nn.Linear(self.n_features, hparams['latent'], bias='no_bias' not in hparams or not hparams['no_bias'])
            # make the linear projection random combinations of two input variables, with coefficients from U[-1, 1]
            weight = torch.zeros_like(feature_nn.weight)
            for i in range(weight.shape[0]):
                weight[i, np.random.choice(self.n_features, 2, replace=False)] = torch.rand(2) * 2 - 1

            feature_nn.weight = torch.nn.Parameter(weight)
            feature_nn = modules.pruned_linear(feature_nn, k=2)
        else:
            assert hparams['f1_variant'] == 'mlp'
            feature_nn = modules.mlp(self.n_features, hparams['latent'], hparams['hidden_dim'], hparams['f1_depth'])

        if self.l1_reg_inputs:
            self.inputs_mask = modules.MaskLayer(self.n_features)
            feature_nn = torch.nn.Sequential(self.inputs_mask,
                                             feature_nn)

        return feature_nn, out_dim

    def do_nothing_optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        # Override this method to adjust the optimizer's behavior.
        pass  # Doing nothing means gradients are not updated.

    def disable_optimization(self):
        self.optimizer_step = self.do_nothing_optimizer_step


    def augment(self, x):
        # This randomly samples times.
        samples = np.random.randint(self.hparams['samp'], x.shape[1]+1)
        x = x[:, np.random.randint(0, x.shape[1], size=samples)]
        return x

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def compute_summary_stats2(self, x):
        # no sampling
        x = self.feature_nn(x)
        sample_mu = torch.mean(x, dim=1, keepdim=True)
        sample_var = torch.std(x, dim=1, keepdim=True)**2
        sample_std = torch.sqrt(torch.abs(sample_var) + EPSILON)

        # clatent = torch.cat((sample_mu, sample_var), dim=1)
        clatent = torch.cat((sample_mu, sample_std), dim=-1)
        self.latents = x

        return clatent

    def compute_mean_cov_stats(self, x):
        # instead of passing x through feature nn, directly calculate the mean and covariance of x
        # for now, ignore the sampling aspect - talk to miles about it
        # x: [B, T, d]
        # mean: [B, d]
        # covariance: [B, d, d]
        B, T, d = x.shape
        self.latents = x

        mean = x.mean(dim=1)
        cov = utils.batch_cov(x)

        if 'no_summary_sample' in self.hparams and self.hparams['no_summary_sample']:
            if 'mean_var' in self.hparams and self.hparams['mean_var']:
                cov = cov[:, torch.arange(d), torch.arange(d)]
            else:
                cov = einops.rearrange(cov, 'B D1 D2 -> B (D1 D2)')  # flatten
            return torch.cat((mean, cov), dim=1)

        # sample!
        n = x.shape[1]
        var = torch.std(x, dim=1)**2
        std = torch.sqrt(torch.abs(var) + EPSILON)

        std_of_mean = torch.sqrt(var/n)
        # variance of covariance: https://stats.stackexchange.com/a/287241/215833
        # var = 1/(n-1) * (1 + rho^2) sigma^2_i sigma^2_j
        # rho = cov / (sigma_i sigma_j)
        rho = cov / (std[:, None, :] * std[:, :, None])
        std_of_cov = torch.sqrt(1/(n-1) * (1 + rho*rho ) * var[:, None, :] * var[:, :, None])

        # Take a "sample" of the average/variance of the learned features
        mean_sample =  torch.randn_like(mean) *std_of_mean  + mean
        cov_sample = torch.randn_like(cov)*std_of_cov + cov

        if 'mean_var' in self.hparams and self.hparams['mean_var']:
            # only returning the variances
            cov_sample = cov_sample[:, torch.arange(d), torch.arange(d)]
        else:
            cov_sample = einops.rearrange(cov_sample, 'B D1 D2 -> B (D1 D2)')  # flatten
        clatent = torch.cat((mean_sample, cov_sample), dim=1)

        # Change to correlation to be unitless?
        # corr_sample = cov_sample / sigma_i_sigma_j
        # corr_sample = einops.rearrange(corr_sample, 'B D1 D2 -> B (D1 D2)')  # flatten
        # clatent = torch.cat((mean_sample, corr_sample), dim=1)

        return clatent


    def compute_summary_stats(self, x, deterministic=False):
        if self.hparams['f1_variant'] == 'mean_cov':
            # no f1 needed, take the inputs straight
            return self.compute_mean_cov_stats(x)

        if 'no_summary_sample' in self.hparams and self.hparams['no_summary_sample']:
            return self.compute_summary_stats2(x)

        x = self.feature_nn(x)

        sample_mu = torch.mean(x, dim=1)
        sample_var = torch.std(x, dim=1)**2
        n = x.shape[1]

        std_in_mu = torch.sqrt(sample_var/n)
        std_in_var = torch.sqrt(2*sample_var**2/(n-1))

        # Take a "sample" of the average/variance of the learned features
        mu_sample =  torch.randn_like(sample_mu) *std_in_mu  + sample_mu
        var_sample = torch.randn_like(sample_var)*std_in_var + sample_var

        # if deterministic, just use the mean mu/vawr
        if deterministic or 'determinisic_summary_stats' in self.hparams and self.hparams['deterministic_summary_stats']:
            mu_sample = sample_mu
            var_sample = sample_var

        # Get to same unit
        std_sample = torch.sqrt(torch.abs(var_sample) + EPSILON)

        if 'no_std' in self.hparams and self.hparams['no_std']:
            clatent = mu_sample
        elif 'no_mean' in self.hparams and self.hparams['no_mean']:
            clatent = std_sample
        else:
            #clatent = torch.cat((mu_sample, var_sample), dim=1)
            clatent = torch.cat((mu_sample, std_sample), dim=1)

        self.latents = x

        return clatent

    def predict_instability(self, summary_stats):
        testy = self.regress_nn(summary_stats)

        if type(self.regress_nn) in [modules.MaskedLinear]:
            # no clamp
            mu = testy[:, [0]]
            std = testy[:, [1]]
        if type(self.regress_nn) in [modules.PySRNet, modules.PySREQBoundsNet, modules.DirectPySRNet]:
            mu = testy[:, [0]]
            std = testy[:, [1]]
            mu = hard_clamp(mu, 4.0, 12.0)
            std = hard_clamp(std, self.lowest, 6.0)
        else:
            mu = soft_clamp(testy[:, [0]], 4.0, 12.0)
            std = soft_clamp(testy[:, [1]], self.lowest, 6.0)

        return mu, std

    def add_input_noise(self, x):
        noise = torch.randn_like(x, device=self.device) * torch.exp(self.input_noise_logvar[None, None, :]/2)
        return x + noise

    def add_summary_noise(self, summary_stats):
        noise = torch.randn_like(summary_stats, device=self.device) * torch.exp(self.summary_noise_logvar[None, :]/2)
        return summary_stats + noise

    def zero_megno(self, x):
        with torch.no_grad():
            mask = torch.zeros_like(x)
            mask[..., self.megno_location] = x[..., self.megno_location].clone()
            x = x - mask
        return x

    def add_combined_mass_feature(self, x):
        with torch.no_grad():
            m1_ix, m2_ix, m3_ix = 35, 36, 37
            combined_mass = x[..., m1_ix] + x[..., m2_ix] + x[..., m3_ix]
            x = torch.cat([x, combined_mass.unsqueeze(-1)], dim=-1)
        return x

    def zero_theta(self, x):
        theta_locations = [15, 16, 24, 25, 33, 34]
        for ix in self.hparams['zero_theta']:
            assert 1 <= ix <= 6
            loc = theta_locations[ix-1]
            with torch.no_grad():
                mask = torch.zeros_like(x)
                mask[..., loc] = x[..., loc].clone()
                x = x - mask
        return x

    def zero_mmr(self, x):
        with torch.no_grad():
            mask = torch.zeros_like(x)
            mask[..., self.mmr_location] = x[..., self.mmr_location].clone()
            x = x - mask
        return x

    def zero_nan(self, x):
        with torch.no_grad():
            mask = torch.zeros_like(x)
            mask[..., self.nan_location] = x[..., self.nan_location].clone()
            x = x - mask
        return x

    def zero_eplusminus(self, x):
        with torch.no_grad():
            mask = torch.zeros_like(x)
            mask[..., self.eplusminus_location] = x[..., self.eplusminus_location].clone()
            x = x - mask
        return x

    def summarize_megno(self, x):
        megno_avg = torch.mean(x[:, :, [self.megno_location]], 1)
        megno_std = torch.std(x[:, :, [self.megno_location]], 1)

        return torch.cat([megno_avg, megno_std], dim=1)

    def forward(self, x, noisy_val=True, return_intermediates=False, deterministic=False):
        '''
        noisy_val: if True, add noise to the inputs and summaries
        '''
        assert x is not None
        if self.fix_megno or self.fix_megno2:
            if self.fix_megno:
                megno_avg_std = self.summarize_megno(x)
            #(batch, 2)
            x = self.zero_megno(x)

        if not self.include_mmr:
            x = self.zero_mmr(x)

        if not self.include_nan:
            x = self.zero_nan(x)

        if not self.include_eplusminus:
            x = self.zero_eplusminus(x)

        if 'zero_theta' in self.hparams and self.hparams['zero_theta'] != 0:
            x = self.zero_theta(x)

        if 'combined_mass_feature' in self.hparams and self.hparams['combined_mass_feature']:
            x = self.add_combined_mass_feature(x)

        if self.random_sample:
            x = self.augment(x)
        #x is (batch, time, feature)
        if noisy_val:
            x = self.add_input_noise(x)

        summary_stats = self.compute_summary_stats(x, deterministic=deterministic)

        if self.fix_megno:
            summary_stats = torch.cat([summary_stats, megno_avg_std], dim=1)

        self._cur_summary = summary_stats

        utils.assert_equal(self.summary_noise_logvar.shape[0], summary_stats.shape[1])

        #summary is (batch, feature)
        self._summary_kl = (1/2) * (
                summary_stats**2
                + torch.exp(self.summary_noise_logvar)[None, :]
                - self.summary_noise_logvar[None, :]
                - 1
            )

        if noisy_val:
            summary_stats = self.add_summary_noise(summary_stats)

        mu, std = self.predict_instability(summary_stats)
        #Each is (batch,)

        pred = torch.cat((mu, std), dim=1)

        if return_intermediates:
            # useful for running pysr to distill parts of the calculation
            d = {
                'inputs': x,
                # 'f1_output': self.feature_nn(x),  # note: should use summary stats usually!
                'summary_stats': summary_stats,
                'prediction': pred,
                'mean': mu,
                'std': std,
            }
            if 'ifthen' in self.hparams['f2_variant']:
                d['ifthen_preds'] = self.regress_nn.preds(summary_stats)
            return d

        return pred

    def sample(self, x, samples=10):
        all_samp = []
        init_settings = [self.random_sample, self.device]
        self.cpu()
        x = x.cpu()
        self.random_sample = False
        for _ in range(samples):
            out = self(x).detach().numpy()
            mu = out[:, 0]
            std = out[:, 1]
            all_samp.append(
                mu + np.random.randn(len(out))*std
            )
        self.random_sample = init_settings[0]
        self.to(init_settings[1])
        return np.average(all_samp, axis=0)

    def _lossfnc(self, testy, y):
        # y: [B, 2] batch of ground truth means. each input system has two
        #  simulations, one with a small initial perturbation, so the two means
        #  are samples from the distribution of instability times for that
        #  initial system
        # so we just sum over the loss for both of them.
        mu = testy[:, [0]]
        std = testy[:, [1]]

        var = std**2
        if 'fix_variance' in self.hparams and self.hparams['fix_variance']:
            var = torch.ones_like(var)

        t_greater_9 = y >= 9

        regression_loss = -(y - mu)**2/(2*var)
        regression_loss += -torch.log(std)

        regression_loss += -safe_log_erf(
                    (mu - 4)/(torch.sqrt(2*var))
                )

        classifier_loss = safe_log_erf(
                    (mu - 9)/(torch.sqrt(2*var))
            )

        safe_regression_loss = torch.where(
                ~torch.isfinite(regression_loss),
                -torch.ones_like(regression_loss)*100,
                regression_loss)
        safe_classifier_loss = torch.where(
                ~torch.isfinite(classifier_loss),
                -torch.ones_like(classifier_loss)*100,
                classifier_loss)

        total_loss = (
            safe_regression_loss * (~t_greater_9) +
            safe_classifier_loss * ( t_greater_9)
        )

        return -total_loss.sum(1)

    def mse_loss(self, testy, y):
        # y: [B, 2] batch of ground truth means. each input system has two
        #  simulations, one with a small initial perturbation, so the two means
        #  are samples from the distribution of instability times for that
        #  initial system
        # so we just sum over the loss for both of them.
        mu = testy[:, [0]]
        std = testy[:, [1]]

        # ignore std, and just compute mse with mu and y
        return torch.sum((mu - y)**2, dim=1)

    def lossfnc(self, x, y, samples=1, noisy_val=True, include_reg=True):

        # change the predicted mean to the eq_model's predicted mean
        # this way the nn learns to predict the std of the eq_model's predictions
        if 'predict_eq_uncertainty' in self.hparams and self.hparams['predict_eq_uncertainty']:

            testy = self(x, noisy_val=noisy_val)
            with torch.no_grad():
                eq_testy = self.eq_model[0](x, noisy_val=noisy_val)
                eq_testy[0, 0] = hard_clamp(eq_testy[0, 0], 4.0, 12.0)

                # sometimes the equations predict nan; get rid of those.
                # [B] tensor of indices of eq_testy that are nan
                nan_ixs = torch.isnan(eq_testy).any(dim=1)
                # exclude nan indices; we know those are highly uncertain already!
                x = x[~nan_ixs]
                y = y[~nan_ixs]
                testy = testy[~nan_ixs]
                eq_testy = eq_testy[~nan_ixs]

            # needs to be outside the no_grad otherwise pytorch complains
            testy[:, 0] = eq_testy[:, 0]
        else:
            testy = self(x, noisy_val=noisy_val)

        if self.mse:
            loss = self.mse_loss(testy, y).sum()
        else:
            loss = self._lossfnc(testy, y).sum()

        if include_reg:
            if self.l1_reg_inputs:
                loss = loss + self.hparams['l1_coeff'] * self.inputs_mask.l1_cost()
            if self.l1_reg_weights:
                l1_cost = sum([p.abs().sum() for p in self.feature_nn.parameters()])
                loss = loss + self.hparams['l1_coeff'] * l1_cost
            if self.l1_reg_f2_weights:
                l1_cost = sum([p.abs().sum() for p in self.regress_nn.parameters()])
                loss = loss + self.hparams['l1_coeff'] * l1_cost

        return loss

    def input_kl(self):
        return (1/2) * (
                torch.exp(self.input_noise_logvar)
                - self.input_noise_logvar
                - 1
            ).sum()

    def summary_kl(self):
        return self._summary_kl.sum()

    def training_step(self, batch, batch_idx):
        fraction = self.global_step / self.hparams['steps']
        beta_in = min([1, fraction/0.3]) * self.beta_in
        beta_out = min([1, fraction/0.3]) * self.beta_out

        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=True)
        #cur_frac = len(X_sample) / self.train_len

        # Want to be important with total number of samples
        input_kl = self.input_kl() * beta_in * len(X_sample)
        summary_kl = self.summary_kl() * beta_out

        prior = input_kl + summary_kl

        total_loss = loss + prior

        lamb = 0.001
        weight_factor = 1
        if self.hparams['f1_variant'] == 'bimt':
            reg = self.regress_nn.get_cc(bias_penalize=False, weight_factor=weight_factor)
            total_loss += lamb*reg
            if self.hparams['steps'] % 200 == 0:
                self.regress_nn.relocate()

        tensorboard_logs = {'train_loss_no_reg': loss/len(X_sample),
                            'train_loss_with_reg': total_loss/len(X_sample),
                            'input_kl': input_kl/len(X_sample),
                            'summary_kl': summary_kl/len(X_sample)}

        return {'loss': total_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=self.hparams['noisy_val'], include_reg=False)/self.test_len

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum()

        tensorboard_logs = {'val_loss_no_reg': avg_loss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt1 = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.hparams['momentum'], weight_decay=self.hparams['weight_decay'])

        assert self.hparams['scheduler_choice'] == 'swa'
        scheduler = CustomOneCycleLR(opt1, self.lr, int(0.9*self.steps), final_div_factor=1e4)
        interval = 'steps'
        name = 'swa_lr'

        sched1 = {
            'scheduler': scheduler,
            'name': name,
            'interval': interval
        }

        return [opt1], [sched1]

    def make_dataloaders(self, train=True, **extra_kwargs):
        kwargs = {
            **self.hparams,
            'model': self,
            **extra_kwargs,
            'train': train,
        }
        if 'ssX' in kwargs:
            dataloader, val_dataloader = get_data(**kwargs)
        else:
            dataloader, val_dataloader = get_data(ssX=self.ssX, **kwargs)

        labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']
        for i in range(len(labels)):
            label = labels[i]
            if not ('cos' in label or
                'sin' in label or
                'nan_' in label or
                label == 'i1' or
                label == 'i2' or
                label == 'i3'):
                continue

            if not self.include_angles:
                print('Tossing', i, label)
                dataloader.dataset.tensors[0][..., i] = 0.0
                val_dataloader.dataset.tensors[0][..., i] = 0.0

        self._dataloader = dataloader
        self._val_dataloader = val_dataloader
        self.train_len = len(dataloader.dataset.tensors[0])
        self.test_len = len(val_dataloader.dataset.tensors[0])

    def train_dataloader(self):
        if self._dataloader is None:
            self.make_dataloaders()
        return self._dataloader

    def val_dataloader(self):
        if self._val_dataloader is None:
            self.make_dataloaders()
        return self._val_dataloader


class SWAGModel(VarModel):
    """Use .load_from_checkpoint(checkpoint_path) to initialize a SWAG model"""
    def init_params(self, swa_params):
        self.swa_params = swa_params
        self.swa_params['swa_lr'] = 0.001 if 'swa_lr' not in self.swa_params else self.swa_params['swa_lr']
        self.swa_params['swa_start'] = 1000 if 'swa_start' not in self.swa_params else self.swa_params['swa_start']
        self.swa_params['swa_recording_lr_factor'] = 0.5 if 'swa_recording_lr_factor' not in self.swa_params else self.swa_params['swa_recording_lr_factor']

        self.n_models = 0
        self.w_avg = None
        self.w2_avg = None
        self.pre_D = None
        self.K = 20 if 'K' not in self.swa_params else self.swa_params['K']
        self.c = 2 if 'c' not in self.swa_params else self.swa_params['c']
        self.swa_params['c'] = self.c
        self.swa_params['K'] = self.K

        if 'eval' in swa_params and swa_params['eval']:
            self.disable_optimization()

        return self

    def configure_optimizers(self):
        opt1 = torch.optim.SGD(self.parameters(), lr=self.swa_params['swa_lr'], momentum=self.hparams['momentum'], weight_decay=self.hparams['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt1, [self.swa_params['swa_start']], self.swa_params['swa_recording_lr_factor'])
        interval = 'steps'
        name = 'swa_record_lr'
        sched1 = {
            'scheduler': scheduler,
            'name': name,
            'interval': interval
        }

        return [opt1], [sched1]

    def training_step(self, batch, batch_idx):
        beta_in = self.beta_in
        beta_out = self.beta_out
        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=True)
        input_kl = self.input_kl() * beta_in * len(X_sample)
        summary_kl = self.summary_kl() * beta_out
        prior = input_kl + summary_kl
        total_loss = loss + prior
        tensorboard_logs = {'train_loss_no_reg': loss/len(X_sample), 'train_loss_with_reg': total_loss/len(X_sample), 'input_kl': input_kl/len(X_sample), 'summary_kl': summary_kl/len(X_sample)}
        return {'loss': total_loss, 'log': tensorboard_logs}

    def flatten(self):
        """Convert state dict into a vector"""
        ps = self.state_dict()
        p_vec = None
        for key in ps.keys():
            p = ps[key]

            if p_vec is None:
                p_vec = p.reshape(-1)
            else:
                p_vec = torch.cat((p_vec, p.reshape(-1)))

        return p_vec

    def load(self, p_vec):
        """Load a vector into the state dict"""
        cur_state_dict = self.state_dict()
        new_state_dict = OrderedDict()
        i = 0
        for key in cur_state_dict.keys():
            old_p = cur_state_dict[key]
            size = old_p.numel()
            shape = old_p.shape
            new_p = p_vec[i:i+size]
            if len(shape) > 0:
                new_p = new_p.reshape(*shape)
            new_state_dict[key] = new_p
            i += size

        self.load_state_dict(new_state_dict)

    def aggregate_model(self):
        """Aggregate models for SWA/SWAG"""

        cur_w = self.flatten()
        cur_w2 = cur_w ** 2
        with torch.no_grad():
            if self.w_avg is None:
                self.w_avg = cur_w
                self.w2_avg = cur_w2
            else:
                self.w_avg = (self.w_avg * self.n_models + cur_w) / (self.n_models + 1)
                self.w2_avg = (self.w2_avg * self.n_models + cur_w2) / (self.n_models + 1)

            if self.pre_D is None:
                self.pre_D = cur_w.clone()[:, None]
            elif self.current_epoch % self.c == 0:
                #Record weights, measure discrepancy with average later
                self.pre_D = torch.cat((self.pre_D, cur_w[:, None]), dim=1)
                if self.pre_D.shape[1] > self.K:
                    self.pre_D = self.pre_D[:, 1:]


        self.n_models += 1

    def validation_step(self, batch, batch_idx):
        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=self.hparams['noisy_val'])/self.test_len

        if self.w_avg is None:
            swa_loss = loss
        else:
            tmp = self.flatten()
            self.load(self.w_avg)
            swa_loss = self.lossfnc(X_sample, y_sample, noisy_val=self.hparams['noisy_val'], include_reg=False)/self.test_len
            self.load(tmp)

        return {'val_loss': loss, 'swa_loss': swa_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum()
        swa_avg_loss = torch.stack([x['swa_loss'] for x in outputs]).sum()
        tensorboard_logs = {'val_loss_no_reg': avg_loss, 'swa_loss_no_reg': swa_avg_loss}
        #TODO: Check
        #fraction = self.global_step / self.hparams['steps']
        #if fraction > 0.5:
        if self.global_step > self.hparams['swa_start']:
            self.aggregate_model()

        # Record validation loss, and aggregated model loss
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_weights(self, scale=1):
        """Sample weights using SWAG:
        - w ~ N(avg_w, 1/2 * sigma + D . D^T/2(K-1))
            - This can be done with the following matrices:
                - z_1 ~ N(0, I_d); d the number of parameters
                - z_2 ~ N(0, I_K)
            - Then, compute:
            - w = avg_w + (1/sqrt(2)) * sigma^(1/2) . z_1 + D . z_2 / sqrt(2(K-1))
        """
        with torch.no_grad():
            avg_w = self.w_avg #[K]
            avg_w2 = self.w2_avg #[K]
            D = self.pre_D - avg_w[:, None]#[d, K]
            d = avg_w.shape[0]
            K = self.K
            z_1 = torch.randn((1, d), device=self.device)
            z_2 = torch.randn((K, 1), device=self.device)
            sigma = torch.abs(torch.diag(avg_w2 - avg_w**2))

            w = avg_w[None] + scale * (1.0/np.sqrt(2.0)) * z_1 @ sigma**0.5
            w += scale * (D @ z_2).T / np.sqrt(2*(K-1))
            w = w[0]

        self.load(w)

    def forward_swag(self, x, scale=0.5):
        """No augmentation happens here."""

        # Sample using SWAG using recorded model moments
        self.sample_weights(scale=scale)

        if self.fix_megno or self.fix_megno2:
            if self.fix_megno:
                megno_avg_std = self.summarize_megno(x)
            #(batch, 2)
            x = self.zero_megno(x)

        if not self.include_mmr:
            x = self.zero_mmr(x)

        if not self.include_nan:
            x = self.zero_nan(x)

        if not self.include_eplusminus:
            x = self.zero_eplusminus(x)

        summary_stats = self.compute_summary_stats(x)
        if self.fix_megno:
            summary_stats = torch.cat([summary_stats, megno_avg_std], dim=1)

        #summary is (batch, feature)
        self._summary_kl = (2/2) * (
                summary_stats**2
                + torch.exp(self.summary_noise_logvar)[None, :]
                - self.summary_noise_logvar[None, :]
                - 1
            )

        mu, std = self.predict_instability(summary_stats)
        #Each is (batch,)

        return torch.cat((mu, std), dim=1)

    def forward_swag_fast(self, x, scale=0.5):
        """No augmentation happens here."""

        # Sample using SWAG using recorded model moments
        self.sample_weights(scale=scale)

        if self.fix_megno or self.fix_megno2:
            if self.fix_megno:
                megno_avg_std = self.summarize_megno(x)
            #(batch, 2)
            x = self.zero_megno(x)

        if not self.include_mmr:
            x = self.zero_mmr(x)

        if not self.include_nan:
            x = self.zero_nan(x)

        if not self.include_eplusminus:
            x = self.zero_eplusminus(x)

        summary_stats = self.compute_summary_stats(x)
        if self.fix_megno:
            summary_stats = torch.cat([summary_stats, megno_avg_std], dim=1)

        #summary is (batch, feature)

        mu, std = self.predict_instability(summary_stats)
        #Each is (batch,)

        return torch.cat((mu, std), dim=1)


def save_swag(swag_model, path):
    save_items = {
        'hparams':swag_model.hparams,
        'swa_params': swag_model.swa_params,
        'w_avg': swag_model.w_avg.cpu(),
        'w2_avg': swag_model.w2_avg.cpu(),
        'pre_D': swag_model.pre_D.cpu()
    }

    torch.save(save_items, path)

def load_swag(path):
    save_items = torch.load(path)
    swag_model = (
        SWAGModel(save_items['hparams'])
        .init_params(save_items['swa_params'])
    )
    swag_model.w_avg = save_items['w_avg']
    swag_model.w2_avg = save_items['w2_avg']
    swag_model.pre_D = save_items['pre_D']
    if 'v50' in path:
        # Assume fixed scale:
        ssX = StandardScaler()
        ssX.scale_ = np.array([2.88976974e+03, 6.10019661e-02, 4.03849732e-02, 4.81638693e+01,
                   6.72583662e-02, 4.17939679e-02, 8.15995339e+00, 2.26871589e+01,
                   4.73612029e-03, 7.09223721e-02, 3.06455099e-02, 7.10726478e-01,
                   7.03392022e-01, 7.07873597e-01, 7.06030923e-01, 7.04728204e-01,
                   7.09420909e-01, 1.90740659e-01, 4.75502285e-02, 2.77188320e-02,
                   7.08891412e-01, 7.05214134e-01, 7.09786887e-01, 7.04371833e-01,
                   7.04371110e-01, 7.09828420e-01, 3.33589977e-01, 5.20857790e-02,
                   2.84763136e-02, 7.02210626e-01, 7.11815232e-01, 7.10512240e-01,
                   7.03646004e-01, 7.08017286e-01, 7.06162814e-01, 2.12569430e-05,
                   2.35019125e-05, 2.04211110e-05, 7.51048890e-02, 3.94254400e-01,
                   7.11351099e-02])
        ssX.mean_ = np.array([ 4.95458585e+03,  5.67411891e-02,  3.83176945e-02,  2.97223474e+00,
                   6.29733979e-02,  3.50074471e-02,  6.72845676e-01,  9.92794768e+00,
                   9.99628430e-01,  5.39591547e-02,  2.92795061e-02,  2.12480714e-03,
                  -1.01500319e-02,  1.82667162e-02,  1.00813201e-02,  5.74404197e-03,
                   6.86570242e-03,  1.25316320e+00,  4.76946516e-02,  2.71326280e-02,
                   7.02054326e-03,  9.83378673e-03, -5.70616748e-03,  5.50782881e-03,
                  -8.44213953e-04,  2.05958338e-03,  1.57866569e+00,  4.31476211e-02,
                   2.73316392e-02,  1.05505555e-02,  1.03922250e-02,  7.36865006e-03,
                  -6.00523246e-04,  6.53016990e-03, -1.72038113e-03,  1.24807860e-05,
                   1.60314173e-05,  1.21732696e-05,  5.67292645e-03,  1.92488263e-01,
                   5.08607199e-03])
        ssX.var_ = ssX.scale_**2
        swag_model.ssX = ssX
    else:
        ssX_file = path[:-4] + '_ssX.pkl'
        try:
            ssX = pkl.load(open(ssX_file, 'rb'))
            swag_model.ssX = ssX
        except FileNotFoundError:
            print(f"ssX file not found! {ssX_file}")
            ...

    return swag_model
