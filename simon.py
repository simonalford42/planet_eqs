
import pysr
pysr.julia_helpers.init_julia()

import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import pyplot as plt
import spock_reg_model
spock_reg_model.HACK_MODEL = True
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np
from scipy.stats import truncnorm
import sys
from parse_swag_args import parse
from einops import rearrange

def compute_features(inputs):
    model = PySRRegressor.from_file('results/hall_of_fame.pkl')
    assert inputs.shape[-1] == 31


def import_inputs_and_nn_features(args, checkpoint_filename):
    # Fixed hyperparams:
    name = 'full_swag_pre_' + checkpoint_filename
    checkpoint_path = checkpoint_filename + '/version=0-v0.ckpt'
    model = spock_reg_model.VarModel.load_from_checkpoint(checkpoint_path)
    model.make_dataloaders()
    model.eval()

    # just takes a random batch of inputs and passes them through the neural network
    batch = next(iter(model.train_dataloader()))
    inputs, targets = model.generate_f1_inputs_and_targets(batch, batch_idx=0)
    print('inputs shape: ', inputs.shape)
    print('targets shape: ', targets.shape)

    N = 100
    X = rearrange(inputs, 'B T F -> (B T) F')
    y = rearrange(targets, 'B T F -> (B T) F')
    indices = np.random.choice(X.shape[0], size=N, replace=False)
    X, y = X[indices], y[indices]
    X, y = X.detach().numpy(), y.detach().numpy()


    all_labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

    skipped = ['nan_mmr_near', 'nan_mmr_far', 'nan_megno', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno']
    assert len(skipped) == 10

    included_ixs = [i for i in range(len(all_labels)) if all_labels[i] not in skipped]
    labels = [all_labels[i] for i in included_ixs]

    assert np.all(X[..., [i for i in range(len(all_labels)) if i not in included_ixs]] == 0)

    # X = X[..., included_ixs]
    # assert X.shape[-1] == 31

    return X, y


def run_regression(X, y, args):
    model = pysr.PySRRegressor(
        equation_file=f'results/hall_of_fame_{args.version}_{args.seed}.pkl',
        niterations=5,  # < Increase me for better results
        binary_operators=["+", "*", '/', '-', '^'],
        unary_operators=[
            "square",
            "cube",
            "exp",
            "log",
            'abs',
            'sqrt',
            'sin',
            'cos',
            'tan',
        ],
    )

    model.fit(X, y)
    torch_model = model.pytorch()

if __name__ == '__main__':
    args, checkpoint_filename = parse()
    # X, y = import_inputs_and_nn_features(args, checkpoint_filename)
    # run_regression(X, y, args)

    feature_nn = pysr.PySRRegressor.from_file('results/hall_of_fame_7955_5.pkl').pytorch()
    # .pytorch() returns a list of 20 nn's, one for each iter (?) or maybe feature?
    feature_nn = feature_nn[-1]

    name = 'full_swag_pre_' + checkpoint_filename
    checkpoint_path = checkpoint_filename + '/version=0-v0.ckpt'
    model = spock_reg_model.VarModel.load_from_checkpoint(checkpoint_path)
    model.make_dataloaders()
    model.eval()

    # just takes a random batch of inputs and passes them through the neural network
    batch = next(iter(model.train_dataloader()))
    feature_nn(batch)
