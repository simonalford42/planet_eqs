
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
import parse_swag_args
from einops import rearrange
from sklearn.decomposition import PCA
import utils

ARGS, CHECKPOINT_FILENAME = parse_swag_args.parse()

def compute_features(inputs):
    model = PySRRegressor.from_file('results/hall_of_fame.pkl')
    assert inputs.shape[-1] == 31


def get_f1_inputs_and_targets():
    # Fixed hyperparams:
    name = 'full_swag_pre_' + CHECKPOINT_FILENAME
    checkpoint_path = CHECKPOINT_FILENAME + '/version=0-v0.ckpt'
    try:
        model = spock_reg_model.VarModel.load_from_checkpoint(checkpoint_path)
    except FileNotFoundError:
        checkpoint_path = CHECKPOINT_FILENAME + '/version=0.ckpt'
        model = spock_reg_model.VarModel.load_from_checkpoint(checkpoint_path)

    model.make_dataloaders()
    model.eval()

    # just takes a random batch of inputs and passes them through the neural network
    batch = next(iter(model.train_dataloader()))
    inputs, targets = model.generate_f1_inputs_and_targets(batch, batch_idx=0)
    return inputs, targets


def update_args(version=4995, seed=0, total_steps=300000, megno=False, angles=True, power_transform=False,
        hidden=40, latent=20, no_mmr=True, no_nan=True, no_eplusminus=True, train_all=False):
    extra = ''
    if no_nan:
        extra += '_nonan=1' 
    if no_eplusminus:
        extra += '_noeplusminus=1' 
    if train_all:
        extra += '_train_all=1' 
    checkpoint_filename = (
            "results/steps=%d_megno=%d_angles=%d_power=%d_hidden=%d_latent=%d_nommr=%d" %
            (total_steps, megno, angles, power_transform, hidden, latent, no_mmr)
        + extra + '_v' + str(version)
    )
    checkpoint_filename += '_%d' %(seed,)

    global ARGS, CHECKPOINT_FILENAME
    ARGS.version = version
    ARGS.seed = seed
    CHECKPOINT_FILENAME = checkpoint_filename


def import_Xy():
    inputs, targets =  get_f1_inputs_and_targets()

    N = 500
    X = rearrange(inputs, 'B T F -> (B T) F')
    y = rearrange(targets, 'B T F -> (B T) F')
    indices = np.random.choice(X.shape[0], size=N, replace=False)
    X, y = X[indices], y[indices]
    X, y = X.detach().numpy(), y.detach().numpy()


    all_labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

    # hard coded based off the default CL args passed
    skipped = ['nan_mmr_near', 'nan_mmr_far', 'nan_megno', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno']
    assert len(skipped) == 10

    included_ixs = [i for i in range(len(all_labels)) if all_labels[i] not in skipped]
    labels = [all_labels[i] for i in included_ixs]

    assert np.all(X[..., [i for i in range(len(all_labels)) if i not in included_ixs]] == 0)

    X = X[..., included_ixs]
    assert X.shape[-1] == 31

    return X, y


def run_regression(X, y, args):
    # go down to 6 features to make SR easier
    # X = PCA(n_components=6).fit_transform(X)

    path = utils.next_unused_path(f'sr_results/hall_of_fame_{ARGS.version}_{ARGS.seed}.pkl', lambda i: f'_{i}')
    # replace '.pkl' with '.csv'
    path = path[:-3] + 'csv'

    model = pysr.PySRRegressor(
        equation_file=path,
        niterations=500000,
        binary_operators=["+", "*", '/', '-', '^'],
        unary_operators=[
           # use fewer operators, nonredundant
           # "square",
           # "cube",
           # "exp",
           "log",
           # 'abs',
           # 'sqrt',
           'sin',
           # 'cos',
           # 'tan',
        ],
        maxsize=60,
        timeout_in_seconds=int(60*60*args.time_in_hours),
        # prevent ^ from using complex exponents, nesting power laws is expressive but uninterpretable
        # base can have any complexity, exponent can have max 1 complexity
        constraints={'^': (-1, 1)},
        nested_constraints={"sin": {"sin": 0}},
    )
    model.fit(X, y)


if __name__ == '__main__':
    # update_args(version=1278)
    X, y = import_Xy()
    run_regression(X, y, ARGS)


