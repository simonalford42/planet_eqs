
import pysr
pysr.julia_helpers.init_julia()

import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import pyplot as plt
import spock_reg_model
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
from utils import assert_equal
import json
import einops

ARGS, CHECKPOINT_FILENAME = parse_swag_args.parse()

def compute_features(inputs):
    model = PySRRegressor.from_file('results/hall_of_fame.pkl')
    assert inputs.shape[-1] == 31


def load_model():
    # Fixed hyperparams:
    name = 'full_swag_pre_' + CHECKPOINT_FILENAME
    checkpoint_path = CHECKPOINT_FILENAME + '/version=0-v0.ckpt'
    try:
        model = spock_reg_model.VarModel.load_from_checkpoint(checkpoint_path)
    except FileNotFoundError:
        checkpoint_path = CHECKPOINT_FILENAME + '/version=0.ckpt'
        model = spock_reg_model.VarModel.load_from_checkpoint(checkpoint_path)

    return model

def get_f1_inputs_and_targets():
    model = load_model()
    model.make_dataloaders()
    model.eval()
    # print(model.inputs_mask.mask.data)
    # print(model.features_mask.mask.data)

    # just takes a random batch of inputs and passes them through the neural network
    batch = next(iter(model.train_dataloader()))
    inputs, targets = model.generate_f1_inputs_and_targets(batch, batch_idx=0)
    return inputs, targets


def update_args(version=1278, seed=0, total_steps=300000, megno=False, angles=True, power_transform=False,
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


def import_Xy(included_ixs):
    inputs, targets =  get_f1_inputs_and_targets()
    print(f'inputs: {inputs.shape}')

    N = 500
    X = rearrange(inputs, 'B T F -> (B T) F')
    y = rearrange(targets, 'B T F -> (B T) F')
    indices = np.random.choice(X.shape[0], size=N, replace=False)
    X, y = X[indices], y[indices]
    X, y = X.detach().numpy(), y.detach().numpy()

    assert_equal(len(LABELS), X.shape[1])
    # all of the skipped ones are just set to zero.
    assert np.all(X[..., [i for i in range(len(LABELS)) if i not in included_ixs]] == 0)

    X = X[..., included_ixs]
    assert_equal(X.shape, (500, 31))
    assert_equal(y.shape, (500, 20))

    return X, y


LABELS = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

LABEL_TO_IX = {label: i for i, label in enumerate(LABELS)}


def get_sr_included_ixs():
    # hard coded based off the default CL args passed
    skipped = ['nan_mmr_near', 'nan_mmr_far', 'nan_megno', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno']
    assert len(skipped) == 10

    included_ixs = [i for i in range(len(LABELS)) if LABELS[i] not in skipped]
    included_labels = [LABELS[i] for i in included_ixs]
    return included_ixs


def run_regression(X, y, args):
    included_indices = get_sr_included_ixs()
    X, y = import_Xy(included_indices)

    # go down to 6 features to make SR easier?
    # X = PCA(n_components=6).fit_transform(X)

    path = utils.next_unused_path(f'sr_results/hall_of_fame_{ARGS.version}_{ARGS.seed}.pkl', lambda i: f'_{i}')
    # replace '.pkl' with '.csv'
    path = path[:-3] + 'csv'
    # save the included ixs
    indices_path = path[:-4] + '_indices.json'
    with open(indices_path, 'w') as f:
        json.dump(included_indices, f)

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


def spock_features(X):
    features = ['a1', 'a2', 'a3']

    def x(f):
        return X[:, LABEL_TO_IX[f]]

    def e_cross_inner(x):
        return (x('a2') - x('a1')) / x('a1')

    def e_cross_outer(x):
        return (x('a3') - x('a2')) / x('a2')

    y = [e_cross_inner(x), e_cross_outer(x)]
    y = einops.rearrange(y, 'n B -> B n')
    return y



def test_pysr():
    included_indices = get_sr_included_ixs()
    X, _ = import_Xy(included_indices)
    y = spock_features(X)

    path = utils.next_unused_path(f'sr_results/sr_test_hof.pkl', lambda i: f'_{i}')
    # replace '.pkl' with '.csv'
    path = path[:-3] + 'csv'

    model = pysr.PySRRegressor(
        equation_file=path,
        niterations=500,
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
        timeout_in_seconds=60*60*24*6,
        # prevent ^ from using complex exponents, nesting power laws is expressive but uninterpretable
        # base can have any complexity, exponent can have max 1 complexity
        constraints={'^': (-1, 1)},
        nested_constraints={"sin": {"sin": 0}},
    )
    model.fit(X, y)



if __name__ == '__main__':
    test_pysr()
    # useful if running from inside a shell

    # run_regression(X, y, ARGS)


