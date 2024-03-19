
import subprocess
import wandb
import pysr
from pysr import PySRRegressor
pysr.julia_helpers.init_julia()
import random

from matplotlib import pyplot as plt
import seaborn as sns
import os
sns.set_style('darkgrid')
import spock_reg_model
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import argparse
from einops import rearrange
from sklearn.decomposition import PCA
import utils
import pickle
from utils import assert_equal
import json
import einops
import torch
import math


def get_f1_inputs_and_targets(args):
    model = spock_reg_model.load(version=args.version, seed=args.seed)
    model.make_dataloaders()
    model.eval()

    # just takes a random batch of inputs and passes them through the neural network
    batch = next(iter(model.train_dataloader()))
    inputs, targets = model.generate_f1_inputs_and_targets(batch)
    return inputs, targets


def get_f2_inputs_and_targets(args, N=250):
    model = spock_reg_model.load(version=args.version, seed=args.seed)
    model.make_dataloaders()
    model.eval()

    all_inputs, all_targets, all_stds = [], [], []
    data_iterator = iter(model.train_dataloader())
    while sum([len(i) for i in all_inputs]) < N:
        # just takes a random batch of inputs and passes them through the neural network
        batch = next(data_iterator)
        inputs, targets, stds = model.generate_f2_inputs_and_targets(batch)
        all_inputs.append(inputs)
        all_targets.append(targets)
        all_stds.append(stds)

    inputs = rearrange(all_inputs, 'l B ... -> (l B) ...')
    targets = rearrange(all_targets, 'l B ... -> (l B) ...')
    stds = rearrange(all_stds, 'l B ... -> (l B) ...')
    return inputs, targets, stds


def import_Xy(args, included_ixs):
    inputs, targets =  get_f1_inputs_and_targets(args)

    N = 250
    X = rearrange(inputs, 'B T F -> (B T) F')
    y = rearrange(targets, 'B T F -> (B T) F')
    ixs = np.random.choice(X.shape[0], size=N, replace=False)
    X, y = X[ixs], y[ixs]
    X, y = X.detach().numpy(), y.detach().numpy()

    assert_equal(len(LABELS), X.shape[1])
    # all of the skipped ones are just set to zero.
    assert np.all(X[..., [i for i in range(len(LABELS)) if i not in included_ixs]] == 0)

    X = X[..., included_ixs]
    assert_equal(X.shape, (500, len(included_ixs)))
    assert_equal(y.shape[0], 500)

    return X, y


def import_Xy_f2(args):
    N = 250
    # [B, 40] and [B, 2] and [B, ]
    X, y, stds =  get_f2_inputs_and_targets(args, N=N/args.std_percent_threshold)

    sorted_stds, _ = torch.sort(stds)
    max_std = sorted_stds[math.ceil(len(sorted_stds) * args.std_percent_threshold) - 1]
    ixs = stds <= max_std
    X, y, stds = X[ixs], y[ixs], stds[ixs]

    ixs = np.random.choice(X.shape[0], size=N, replace=False)
    X, y = X[ixs], y[ixs]
    X, y = X.detach().numpy(), y.detach().numpy()

    return X, y

def import_Xy_f2_ifthen(args):
    N = 250
    # [B, 40] and [B, 2] and [B, ]
    X, y = get_f2_ifthen_inputs_and_targets(args, N=N)

    ixs = np.random.choice(X.shape[0], size=N, replace=False)
    X, y = X[ixs], y[ixs]
    X, y = X.detach().numpy(), y.detach().numpy()

    return X, y

def get_f2_ifthen_inputs_and_targets(args, N):
    model = spock_reg_model.load(version=args.version, seed=args.seed)
    model.make_dataloaders()
    model.eval()

    all_inputs, all_targets = [], []
    data_iterator = iter(model.train_dataloader())
    while sum([len(i) for i in all_inputs]) < N:
        # just takes a random batch of inputs and passes them through the neural network
        batch = next(data_iterator)
        inputs, targets = model.generate_f2_ifthen_inputs_and_targets(batch)
        all_inputs.append(inputs)
        all_targets.append(targets)

    inputs = rearrange(all_inputs, 'l B ... -> (l B) ...')
    targets = rearrange(all_targets, 'l B ... -> (l B) ...')
    return inputs, targets


LABELS = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

LABEL_TO_IX = {label: i for i, label in enumerate(LABELS)}

def get_sr_included_ixs():
    # hard coded based off the default CL args passed
    skipped = ['nan_mmr_near', 'nan_mmr_far', 'nan_megno', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno']
    assert len(skipped) == 10

    included_ixs = [i for i in range(len(LABELS)) if LABELS[i] not in skipped]
    included_labels = [LABELS[i] for i in included_ixs]
    return included_ixs


def run_pysr(args):
    id = random.randint(0, 100000)
    while os.path.exists(f'sr_results/{id}.pkl'):
        id = random.randint(0, 100000)

    included_ixs = get_sr_included_ixs()

    path = f'sr_results/{id}.pkl'
    # replace '.pkl' with '.csv'
    path = path[:-3] + 'csv'
    # save the included ixs
    ixs_path = path[:-4] + '_ixs.json'
    with open(ixs_path, 'w') as f:
        json.dump(included_ixs, f)

    pysr_config = dict(
        # https://stackoverflow.com/a/57474787/4383594
        procs=int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES')),
        cluster_manager='slurm',
        equation_file=path,
        niterations=500000,
        binary_operators=["+", "*", '/', '-', '^'],
        unary_operators=["log", 'sin'],
        maxsize=args.max_size,
        timeout_in_seconds=int(60*60*args.time_in_hours),
        # prevent ^ from using complex exponents, nesting power laws is expressive but uninterpretable
        # base can have any complexity, exponent can have max 1 complexity
        constraints={'^': (-1, 1)},
        nested_constraints={"sin": {"sin": 0}},
        ncyclesperiteration=2000, # increase utilization since usually using 32-ish cores?
    )

    config = vars(args)
    config.update(pysr_config)
    config.update({
        'id': id,
        'results_cmd': f'vim $(ls {path[:-4]}.csv*)',
    })

    if not args.no_log:
        wandb.init(
            entity='bnn-chaos-model',
            project='planets-sr',
            config=config,
        )

    command = utils.get_script_execution_command()
    print(command)

    if args.target == 'f1':
        X, y = import_Xy(args, included_ixs)
        variables = variable_names(included_ixs)
    elif args.target == 'f2':
        X, y = import_Xy_f2(args)
        n = X.shape[1] // 2
        variables = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]
    elif args.target == 'f2_ifthen':
        X, y = import_Xy_f2_ifthen(args)
        n = X.shape[1] // 2
        variables = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

    model = pysr.PySRRegressor(**pysr_config)
    model.fit(X, y, variable_names=variables)
    print('Done running pysr')

    losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]
    if not args.no_log:
        wandb.log({'avg_loss': sum(losses)/len(losses),
                   'losses': losses,
                   })

    # delete the backup files
    try:
        subprocess.run(f'rm {path[:-4]}.csv.out*.bkup', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to delete the backup files: {e}")

    print(f'Saved to path: {path}')


def variable_names(included_ixs):
    return [LABELS[ix] for ix in included_ixs]

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
    # X = torch.ones((500, 31))
    y = spock_features(X)

def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    # when importing from jupyter nb, it passes an arg to --f which we should just ignore
    parser.add_argument('--no_log', action='store_true', default=False, help='disable wandb logging')
    parser.add_argument('--slurm_id', type=int, default=-1, help='slurm job id')
    parser.add_argument('--slurm_name', type=str, default='', help='slurm job name')
    parser.add_argument('--version', type=int, help='', default=63524)
    parser.add_argument('--seed', type=int, default=0, help='default=0')

    parser.add_argument('--time_in_hours', type=float, default=1)
    parser.add_argument('--max_size', type=int, default=30)
    parser.add_argument('--target', type=str, default='f1', choices=['f1', 'f2', 'f2_ifthen'])
    # use the bottom % of stds to target sr on higher confidence predictions
    parser.add_argument('--std_percent_threshold', type=float, default=1)

    args = parser.parse_args()
    return args


def load_results(id):
    path = 'sr_results/id.pkl'
    results: PySRRegressor = pickle.load(open(path, 'rb'))
    return results


def plot_pareto(path):
    results = pickle.load(open(path, 'rb'))
    results = results.equations_[0]
    x = results['complexity']
    y = results['loss']
    # plot the pareto frontier
    plt.scatter(x, y)
    plt.xlabel('complexity')
    plt.ylabel('loss')
    plt.title('pareto frontier for' + path)
    # save the plot
    plt.savefig('pareto.png')


if __name__ == '__main__':
    args = parse_args()
    run_pysr(args)
