
import subprocess
import wandb
import pysr
from pysr import PySRRegressor
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

ELEMENTWISE_LOSS = """elementwise_loss(prediction, target) = begin
    mu = prediction
    sigma = one(prediction)

    safe_log_erf(x) = x < -1 ? (
        T = typeof(x);
        T(0.485660082730562) * x + T(0.643278438654541) * exp(x) +
        T(0.00200084619923262) * x^3 - T(0.643250926022749) - T(0.955350621183745) * x^2
    ) : log(1 + erf(x))

    log_like = target >= 9 ? safe_log_erf((mu - 9) / sqrt(2 * sigma^2)) : (
        zero(prediction) - (target - mu)^2 / (2 * sigma^2) - log(sigma) - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
    )

    return -log_like
end
"""

LABELS = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

LABEL_TO_IX = {label: i for i, label in enumerate(LABELS)}

def get_sr_included_ixs():
    # hard coded based off the default CL args passed
    skipped = ['nan_mmr_near', 'nan_mmr_far', 'nan_megno', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno']
    assert len(skipped) == 10

    included_ixs = [i for i in range(len(LABELS)) if LABELS[i] not in skipped]
    return included_ixs


INCLUDED_IXS = get_sr_included_ixs()
INPUT_VARIABLE_NAMES = [LABELS[ix] for ix in INCLUDED_IXS]


def load_inputs_and_targets(args):
    model = spock_reg_model.load(version=args.version, seed=args.seed)
    model.make_dataloaders()
    model.eval()

    batch = next(iter(model.train_dataloader()))
    x, y = batch
    # we use noisy val bc it is used during training the NN too
    out_dict = model.forward(x, return_intermediates=True, noisy_val=True)

    if args.target == 'f1':
        # inputs to SR are the inputs to f1 neural network
        # we use this instead of x because the model zeros the unused inputs,
        #  which is a nice check to have
        X = out_dict['inputs']  # [B, T, F]
        # targets for SR are the outputs of the f1 neural network
        y = out_dict['f1_output']  # [B, T, F]
        # f1 acts on timesteps independently, so we can just use different
        #  time steps as different possible samples
        X = rearrange(X, 'B T F -> (B T) F')
        y = rearrange(y, 'B T F -> (B T) F')

        # extract just the input variables we're actually using
        assert_equal(len(LABELS), X.shape[1])
        X = X[..., INCLUDED_IXS]
        in_dim = len(INCLUDED_IXS)
        out_dim = model.hparams['latent']
        variable_names = INPUT_VARIABLE_NAMES
    elif args.target == 'f2':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # target for SR is the predicted mean
        y = out_dict['predicted_mean']  # [B, 1]
        in_dim = model.hparams['latent'] * 2
        out_dim = 1
        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]
    elif args.target == 'f2_ifthen':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # target for SR is the predicates from the ifthen network
        y = out_dict['ifthen_preds']  # [B, 10]
        in_dim = model.hparams['latent'] * 2
        out_dim = 10
        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]
    elif args.target == 'f2_direct':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # target for SR is the ground truth mean, which we already have
        y = y  # [B, 2]
        # there are two ground truth predictions. create a data point for each
        X = einops.repeat(X, 'B F -> (B two) F', two=2)
        y = einops.rearrange(y, 'B two -> (B two) 1')
        in_dim = model.hparams['latent'] * 2
        out_dim = 1  #
        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]
    else:
        raise ValueError(f'Unknown target: {args.target}')

    if args.residual:
        assert args.target == 'f2_direct', 'residual requires a direct target'
        # target is the residual error of the model's prediction from the ground truth
        y = y - out_dict['predicted_mean']


    # go down from having a batch of size B to just N
    ixs = np.random.choice(X.shape[0], size=args.n, replace=False)
    X, y = X[ixs], y[ixs]
    X, y = X.detach().numpy(), y.detach().numpy()

    assert_equal(X.shape, (args.n, in_dim))
    assert_equal(y.shape, (args.n, out_dim))

    return X, y, variable_names


def run_pysr(args):
    id = random.randint(0, 100000)
    while os.path.exists(f'sr_results/{id}.pkl'):
        id = random.randint(0, 100000)

    path = f'sr_results/{id}.pkl'
    # replace '.pkl' with '.csv'
    path = path[:-3] + 'csv'

    pysr_config = dict(
        # https://stackoverflow.com/a/57474787/4383594
        procs=int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES')),
        cluster_manager='slurm',
        equation_file=path,
        niterations=args.niterations,
        binary_operators=["+", "*", '/', '-', '^'],
        unary_operators=["log", 'sin'],
        maxsize=args.max_size,
        timeout_in_seconds=int(60*60*args.time_in_hours),
        # prevent ^ from using complex exponents, nesting power laws is expressive but uninterpretable
        # base can have any complexity, exponent can have max 1 complexity
        constraints={'^': (-1, 1)},
        nested_constraints={"sin": {"sin": 0}},
        ncyclesperiteration=2000, # increase utilization since usually using 32-ish cores?
        elementwise_loss=ELEMENTWISE_LOSS,
    )

    config = vars(args)
    config.update(pysr_config)
    config.update({
        'id': id,
        'results_cmd': f'vim $(ls {path[:-4]}.csv*)',
        'slurm_id': os.environ.get('SLURM_JOB_ID', None),
        'slurm_name': os.environ.get('SLURM_JOB_NAME', None),
    })

    if not args.no_log:
        wandb.init(
            entity='bnn-chaos-model',
            project='planets-sr',
            config=config,
        )

    command = utils.get_script_execution_command()
    print(command)

    X, y, variable_names = load_inputs_and_targets(args)
    model = pysr.PySRRegressor(**pysr_config)
    model.fit(X, y, variable_names=variable_names)
    print('Done running pysr')

    losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]
    if not args.no_log:
        wandb.log({'avg_loss': sum(losses)/len(losses),
                   'losses': losses,
                   })

    try:
        # delete the backup files
        subprocess.run(f'rm {path[:-4]}.csv.out*.bkup', shell=True, check=True)
        # delete julia files: julia-1911988-17110333239-0016.out
        subprocess.run(f'rm julia*.out', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to delete the backup files: {e}")

    print(f'Saved to path: {path}')


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


def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    # when importing from jupyter nb, it passes an arg to --f which we should just ignore
    parser.add_argument('--no_log', action='store_true', default=False, help='disable wandb logging')
    parser.add_argument('--version', type=int, help='', required=True)
    parser.add_argument('--seed', type=int, default=0, help='default=0')

    parser.add_argument('--time_in_hours', type=float, default=1)
    parser.add_argument('--niterations', type=float, default=500000) # by default, use time in hours as limit
    parser.add_argument('--max_size', type=int, default=30)
    parser.add_argument('--target', type=str, default='f1', choices=['f1', 'f2', 'f2_ifthen', 'f2_direct'])
    parser.add_argument('--residual', action='store_true', help='do residual training of your target')
    parser.add_argument('--n', type=int, default=250, help='number of data points for the SR problem')

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
