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
import numpy as np
import argparse
from einops import rearrange
import utils
import pickle
from utils import assert_equal
import einops
import torch
import math


LL_LOSS = """
function elementwise_loss(prediction, target)

    function safe_log_erf(x)
        if x < -1
            0.485660082730562*x + 0.643278438654541*exp(x)
            + 0.00200084619923262*x^3 - 0.643250926022749
            - 0.955350621183745*x^2
        else
            log(1 + erf(x))
        end
    end

    mu = prediction

    if mu < 1 || mu > 14
        # The farther away from a reasonable range, the more we punish it
        return 100 * (mu - 7)^2
    end

    sigma = one(prediction)

    # equation 8 in Bayesian neural network paper
    log_like = if target >= 9
        safe_log_erf((mu - 9) / sqrt(2 * sigma^2))
    else
        (
            zero(prediction)
            - (target - mu)^2 / (2 * sigma^2)
            - log(sigma)
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    end

    return -log_like
end
"""


LABELS = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

LABEL_TO_IX = {label: i for i, label in enumerate(LABELS)}

def get_sr_included_ixs():
    ''' for running pysr to imitate f1, we only want to use the inputs that f1 uses '''
    # hard coded based off the default CL args passed
    skipped = ['nan_mmr_near', 'nan_mmr_far', 'nan_megno', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno']
    assert len(skipped) == 10

    included_ixs = [i for i in range(len(LABELS)) if LABELS[i] not in skipped]
    return included_ixs


INCLUDED_IXS = get_sr_included_ixs()
INPUT_VARIABLE_NAMES = [LABELS[ix] for ix in INCLUDED_IXS]


def load_inputs_and_targets(config):
    model = spock_reg_model.load(version=config['version'])
    model.make_dataloaders()
    model.eval()

    data_iterator = iter(model.train_dataloader())
    x, y = next(data_iterator)
    while x.shape[0] < config['n']:
        next_x, next_y = next(data_iterator)
        x = torch.cat([x, next_x], dim=0)
        y = torch.cat([y, next_y], dim=0)

    # we use noisy val bc it is used during training the NN too
    out_dict = model.forward(x, return_intermediates=True, noisy_val=False)

    if config['target'] == 'f1':
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
    elif config['target'] == 'f2':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # outputs are the (mean, std) predictions of the nn
        y = out_dict['prediction']  # [B, 2]

        in_dim = model.summary_dim
        out_dim = 2

        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

        if config['sr_residual']:
            # 1. load the previous pysr results, using highest complexity
            # 2. calculate mean and std from previous results (previous run must have used target=='f2')
            # 3. concatenate mean and std to produce previous "prediction" of shape [B, 2]
            # 4. subtract previous prediction from y to get residual target

            # 1
            with open(config['previous_sr_path'], 'rb') as f:
                previous_sr_model = pickle.load(f)

            # 2
            summary_stats_np = out_dict['summary_stats'].detach().numpy()

            # get the highest complexity equation
            max_complexity_idx = previous_sr_model.equations['complexity'].max()
            mean_equation = previous_sr_model.equations_[0].iloc[max_complexity_idx]
            std_equation = previous_sr_model.equations_[1].iloc[max_complexity_idx]

            # mean and std
            results = []
            for equation in mean_equation, std_equation:
                lambda_func = equation['lambda_format']
                evaluated_result = lambda_func(summary_stats_np)
                # Ensure the result is reshaped to match the batch size
                evaluated_result = evaluated_result.reshape(-1, 1)
                results.append(evaluated_result)

            # 3
            previous_prediction = np.hstack(results)
            assert_equal(previous_prediction.shape, (X.shape[0], 2))  # [B, 2]

            # 4
            y = y - previous_prediction




    elif config['target'] == 'f2_ifthen':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # target for SR is the predicates from the ifthen network
        y = out_dict['ifthen_preds']  # [B, 10]

        in_dim = model.summary_dim
        out_dim = 10

        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]
    elif config['target'] == 'f2_direct':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # target for SR is the ground truth mean, which we already have
        y = y  # [B, 2]
        # there are two ground truth predictions. create a data point for each
        X = einops.repeat(X, 'B F -> (B two) F', two=2)
        y = einops.rearrange(y, 'B two -> (B two) 1')

        in_dim = model.summary_dim
        out_dim = 1

        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

        if config['sr_residual']:

            # Load the PySR equations from the previous round
            with open(config['previous_sr_path'], 'rb') as f:
                previous_sr_model = pickle.load(f)

            # Evaluate the previous PySR equations on the inputs
            additional_features = []
            summary_stats_np = out_dict['summary_stats'].detach().numpy()
            for equation_set in previous_sr_model.equations_:
                for index, equation in equation_set.iterrows():
                    lambda_func = equation['lambda_format']
                    evaluated_result = lambda_func(summary_stats_np)
                    # Ensure the result is reshaped to match the batch size
                    evaluated_result = evaluated_result.reshape(-1, 1)
                    additional_features.append(evaluated_result)

            # Convert list of arrays to a single numpy array
            additional_features = np.hstack(additional_features)
            # Concatenate the original summary stats with the evaluated results
            # summary_stats_np: [1, 2000, 40]
            # additional_features: (2000, 49)
            # (args.n, in_dim) = (250, 2*20+49)
            X = np.concatenate([summary_stats_np, additional_features], axis=1)

            in_dim = model.summary_dim + additional_features.shape[1]

    else:
        raise ValueError(f"Unknown target: {config['target']}")

    if config['residual']:
        assert config['target'] == 'f2_direct', 'residual requires a direct target'
        # target is the residual error of the model's prediction from the ground truth
        # because target was f2 direct, y shape is [B * 2, 1]
        # but predicted mean is [B, 1]
        # so repeat the predicted means
        y_old = out_dict['mean']
        assert_equal(y_old.shape[1], 1)
        y_old = einops.repeat(y_old, 'B one -> (B repeat) one', repeat=2)
        assert_equal(y_old.shape, y.shape)
        y = y - y_old

    # go down from having a batch of size B to just N
    ixs = np.random.choice(X.shape[0], size=config['n'], replace=False)
    X, y = X[ixs], y[ixs]

    # Ensure X and y are NumPy arrays
    if isinstance(X, torch.Tensor):
        X, y = X.detach().numpy(), y.detach().numpy()

    assert_equal(X.shape, (config['n'], in_dim))
    assert_equal(y.shape, (config['n'], out_dim))

    return X, y, variable_names


def get_config(args):
    id = random.randint(0, 100000)
    while os.path.exists(f'sr_results/{id}.pkl'):
        id = random.randint(0, 100000)

    path = f'sr_results/{id}.pkl'
    # replace '.pkl' with '.csv'
    path = path[:-3] + 'csv'

    # https://stackoverflow.com/a/57474787/4383594
    num_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES'))
    pysr_config = dict(
        procs=num_cpus,
        populations=3*num_cpus,
        batching=True,
        # cluster_manager='slurm',
        equation_file=path,
        niterations=args.niterations,
        # multithreading=False,
        binary_operators=["+", "*", '/', '-', '^'],
        unary_operators=['sin'],  # removed "log"
        maxsize=args.max_size,
        timeout_in_seconds=int(60*60*args.time_in_hours),
        # prevent ^ from using complex exponents, nesting power laws is expressive but uninterpretable
        # base can have any complexity, exponent can have max 1 complexity
        constraints={'^': (-1, 1)},
        nested_constraints={"sin": {"sin": 0}},
        ncyclesperiteration=1000, # increase utilization since usually using 32-ish cores?
    )

    if args.loss_fn == 'll':
        assert args.target == 'f2_direct', 'log likelihood loss only useful for f2_direct'
        pysr_config['elementwise_loss'] = LL_LOSS

    config = vars(args)
    config.update(pysr_config)
    config['pysr_config'] = pysr_config
    config.update({
        'id': id,
        'results_cmd': f'vim $(ls {path[:-4]}.csv*)',
        'slurm_id': os.environ.get('SLURM_JOB_ID', None),
        'slurm_name': os.environ.get('SLURM_JOB_NAME', None),
    })

    return config


def run_pysr(config):
    command = utils.get_script_execution_command()
    print(command)

    X, y, variable_names = load_inputs_and_targets(config)

    model = pysr.PySRRegressor(**config['pysr_config'])

    if not config['no_log']:
        wandb.init(
            entity='bnn-chaos-model',
            project='planets-sr',
            config=config,
        )

    model.fit(X, y, variable_names=variable_names)
    print('Done running pysr')

    losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]

    if not config['no_log']:
        wandb.log({'avg_loss': sum(losses)/len(losses),
                   'losses': losses,
                   })

    try:
        # delete julia files: julia-1911988-17110333239-0016.out
        subprocess.run(f'rm julia*.out', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to delete the backup files: {e}")

    print(f"Saved to path: {config['equation_file']}")


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
    parser.add_argument('--version', type=int, help='')

    parser.add_argument('--time_in_hours', type=float, default=1)
    parser.add_argument('--niterations', type=float, default=500000) # by default, use time in hours as limit
    parser.add_argument('--max_size', type=int, default=30)
    parser.add_argument('--target', type=str, default='f2_direct', choices=['f1', 'f2', 'f2_ifthen', 'f2_direct', 'f2_2'])
    parser.add_argument('--residual', action='store_true', help='do residual training of your target')
    parser.add_argument('--n', type=int, default=5000, help='number of data points for the SR problem')
    parser.add_argument('--sr_residual', action='store_true', help='do residual training of your target with previous sr run as base')
    parser.add_argument('--loss_fn', type=str, choices=['mse', 'll'], help='choose "ll" to use loglikelidhood loss')
    parser.add_argument('--previous_sr_path', type=str, default='sr_results/92985.pkl')

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
    config = get_config(args)
    run_pysr(config)
