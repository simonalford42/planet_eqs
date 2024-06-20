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

ELEMENTWISE_LOSS = """
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
    model = spock_reg_model.load(version=config.version, seed=config.seed)
    model.make_dataloaders()
    model.eval()

    data_iterator = iter(model.train_dataloader())
    x, y = next(data_iterator)
    while x.shape[0] < config.n:
        next_x, next_y = next(data_iterator)
        x = torch.cat([x, next_x], dim=0)
        y = torch.cat([y, next_y], dim=0)

    out_dict = model.forward(x, return_intermediates=True, noisy_val=True)

    if config.target == 'f1':
        X = out_dict['inputs']
        y = out_dict['f1_output']
        X = rearrange(X, 'B T F -> (B T) F')
        y = rearrange(y, 'B T F -> (B T) F')
        X = X[..., INCLUDED_IXS]
        in_dim = len(INCLUDED_IXS)
        out_dim = model.hparams['latent']
        variable_names = INPUT_VARIABLE_NAMES
    elif config.target == 'f2':
        X = out_dict['summary_stats']
        y = out_dict['predicted_mean']
        in_dim = model.summary_dim
        out_dim = 1
        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]
    elif config.target == 'f2_ifthen':
        X = out_dict['summary_stats']
        y = out_dict['ifthen_preds']
        in_dim = model.summary_dim
        out_dim = 10
        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]
    elif config.target == 'f2_direct':
        X = out_dict['summary_stats']
        y = y
        X = einops.repeat(X, 'B F -> (B two) F', two=2)
        y = einops.rearrange(y, 'B two -> (B two) 1')
        in_dim = model.hparams['latent'] * 2 + 49
        out_dim = 1
        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

        if config.sr_residual:
            with open(config.previous_sr_path, 'rb') as f:
                previous_sr_model = pickle.load(f)

            additional_features = []
            summary_stats_np = out_dict['summary_stats'].detach().numpy()
            for equation_set in previous_sr_model.equations_:
                for index, equation in equation_set.iterrows():
                    lambda_func = equation['lambda_format']
                    evaluated_result = lambda_func(summary_stats_np)
                    evaluated_result = evaluated_result.reshape(-1, 1)
                    additional_features.append(evaluated_result)

            additional_features = np.hstack(additional_features)
            X = np.concatenate([summary_stats_np, additional_features], axis=1)
    else:
        raise ValueError(f"Unknown target: {config.target}")

        if config.residual:
            predicted_mean = out_dict['predicted_mean']
            predicted_mean = einops.repeat(predicted_mean, 'B F -> (B two) F', two=2)
            predicted_mean = einops.rearrange(predicted_mean, 'B two -> (B two) 1')
            if y.shape[0] != predicted_mean.shape[0]:
                raise ValueError(f"Shape mismatch: y has shape {y.shape}, but predicted_mean has shape {predicted_mean.shape}")
            y = y - predicted_mean

            with open(config.previous_sr_path, 'rb') as f:
                previous_sr_model = pickle.load(f)

            additional_features = []
            summary_stats_np = out_dict['summary_stats'].detach().numpy()
            for equation_set in previous_sr_model.equations_:
                for index, equation in equation_set.iterrows():
                    lambda_func = equation['lambda_format']
                    evaluated_result = lambda_func(summary_stats_np)
                    evaluated_result = evaluated_result.reshape(-1, 1)
                    additional_features.append(evaluated_result)

            additional_features = np.hstack(additional_features)
            X = np.concatenate([summary_stats_np, additional_features], axis=1)

    additional_variable_names = [f'additional_{i}' for i in range(additional_features.shape[1])]
    variable_names += additional_variable_names

    ixs = np.random.choice(X.shape[0], size=config.n, replace=False)
    X, y = X[ixs], y[ixs]

    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().numpy()

    assert_equal(X.shape, (config.n, in_dim))
    assert_equal(y.shape, (config.n, out_dim))

    return X, y, variable_names

def run_pysr(config):
    if not config.no_log:
        wandb.init(
            entity='bnn-chaos-model',
            project='planets-sr',
            config=vars(config),
        )

    command = utils.get_script_execution_command()
    print(command)

    X, y, variable_names = load_inputs_and_targets(config)

    pysr_config = dict(
        procs=int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES')),
        populations=3*int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES')),
        batching=True,
        equation_file=f'sr_results/{config.id}.csv',
        niterations=config.niterations,
        binary_operators=["+", "*", '/', '-', '^'],
        unary_operators=['sin'],
        maxsize=config.max_size,
        timeout_in_seconds=int(60*60*config.time_in_hours),
        constraints={'^': (-1, 1)},
        nested_constraints={"sin": {"sin": 0}},
        ncyclesperiteration=1000,
    )

    if config.target == 'f2_direct' and config.loss_fn != 'mse':
        print('Using custom loss function')
        pysr_config['elementwise_loss'] = ELEMENTWISE_LOSS
    else:
        print('Using default MSE loss')

    model = PySRRegressor(**pysr_config)
    model.fit(X, y, variable_names=variable_names)
    print('Done running pysr')

    losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]
    if not config.no_log:
        wandb.log({'avg_loss': sum(losses)/len(losses),
                   'losses': losses,
                   })

    try:
        subprocess.run(f"rm {pysr_config['equation_file'][:-4]}.csv.out*.bkup", shell=True, check=True)
        subprocess.run(f'rm julia*.out', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to delete the backup files: {e}")

    print(f"Saved to path: {pysr_config['equation_file']}")


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
    parser.add_argument('--seed', type=int, default=0, help='default=0')

    parser.add_argument('--time_in_hours', type=float, default=1)
    parser.add_argument('--niterations', type=float, default=500000) # by default, use time in hours as limit
    parser.add_argument('--max_size', type=int, default=30)
    parser.add_argument('--target', type=str, default='f2_direct', choices=['f1', 'f2', 'f2_ifthen', 'f2_direct'])
    parser.add_argument('--residual', action='store_true', help='do residual training of your target')
    # default will use custom when the targets are instability predictions, mse otherwise
    parser.add_argument('--loss_fn', choices=['mse', 'custom'], default=None)
    parser.add_argument('--n', type=int, default=5000, help='number of data points for the SR problem')
    parser.add_argument('--sr_residual', action='store_true', help='do residual training of your target with previous sr run as base')
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


"""
==================================================================
==================================================================
BOOSTED SR STARTS HERE
==================================================================
==================================================================
"""


# can try slope of previous to current, angles, first & last, random, etc.
def choose_topk(self, version, top_k):
    path = f'sr_results/{version}.pkl'
    results = pickle.load(open(path, 'rb')) 
    results = results.equations_[0]
    x = results['complexity']
    y = results['loss']
    sorted_indices = np.argsort(x) # Sort the points by complexity (x)
    x_sorted = np.array(x)[sorted_indices]
    y_sorted = np.array(y)[sorted_indices]
    slopes = np.diff(y_sorted) / np.diff(x_sorted)
    slope_changes = np.abs(np.diff(slopes))
    greatest_inflection_indices = np.argsort(slope_changes)[-top_k:]
    top_k_points = [(x_sorted[i+1], y_sorted[i+1]) for i in greatest_inflection_indices]
    return top_k_points

#choose points along the convex hull of pareto

# def direct_pysr_run(args):
#     X, y = import_Xy_f2_direct(args)
#     run_pysr(X, y, args)

# def residual_pysr_run(args):    
#     X, y = import_Xy_f2_residual(args)
#     run_pysr(X, y, args)


"""
0) Run find_minima.py with f1: linear and f2: mlp to get version
for i in range(num_iters):
    1) Run PySR on data
        run_pysr(version=21101, target=direct) (this first round would use import_Xy_direct)
        21101 is the version of the model with the desired f1 network
        Output is model 87543.pkl, results, etc.
    2) Construct pareto front and choose top_k models from the pareto front.
        complexities = choose_topk(version=87543, top_k=top_k)
        complexities might be [4, 8, 12]: gives the complexity of the models that we're choosing
        note: for loading the residual model of a certain complexity, you can look at the code in modules.py
    3) for each model chosen, calculate a separate residual.
        for complexity in complexities:
            load_pysr_module_list(path, model_selection)
            run_pysr(version=21101, target=residual, residual_model=87543.pkl, complexity=4) (this round would use import_Xy_residual)
    4) Next iteration of loop to get residual of residual
"""
# def boosted_regression(num_iters, top_k, args):
#     for i in range(num_iters):
#         if i == 0:
#             X, y = import_Xy_f2_direct(args)
#             model_version = run_pysr(X, y, args) #target is direct
#         else:
#             complexities = choose_topk(args.version, top_k=top_k)
#             for complexity in complexities:
                
#                 # need to implement complexity model selection
#                 residual_model = modules.PySRNet(filepath='sr_results/hall_of_fame_21101_0_1.pkl', model_selection='complexity')
                
#                 X, y = import_Xy_f2_residual(args, residual_model)

#                 model_version = run_pysr(X, y, args) #target is residual

#         args.version = model_version


if __name__ == '__main__':
    args = parse_args()

    run_pysr(args)

    #direct_pysr_run(args)
    #residual_pysr_run(args)

    # num_iters = 5
    # top_k = 3
    # boosted_regression(num_iters, top_k, args)



    # cmd = f'python test_multi_sim.py --id {ids[i]} --dir {args.dir} ' \
    #       f'--steps {args.steps} --body-type {args.body_type} --sdf-dx {args.sdf_dx} '
    # if not args.non_fixed: cmd += '--fixed'

    # version=((1 + math.random % 999999))
    # version2=((1 + math.random % 999999))
    
    # cmd = f'python -u find_minima.py --total_steps 300000 --version $version \
    #         --slurm_id $SLURM_JOB_ID --slurm_name $SLURM_JOB_NAME --f1_variant linear --f2_variant mlp '
    # os.system(cmd)