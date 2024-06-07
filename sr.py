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


def get_f2_inputs_and_targets_direct(args, N=500):
    model = spock_reg_model.load(version=args.version, seed=args.seed)
    model.make_dataloaders()
    model.eval()

    all_inputs, all_targets = [], []
    data_iterator = iter(model.train_dataloader())
    while sum([len(i) for i in all_inputs]) < N:
        # just takes a random batch of inputs and passes them through the neural network
        batch = next(data_iterator)
        X, y = batch 
        summary_stats, _, _ = model.generate_f2_inputs_and_targets(batch)
        all_inputs.append(summary_stats)
        all_targets.append(y)

    inputs = rearrange(all_inputs, 'l B ... -> (l B) ...')
    targets = rearrange(all_targets, 'l B ... -> (l B) ...')
    return inputs, targets


def get_f2_inputs_and_targets_residual(args, N=500):
    # model = spock_reg_model.load(version=args.version, seed=args.seed)
    TOTAL_STEPS = 300000
    TRAIN_LEN = 78660
    batch_size = 2000 #ilog_rand(32, 3200)
    steps_per_epoch = int(1+TRAIN_LEN/batch_size)
    epochs = int(1+TOTAL_STEPS/steps_per_epoch)
    args = {
        'seed': args.seed,
        'batch_size': 2000,
        'f1_depth': 1,
        'swa_lr': 5e-4 / 2,
        #'f2_depth': args.f2_depth,
        'samp': 5,
        'swa_start': epochs//2,
        'weight_decay': 1e-14,
        'to_samp': 1,
        'epochs': epochs,
        'scheduler': True,
        'scheduler_choice': 'swa',
        'steps': TOTAL_STEPS,
        'beta_in': 1e-5,
        'beta_out': 0.001,
        'act': 'softplus',
        'noisy_val': False,
        'gradient_clip': 0.1,
        # Much of these settings turn off other parameters tried:
        'fix_megno': False, #avg,std of megno
        'fix_megno2': True, #Throw out megno completely
        'include_angles': True,
        'include_mmr': False,
        'include_nan': False,
        'include_eplusminus': False,
        'power_transform': False,
        # moving some parse args to here to clean up
        'plot': False,
        'plot_random': False,
        'train_all': False,
        'lower_std': False,
    }

    arguments = {
        'no_log': False,
       # 'run_swag': None,
        'slurm_id': -1,
        'slurm_name': '',
        'version': 1278,
        'seed': 0,
        'total_steps': 300000,
        'hidden_dim': 40,
        'latent': 20,
        'swa_steps': 50000,
        'batch_size': 2000,
        'lr': 5e-4,
        #'eval': None,
        'sr_f1': False,
        'loss_ablate': 'default',
        'zero_theta': 0,
        'no_summary_sample': False,
        'init_special': False,
        #'no_std': None,
        #'no_mean': None,
        #'f2_ablate': None,
        #'f2_dropout': None,
        #'mean_var': None,
        #'tsurv': None,
        #'n_predicates': 10,
        'f1_variant': 'linear',
        'f2_variant': 'mlp',
        #'f2_depth': 1,
        #'l1_reg': None,
        #'l1_coeff': None,
        #'prune_f1_topk': None,
        #'prune_f1_topn': None,
        #'freeze_f1': None,
        #'freeze_f2': None,
        'load_f1': 29170,
        #'load_f2': None,
        #'load_f1_f2': None,
        #'pysr_f1': None,
        #'pysr_f1_model_selection': 'best',
        'pysr_f2': 'sr_results/5456.pkl',
        'pysr_f2_model_selection': 'best',
        #'f2_residual': None,
        #'pysr_f2_residual': None,
        #'pysr_f2_residual_model_selection': None,
        'out': None
    }

    # by default, parsed args get sent as hparams
    for k, v in arguments.items():
        args[k] = v

    model = spock_reg_model.VarModel(args)
    model.make_dataloaders()
    model.eval()

    all_inputs, all_targets, all_preds = [], [], []
    data_iterator = iter(model.train_dataloader())
    while sum([len(i) for i in all_inputs]) < N:
        # just takes a random batch of inputs and passes them through the neural network
        batch = next(data_iterator)
        X, y = batch 
        summary_stats, _, _ = model.generate_f2_inputs_and_targets(batch)
        all_inputs.append(summary_stats)
        all_targets.append(y)

        # residual target predictions
        y_preds = model(X, noisy_val=False)
        all_preds.append(model(X, noisy_val=False))

    # [B, 40] and [B, 2] and [B, 2]
    X = rearrange(all_inputs, 'l B ... -> (l B) ...')
    y = rearrange(all_targets, 'l B ... -> (l B) ...')
    y_preds = rearrange(all_preds, 'l B ... -> (l B) ...')
    # calculate the residual target: whatever of the target not explained by the preds
    y = y - y_preds
    return X, y


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
    assert_equal(X.shape, (N, len(included_ixs)))
    assert_equal(y.shape[0], N)

    return X, y

def import_Xy_f2(args):
    N = 250
    # [B, 40] and [B, 2] and [B, ]
    # X, y, stds =  get_f2_inputs_and_targets(args, N=N/args.std_percent_threshold)
    #X, y =  get_f2_inputs_and_targets_direct(args, N=N/args.std_percent_threshold) # for direct
    X, y = get_f2_inputs_and_targets_residual(args, N=N/args.std_percent_threshold) # for residual

    # sorted_stds, _ = torch.sort(stds)
    # max_std = sorted_stds[math.ceil(len(sorted_stds) * args.std_percent_threshold) - 1]
    # ixs = stds <= max_std
    # X, y, stds = X[ixs], y[ixs], stds[ixs]

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
    model = spock_reg_model.load(version=config['version'], seed=config['seed'])
    model.make_dataloaders()
    model.eval()

    data_iterator = iter(model.train_dataloader())
    x, y = next(data_iterator)
    while x.shape[0] < config['n']:
        next_x, next_y = next(data_iterator)
        x = torch.cat([x, next_x], dim=0)
        y = torch.cat([y, next_y], dim=0)

    # we use noisy val bc it is used during training the NN too
    out_dict = model.forward(x, return_intermediates=True, noisy_val=True)

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
        # target for SR is the predicted mean
        y = out_dict['predicted_mean']  # [B, 1]
        in_dim = model.summary_dim
        out_dim = 1
        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]
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
        in_dim = model.hparams['latent'] * 2 + 49   # model.summary_dim
        out_dim = 1
        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

        if args.residual:
            # Ensure y and predicted_mean have the same shape
            predicted_mean = out_dict['predicted_mean']
            predicted_mean = einops.repeat(predicted_mean, 'B F -> (B two) F', two=2)
            predicted_mean = einops.rearrange(predicted_mean, 'B two -> (B two) 1')
            if y.shape[0] != predicted_mean.shape[0]:
                raise ValueError(f"Shape mismatch: y has shape {y.shape}, but predicted_mean has shape {predicted_mean.shape}")
            # target is the residual error of the model's prediction from the ground truth
            y = y - predicted_mean

            # Load the PySR equations from the previous round
            with open(args.previous_sr_path, 'rb') as f:
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
    else:
        raise ValueError(f'Unknown target: {args.target}')
    
    # Update variable_names to include names for additional features
    additional_variable_names = [f'additional_{i}' for i in range(additional_features.shape[1])]
    variable_names += additional_variable_names

    # go down from having a batch of size B to just N
    ixs = np.random.choice(X.shape[0], size=config['n'], replace=False)
    X, y = X[ixs], y[ixs]

    # Ensure X and y are NumPy arrays
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().numpy()

    assert_equal(X.shape, (config['n'], in_dim))
    assert_equal(y.shape, (config['n'], out_dim))

    return X, y, variable_names


def run_pysr(args, xx=None, yy=None):
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
        unary_operators=['sin'], # removed "log"
        maxsize=args.max_size,
        timeout_in_seconds=int(60*60*args.time_in_hours),
        # prevent ^ from using complex exponents, nesting power laws is expressive but uninterpretable
        # base can have any complexity, exponent can have max 1 complexity
        constraints={'^': (-1, 1)},
        nested_constraints={"sin": {"sin": 0}},
        ncyclesperiteration=1000, # increase utilization since usually using 32-ish cores?
    )

    if args.target == 'f2_direct':
        # use custom loss function when predicting directly
        pysr_config['elementwise_loss'] = ELEMENTWISE_LOSS
        pass

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
    if not config['no_log']:
        wandb.init(
            entity='bnn-chaos-model',
            project='planets-sr',
            config=config,
        )

    command = utils.get_script_execution_command()
    print(command)

    X, y, variable_names = load_inputs_and_targets(config)

    model = pysr.PySRRegressor(**config['pysr_config'])
    model.fit(X, y, variable_names=variable_names)
    print('Done running pysr')

    losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]
    if not config['no_log']:
        wandb.log({'avg_loss': sum(losses)/len(losses),
                   'losses': losses,
                   })

    try:
        # delete the backup files
        subprocess.run(f"rm {config['equation_file'][:-4]}.csv.out*.bkup", shell=True, check=True)
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
    parser.add_argument('--seed', type=int, default=0, help='default=0')

    parser.add_argument('--time_in_hours', type=float, default=1)
    parser.add_argument('--niterations', type=float, default=500000) # by default, use time in hours as limit
    parser.add_argument('--max_size', type=int, default=30)
    parser.add_argument('--target', type=str, default='f2_direct', choices=['f1', 'f2', 'f2_ifthen', 'f2_direct'])
    parser.add_argument('--residual', action='store_true', help='do residual training of your target')
    parser.add_argument('--n', type=int, default=5000, help='number of data points for the SR problem')
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

import math
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