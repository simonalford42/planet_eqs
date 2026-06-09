import subprocess
import wandb
import pysr
from pysr import jl
import random

from matplotlib import pyplot as plt
import os
import spock_reg_model
import numpy as np
import argparse
from einops import rearrange
import utils
import pickle
from utils import assert_equal
import einops
import torch


LL_LOSS = """
function safe_log_erf(x)
    if x < -1
        (
            0.485660082730562*x + 0.643278438654541*exp(x)
            + 0.00200084619923262*x^3 - 0.643250926022749
            - 0.955350621183745*x^2
        )
    else
        log(1 + erf(x))
    end
end

function elementwise_loss(prediction, target)


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
            zero(mu)
            - (target - mu)^2 / (2 * sigma^2)
            - log(sigma)
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    end

    return -log_like
end
"""

LL_LOSS2 = """
function safe_log_erf(x)
    if x < -1
        (
            0.485660082730562*x + 0.643278438654541*exp(x)
            + 0.00200084619923262*x^3 - 0.643250926022749
            - 0.955350621183745*x^2
        )
    else
        log(1 + erf(x))
    end
end

function elementwise_loss(prediction, target)
    mu = prediction

    if mu < 1 || mu > 14
        # The farther away from a reasonable range, the more we punish it
        return 100 * (mu - 7)^2
    end

    sigma = one(prediction)

    # equation 8 in Bayesian neural network paper
    log_like = if target >= 9
        (
            safe_log_erf((mu - 9) / sqrt(2 * sigma^2))
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    else
        (
            zero(mu)
            - (target - mu)^2 / (2 * sigma^2)
            - log(sigma)
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    end

    return -log_like
end
"""

NN_STD_LOSS = """
function safe_log_erf(x)
    if x < -1
        (
            0.485660082730562*x + 0.643278438654541*exp(x)
            + 0.00200084619923262*x^3 - 0.643250926022749
            - 0.955350621183745*x^2
        )
    else
        log(1 + erf(x))
    end
end

function elementwise_loss(mu, sigma, target)
    if mu < 1 || mu > 14
        # The farther away from a reasonable range, the more we punish it
        return 100 * (mu - 7)^2
    end

    # equation 8 in Bayesian neural network paper
    log_like = if target >= 9
        (
            safe_log_erf((mu - 9) / sqrt(2 * sigma^2))
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    else
        (
            zero(mu)
            - (target - mu)^2 / (2 * sigma^2)
            - log(sigma)
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    end

    return -log_like
end

function my_loss(tree, dataset::Dataset{T,L}, options, idx)::L where {T,L}
    inds = idx === nothing ? Colon() : idx
    X = view(dataset.X, :, inds)
    y = view(dataset.y, inds)
    # sigma (per-sample NN std) is passed via the dataset.weights side-channel
    # so the tree doesn't see it as an input variable
    sigma = view(dataset.weights, inds)
    mu, ok = eval_tree_array(tree, X, options)
    if !ok
        return L(100_000)
    end

    loss = sum(
        i -> elementwise_loss(mu[i], sigma[i], y[i]),
        eachindex(mu)
    ) / length(mu)

    if loss < 0
        error("Loss is negative!")
    end

    return loss
end
"""

CLIPPED_LOSS = """
function elementwise_loss(prediction, target)
    return (target - min(prediction, 9))^2
end
"""

CLIPPED_LOSS2 = """
function elementwise_loss(prediction, target)
    return (target - max(4, min(prediction, 12)))^2
"""


SELECTIVE_EQ_LOSS_TEMPLATE = """
function my_loss(tree, dataset::Dataset{{T,L}}, options, idx)::L where {{T,L}}
    inds = idx === nothing ? Colon() : idx
    X = view(dataset.X, :, inds)
    y = view(dataset.y, inds)  # per-sample LL2 of frozen equation (sigma=1)

    scores, ok = eval_tree_array(tree, X, options)
    if !ok
        return L(100_000)
    end

    n = length(scores)
    k = max(1, Int(floor({p} * n)))
    perm = sortperm(scores)

    sum_loss = zero(L)
    for j in 1:k
        sum_loss += y[perm[j]]
    end
    return sum_loss / k
end
"""


def make_selective_eq_loss(p):
    assert 0 < p < 1, f"p must be in (0, 1), got {p}"
    return SELECTIVE_EQ_LOSS_TEMPLATE.format(p=float(p))


def _safe_log_erf_np(x):
    """Numpy port of safe_log_erf used in LL_LOSS2 / evaluation.lossfnc."""
    from scipy.special import erf
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    under = x < -1
    xu = x[under]
    out[under] = (
        0.485660082730562 * xu + 0.643278438654541 * np.exp(xu)
        + 0.00200084619923262 * xu**3 - 0.643250926022749
        - 0.955350621183745 * xu**2
    )
    xo = x[~under]
    out[~under] = np.log1p(erf(xo))
    return out


def ll2_per_sample(mu, y_true, sigma=1.0):
    """Per-sample negative log-likelihood (LL_LOSS2 formula, censored-aware).

    For y < 9   : NLL = (y-mu)^2/(2*var) + log(sigma) + safe_log_erf((mu-4)/sqrt(2*var))
    For y >= 9  : NLL = -[ safe_log_erf((mu-9)/...) - safe_log_erf((mu-4)/...) ]

    Returns float64 array of same shape as mu/y_true. Non-finite values are
    clamped to 100 (matches the safety mask in evaluation.lossfnc).
    """
    mu = np.asarray(mu, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sigma = float(sigma)
    var = sigma * sigma
    sqrt2var = np.sqrt(2 * var)

    arg4 = (mu - 4) / sqrt2var
    arg9 = (mu - 9) / sqrt2var
    sle4 = _safe_log_erf_np(arg4)
    sle9 = _safe_log_erf_np(arg9)

    # log_like under each branch
    reg_ll = -(y_true - mu) ** 2 / (2 * var) - np.log(sigma) - sle4
    cls_ll = sle9 - sle4

    geq9 = y_true >= 9
    log_like = np.where(geq9, cls_ll, reg_ll)
    nll = -log_like

    # Clamp non-finite (matches evaluation.lossfnc safety masks)
    nll = np.where(np.isfinite(nll), nll, 100.0)
    return nll

### predicting < 0 is a -1
### predicting >= 0 is a +1

### imbalanced/conservative loss
MODIFIED_PERCEPTRON_LOSS ='''
function elementwise_loss(x, y)
    if y == 1
        return x < 0 ? 1.0 : 0.0
    elseif y == -1
        return x < 0 ? 0.0 : -y * x + 1
    else
        error("y must be 1 or -1")
    end
end
'''

MODIFIED_ZEROONE_LOSS = '''
function elementwise_loss(x, y)
    if x == 0
        return y == -1 ? 1.0 : 0.0
    else
        return y * x < 0 ? 1.0 : 0.0
    end
end
'''


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


FORWARD_CACHE_NSAMPLES = 20000  # pool size; subsampled to config['n'] per run


def forward_cache_path(nn_version):
    return f'pickles/forward_cache_{nn_version}.pkl'


def build_forward_cache(nn_version, n_samples=FORWARD_CACHE_NSAMPLES):
    """Run the NN once, save (out_dict, y, summary_dim, latent) to a pickle.

    Subsequent SR runs read the pickle and skip the NN entirely — no GPU /
    torch CUDA needed. Re-run this (or delete the pickle) if the NN changes.
    """
    print(f"Building forward cache for nn_version={nn_version} ({n_samples} samples)...")
    model = spock_reg_model.load(version=nn_version)
    model.make_dataloaders()
    model.eval()

    data_iterator = iter(model.train_dataloader())
    x, y = next(data_iterator)
    while x.shape[0] < n_samples:
        try:
            next_x, next_y = next(data_iterator)
        except StopIteration:
            break
        x = torch.cat([x, next_x], dim=0)
        y = torch.cat([y, next_y], dim=0)

    with torch.no_grad():
        out_dict = model.forward(x, return_intermediates=True, noisy_val=False)

    out_dict_cpu = {}
    for k, v in out_dict.items():
        if isinstance(v, torch.Tensor):
            out_dict_cpu[k] = v.detach().cpu()
        else:
            out_dict_cpu[k] = v

    cache = {
        'out_dict': out_dict_cpu,
        'y': y.detach().cpu(),
        'summary_dim': model.summary_dim,
        'latent': model.hparams.get('latent'),
        'n_samples': x.shape[0],
        'nn_version': nn_version,
    }

    path = forward_cache_path(nn_version)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(cache, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved forward cache to {path} ({cache['n_samples']} samples, {size_mb:.1f} MB)")
    return cache


def load_or_build_forward_cache(nn_version):
    path = forward_cache_path(nn_version)
    if os.path.exists(path):
        print(f"Loading forward cache from {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)
    return build_forward_cache(nn_version)


def load_inputs_and_targets(config):
    cache = load_or_build_forward_cache(config['nn_version'])
    out_dict = cache['out_dict']
    y = cache['y']
    summary_dim = cache['summary_dim']
    latent = cache['latent']
    assert cache['n_samples'] >= config['n'], (
        f"forward cache has {cache['n_samples']} samples but --n={config['n']}. "
        f"Delete {forward_cache_path(config['nn_version'])} and re-run to rebuild "
        f"with more samples.")

    nn_std_arr = None
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
        out_dim = latent

        variable_names = INPUT_VARIABLE_NAMES
    elif config['target'] == 'f2':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # outputs are the (mean, std) predictions of the nn
        y = out_dict['prediction']  # [B, 2]
        y = y[:, 0:1]  # stop predicting std too haha


        in_dim = summary_dim
        out_dim = 1

        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

        if config['loss_fn'] == 'nn_std':
            std = out_dict['std']
            B, _ = X.shape
            assert_equal(std.shape, (B, 1))
            # Pass sigma out-of-band via the weights side-channel so the tree
            # doesn't see it as an input. Subsampled later alongside X, y.
            nn_std_arr = std.detach().numpy().astype(np.float32).flatten()

        if config['sr_residual']:
            raise ValueError("Implementation is currently broken")
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

        in_dim = summary_dim
        out_dim = 10

        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

    elif config['target'] in ['f2_direct', 'equation_bounds']:
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # target for SR is the ground truth mean, which we already have
        y = y  # [B, 2]

        n = X.shape[1] // 2
        in_dim = summary_dim
        out_dim = 1

        if config['loss_fn'] == 'nn_std':
            std = out_dict['std']
            B, _ = X.shape
            assert_equal(std.shape, (B, 1))
            # Pass sigma out-of-band via the weights side-channel; will be
            # repeat-interleaved below to mirror the X repeat.
            nn_std_arr = std.detach().numpy().astype(np.float32)  # [B, 1]

        # there are two ground truth predictions. create a data point for each
        X = einops.repeat(X, 'B F -> (B two) F', two=2)
        y = einops.rearrange(y, 'B two -> (B two) 1')
        if nn_std_arr is not None:
            nn_std_arr = einops.repeat(nn_std_arr, 'B one -> (B two) one', two=2).flatten()

        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

        if config['sr_residual']:
            raise ValueError("implementation is currently broken")

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

            in_dim = summary_dim + additional_features.shape[1]

        if config['target'] == 'equation_bounds':
            # for predicting whether or not an equation applies, learn equations
            # that predict when error is low

            with open(config['previous_sr_path'], 'rb') as f:
                model = pickle.load(f)

            # get the highest complexity equation
            max_complexity_idx = model.equations_[0]['complexity'].argmax()
            mean_equation = model.equations_[0].iloc[max_complexity_idx]

            # get the lambda eq for it, and evaluate on X
            X, y = X.detach().numpy(), y.detach().numpy()
            y_pred = mean_equation['lambda_format'](X)

            # calculate the error
            assert_equal(y_pred.shape, (X.shape[0], ))
            assert_equal(y.shape, (X.shape[0], 1))
            mse = (y - y_pred[:, None])**2
            assert_equal(mse.shape, y.shape)

            # predict with binary classification whether equation applies or not
            # to use ZeroOneLoss, need +/- 1 targets
            # https://astroautomata.com/SymbolicRegression.jl/dev/losses/
            y = np.zeros_like(mse)
            y[mse > config['eq_bound_mse_threshold']] = -1
            y[mse <= config['eq_bound_mse_threshold']] = 1
            print(f'Equation bound data has {(y == -1).sum()} oob targets and {(y == 1).sum()} good-predicting targets')

    elif config['target'] == 'selective_eq':
        # Learn an equation that predicts which subset of points the frozen
        # baseline equation predicts accurately. Hardcoded baseline for v1:
        # pysr_version=11003, model_selection=26 (complexity).
        PREV_PYSR_PATH = 'sr_results/11003.pkl'
        PREV_MODEL_SELECTION = 26

        X_np = out_dict['summary_stats'].detach().numpy()  # [B, 40]
        y_true = y.detach().numpy()                        # [B, 2]
        assert_equal(y_true.shape[1], 2)

        prev_reg = pysr.PySRRegressor.from_file(PREV_PYSR_PATH)
        # 11003 was trained with 2 outputs (mean, std); we only use the mean
        # equation here for prediction error.
        eqs_df = prev_reg.equations_[0] if isinstance(prev_reg.equations_, list) else prev_reg.equations_
        complexities = eqs_df['complexity'].values
        ix = int(np.argmin(np.abs(complexities - PREV_MODEL_SELECTION)))
        prev_eq = eqs_df.iloc[ix]
        print(f"selective_eq: using baseline mean equation at row {ix}, "
              f"complexity={prev_eq['complexity']}, loss={prev_eq['loss']:.4f}")
        y_pred = prev_eq['lambda_format'](X_np)            # [B]
        assert_equal(y_pred.shape, (X_np.shape[0],))

        # Per-system avg LL2 (negative log-likelihood with sigma=1) over both
        # ground truths. Captures both the regression error (y<9 branch) and
        # the censored classification error (y>=9 branch).
        ll_1 = ll2_per_sample(y_pred, y_true[:, 0], sigma=1.0)
        ll_2 = ll2_per_sample(y_pred, y_true[:, 1], sigma=1.0)
        per_system_ll2 = (ll_1 + ll_2) / 2.0
        y = per_system_ll2.reshape(-1, 1).astype(np.float32)  # [B, 1]
        X = X_np
        print(f"selective_eq target stats: min={y.min():.4f} max={y.max():.4f} "
              f"mean={y.mean():.4f} median={np.median(y):.4f}")

        in_dim = summary_dim
        out_dim = 1
        n_pairs = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n_pairs)] + [f's{i}' for i in range(n_pairs)]

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
    if nn_std_arr is not None:
        nn_std_arr = nn_std_arr[ixs]

    # Ensure X and y are NumPy arrays
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().numpy()

    assert_equal(X.shape, (config['n'], in_dim))
    assert_equal(y.shape, (config['n'], out_dim))
    if nn_std_arr is not None:
        assert_equal(nn_std_arr.shape, (config['n'],))

    return X, y, variable_names, nn_std_arr


def get_config(args):
    if args.version is None:
        version = random.randint(0, 100000)
        while os.path.exists(f'sr_results/{version}.pkl'):
            version = random.randint(0, 100000)
    else:
        version = args.version

    path = f'sr_results/{version}.csv'
    # create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # https://stackoverflow.com/a/57474787/4383594
    try:
        num_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES'))
    except TypeError:
        num_cpus = 10

    pysr_config = dict(
        procs=num_cpus,
        populations=3*num_cpus,
        batching=True,
        batch_size=args.batch_size,
        equation_file=path,
        niterations=args.niterations,
        binary_operators=["+", "*", '/', '-', '^'],
        # unary_operators=['sin'],
        maxsize=args.max_size,
        timeout_in_seconds=int(60*60*args.time_in_hours),
        # prevent ^ from using complex exponents, nesting power laws is expressive but uninterpretable
        # base can have any complexity, exponent can have max 1 complexity
        constraints={'^': (-1, 1)},
        # nested_constraints={"sin": {"sin": 0}},
        ncyclesperiteration=1000, # increase utilization since usually using 32-ish cores?
        random_state=args.seed,
    )

    if args.loss_fn == 'll':
        pysr_config['elementwise_loss'] = LL_LOSS
    elif args.loss_fn == 'll2':
        pysr_config['elementwise_loss'] = LL_LOSS2
    elif args.loss_fn == 'clipped':
        pysr_config['elementwise_loss'] = CLIPPED_LOSS
    elif args.loss_fn == 'clipped2':
        pysr_config['elementwise_loss'] = CLIPPED_LOSS2
    elif args.loss_fn == 'nn_std':
        jl.seval(NN_STD_LOSS)
        pysr_config['loss_function'] = 'my_loss'
        pysr_config['elementwise_loss'] = None
    elif args.loss_fn == 'selective_eq':
        jl.seval(make_selective_eq_loss(args.p))
        pysr_config['loss_function'] = 'my_loss'
        pysr_config['elementwise_loss'] = None

    if args.target == 'equation_bounds':
        pysr_config['elementwise_loss'] = MODIFIED_ZEROONE_LOSS
        if args.loss_fn == 'perceptron':
            pysr_config['elementwise_loss'] = MODIFIED_PERCEPTRON_LOSS

    config = vars(args)
    config.update(pysr_config)
    config['pysr_config'] = pysr_config
    config.update({
        'version': version,
        'slurm_id': os.environ.get('SLURM_JOB_ID', None),
        'slurm_name': os.environ.get('SLURM_JOB_NAME', None),
    })

    return config


def run_pysr(config):
    command = utils.get_script_execution_command()
    print(command)

    X, y, variable_names, nn_std_arr = load_inputs_and_targets(config)

    model = pysr.PySRRegressor(**config['pysr_config'])

    if not config['no_log']:
        wandb.init(
            entity='bnn-chaos-model',
            project='planets-sr',
            config=config,
        )

    # nn_std loss reads sigma from dataset.weights inside the Julia loss fn
    model.fit(X, y, variable_names=variable_names, weights=nn_std_arr)
    print('Done running pysr')

    losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]

    if not config['no_log']:
        wandb.log({'avg_loss': sum(losses)/len(losses),
                   'losses': losses,
                   })

    print(f"Saved to path: {config['equation_file']}")
    print(f'Finished running pysr with pysr_version {config["version"]}')


def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    # when importing from jupyter nb, it passes an arg to --f which we should just ignore
    parser.add_argument('--no_log', action='store_true', default=False, help='disable wandb logging')
    parser.add_argument('--nn_version', type=int, help='', required=True)
    parser.add_argument('--version', type=int, default=None)
    parser.add_argument('--time_in_hours', type=float, default=8)
    parser.add_argument('--niterations', type=float, default=500000) # by default, use time in hours as limit
    parser.add_argument('--max_size', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', type=str, default='f2', choices=['f1', 'f2', 'f2_ifthen', 'f2_direct', 'equation_bounds', 'selective_eq'])
    parser.add_argument('--residual', action='store_true', help='do residual training of your target')
    parser.add_argument('--n', type=int, default=10000, help='number of data points for the SR problem')
    parser.add_argument('--batch_size', type=int, default=1000, help='number of data points for the SR problem')
    parser.add_argument('--sr_residual', action='store_true', help='do residual training of your target with previous sr run as base')
    parser.add_argument('--loss_fn', type=str, choices=['mse', 'll', 'perceptron', 'clipped', 'clipped2', 'll2', 'nn_std', 'selective_eq'], default='mse')
    parser.add_argument('--p', type=float, default=0.5, help='fraction of points to select (for loss_fn=selective_eq); 0<p<1')
    parser.add_argument('--previous_sr_path', type=str, default='sr_results/92985.pkl', help='path to previous sr run, used for residual/recursive training')
    parser.add_argument('--eq_bound_mse_threshold', type=float, default=1, help='mse threshold below which to consider an equation good')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args)
    run_pysr(config)
