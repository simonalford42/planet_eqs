import pickle as pkl

import numpy as np
import click
from pysr import PySRRegressor, jl
import os
import wandb
import subprocess
import random
import spock_reg_model
import torch
import einops

def special_loss(T):
    s = """
        function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
            if tree.degree != 2
                return L(1_000_000)
            end

            function safe_log_erf(x)
                if x < -1
                    T(0.485660082730562)*x + T(0.643278438654541)*exp(x)
                    + T(0.00200084619923262)*x^3 - T(0.643250926022749)
                    - T(0.955350621183745)*x^2
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
                        zero(prediction)
                        - (target - mu)^2 / (2 * sigma^2)
                        - log(sigma)
                        - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
                    )
                end

                return -log_like
            end

            f1 = tree.l
            f2 = tree.r

            X = dataset.X  # (feature, batch * time)
            y = dataset.y  # (batch * time)
            true_times = y[begin:NUM_TIMES:end]

            prediction_flat_f1, complete_f1 = eval_tree_array(f1, X, options)
            if !complete_f1
                return L(100_000)
            end

            prediction_flat_f1 # (batch * time)

            function _mean(f::F, v; dims) where {F}
                sum(f, v, dims=dims) / prod(d -> size(v, d), dims)
            end
            function _var(v; dims)
                _mean(vi -> vi^2, v; dims=dims) - _mean(identity, v; dims=dims) .^ 2
            end
            function _std(v; dims)
                va = _var(v; dims=dims)
                all(vi -> isfinite(vi) && vi >= 0, va) ? sqrt.(va) : (eltype(v)(NaN) .* v)
            end

            prediction_f1 = reshape(prediction_flat_f1, (NUM_TIMES, :))
            # ^ Reshape to (time, batch)
            stdev_f1 = _std(prediction_f1, dims=1)
            # ^ (batch,)

            if any(x -> !isfinite(x), stdev_f1)
                return L(50_000)
            end

            input_to_f2 = zero(@view X[:, begin:NUM_TIMES:end])
            for feature in eachindex(axes(X, 1))
                input_to_f2[feature, :] .= stdev_f1[1, :]
            end

            prediction_f2, complete_f2 = eval_tree_array(f2, input_to_f2, options)
            if !complete_f2
                return L(10_000)
            end
            prediction_f2 # (batch,)

            loss = sum(
                i -> elementwise_loss(prediction_f2[i], true_times[i]),
                eachindex(prediction_f2, true_times)
            ) / length(true_times)

            if loss < 0
                error("Loss is negative!")
            end

            return loss
        end
        """
    return s.replace("NUM_TIMES", str(T))


def mse_loss(T):
    s = """
        function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
            if tree.degree != 2
                return L(1_000_000)
            end

            function elementwise_loss(prediction, target)
                return (prediction - target)^2
            end

            f1 = tree.l
            f2 = tree.r

            X = dataset.X  # (feature, batch * time)
            y = dataset.y  # (batch * time)
            true_times = y[begin:NUM_TIMES:end]

            prediction_flat_f1, complete_f1 = eval_tree_array(f1, X, options)
            if !complete_f1
                return L(100_000)
            end

            prediction_flat_f1 # (batch * time)

            function _mean(f::F, v; dims) where {F}
                sum(f, v, dims=dims) / prod(d -> size(v, d), dims)
            end
            function _var(v; dims)
                _mean(vi -> vi^2, v; dims=dims) - _mean(identity, v; dims=dims) .^ 2
            end
            function _std(v; dims)
                va = _var(v; dims=dims)
                all(vi -> isfinite(vi) && vi >= 0, va) ? sqrt.(va) : (eltype(v)(NaN) .* v)
            end

            prediction_f1 = reshape(prediction_flat_f1, (NUM_TIMES, :))
            # ^ Reshape to (time, batch)
            stdev_f1 = _std(prediction_f1, dims=1)
            # ^ (batch,)

            if any(x -> !isfinite(x), stdev_f1)
                return L(50_000)
            end

            input_to_f2 = zero(@view X[:, begin:NUM_TIMES:end])
            for feature in eachindex(axes(X, 1))
                input_to_f2[feature, :] .= stdev_f1[1, :]
            end

            prediction_f2, complete_f2 = eval_tree_array(f2, input_to_f2, options)
            if !complete_f2
                return L(10_000)
            end
            prediction_f2 # (batch,)

            loss = sum(
                i -> elementwise_loss(prediction_f2[i], true_times[i]),
                eachindex(prediction_f2, true_times)
            ) / length(true_times)

            if loss < 0
                error("Loss is negative!")
            end

            return loss
        end
        """
    return s.replace("NUM_TIMES", str(T))



def load_data(n):
    # use input data that's the same as the neural network (already normalized, etc)
    # this makes it easier to evaluate and compare to my other approaches
    model = spock_reg_model.load(24880)
    model.make_dataloaders()
    model.eval()

    data_iterator = iter(model.train_dataloader())
    x, y = next(data_iterator)
    while x.shape[0] < n:
        next_x, next_y = next(data_iterator)
        x = torch.cat([x, next_x], dim=0)
        y = torch.cat([y, next_y], dim=0)

    x = x.cuda()
    model = model.cuda()
    out_dict = model.forward(x, return_intermediates=True, noisy_val=False)
    # inputs to SR are the inputs to the whole system
    X = out_dict['inputs']  # [B, T, N]

    # take every 10th sample on the time axis
    X = X[:, ::10, :]

    # there are two ground truth predictions. create a data point for each
    X = einops.repeat(X, 'B ... -> (B two) ...', two=2)
    y = einops.rearrange(y, 'B two -> (B two)')

    ixs = np.random.choice(X.shape[0], size=n, replace=False)
    X, y, = X[ixs], y[ixs]

    Xflat = einops.rearrange(X, "B T f -> (B T) f")
    yflat = einops.repeat(y, "B  -> (B T)", T=X.shape[1])

    Xflat, yflat = Xflat.cpu().detach().numpy(), yflat.cpu().detach().numpy()

    return X, Xflat, yflat


def option(*param_decls, **attrs):
    attrs.setdefault("show_default", True)
    return click.option(*param_decls, **attrs)

@click.command()
@option("--procs", type=int, default=1)
@option("--niterations", type=int, default=100000)
@option("--populations", type=int, default=1)
@option("--population-size", type=int, default=33)
@option("--ncycles-per-iteration", type=int, default=550)
@option("--distributed/--no-distributed", default=True)
@option("--multithreading/--no-multithreading", default=False)
@option("--weight-optimize", type=float, default=0.001)
@option("--seed", type=int, default=0)
# choices = ['mse', 'special']
@option("--loss_fn", type=str, default='mse')
@option("--parsimony", type=float, default=0.01)
@option("--maxsize", type=int, default=50)
@option("--no-log", default=False)
@option("--time-in-hours", type=float, default=240)
@option("--heap-size-hint-in-bytes", type=int, default=300_000_000)
@option("--n", type=int, default=1000)
@option(
    "--binary-operators",
    type=str,
    default="+;*;/;^",
    help="List of binary operators to use.",
)
@option(
    "--unary-operators",
    type=str,
    default="sin",
    help="List of unary operators to use.",
)
def main(
    procs,
    niterations,
    populations,
    population_size,
    ncycles_per_iteration,
    distributed,
    multithreading,
    weight_optimize,
    maxsize,
    heap_size_hint_in_bytes,
    parsimony,
    binary_operators,
    unary_operators,
    no_log,
    time_in_hours,
    n,
    seed,
    loss_fn,
):

    # split into lists
    unary_operators = unary_operators.split(';')
    binary_operators = binary_operators.split(';')

    X, Xflat, yflat = load_data(n)

    if loss_fn == 'mse':
        loss_f = mse_loss
    else:
        assert loss_fn == 'special', "loss_fn must be 'mse' or 'special'"
        loss_f = special_loss

    jl.seval(
        loss_f(X.shape[1]),
    )

    # default_nested = {op: 1 for op in unary_operators}
    # default_interactions = {
    #     "cos": {**default_nested, "cos": 0},
    #     "sqrt": {**default_nested, "sqrt": 0, "cbrt": 0},
    #     "cbrt": {**default_nested, "cbrt": 0, "sqrt": 0},
    #     "square": {**default_nested, "square": 1, "cube": 1},
    #     "cube": {**default_nested, "cube": 1, "square": 1},
    #     "exp": {**default_nested, "exp": 0, "log": 0},
    #     "log": {**default_nested, "log": 0, "exp": 0, "cos": 0, "cbrt": 0},
    # }
    # nested_constraints = {}
    # for op in unary_operators:
    #     nested_constraints[op] = {
    #         nest_op: nestedness
    #         for nest_op, nestedness in default_interactions[op].items()
    #         if nest_op in unary_operators
    #     }

    id = random.randint(0, 100000)
    while os.path.exists(f'sr_results/{id}.pkl'):
        id = random.randint(0, 100000)

    config = {
        'equation_file': f'sr_results/{id}.csv',
        'id': id,
        'pure_sr': True,
        'slurm_id': os.environ.get('SLURM_JOB_ID', None),
        'slurm_name': os.environ.get('SLURM_JOB_NAME', None),
        'time_in_hours': time_in_hours,
        'loss_fn': loss_fn,
    }

    try:
        if not no_log:
            wandb.init(
                entity='bnn-chaos-model',
                project='planets-sr',
                config=config,
            )

        model = PySRRegressor(
            equation_file=config['equation_file'],
            niterations=niterations,
            populations=populations,
            population_size=population_size,
            ncycles_per_iteration=ncycles_per_iteration,
            procs=procs,
            multithreading=multithreading,
            # cluster_manager="slurm" if distributed else None,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            maxsize=maxsize,
            weight_optimize=weight_optimize,
            parsimony=parsimony,
            adaptive_parsimony_scaling=1000.0,
            # turbo=False,
            # bumper=False,
            # nested_constraints=nested_constraints,
            constraints={'^': (-1, 1)},
            nested_constraints={"sin": {"sin": 0}},
            random_state=seed,
            loss_function="my_loss",
            heap_size_hint_in_bytes=heap_size_hint_in_bytes,
            timeout_in_seconds=int(60*60*time_in_hours),
        )

        labels = pkl.load(open("./data/combined.pkl", "rb"))["labels"]
        clean_variables = list(
            map(lambda label: label.replace("+", "p").replace("-", "m"), labels)
        )

        model.fit(Xflat, yflat, variable_names=clean_variables)

        losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]
        if not no_log:
            wandb.log({'avg_loss': sum(losses)/len(losses),
                       'losses': losses,
                       })

        print(f"Saved to path: {config['equation_file']}")

    except Exception as e:
        print(f"An error occurred: {e}")
        # print the stack trace
        import traceback
        traceback.print_exc()

    finally:
        try:
            # delete julia files: julia-1911988-17110333239-0016.out
            subprocess.run(f'rm julia*.out', shell=True, check=True)
        except subprocess.CalledProcessError as e:
            pass


if __name__ == "__main__":
    main()
