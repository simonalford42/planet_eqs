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
from sr import LL_LOSS

def ll_loss(T):
    s = """
        function my_loss(tree, dataset::Dataset{T,L}, options, idx)::L where {T,L}
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
        function my_loss(tree, dataset::Dataset{T,L}, options, idx)::L where {T,L}
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
@option("--niterations", type=int, default=100000)
@option("--seed", type=int, default=0)
@option("--loss_fn", type=click.Choice(['mse', 'll']), default='ll')
@option("--maxsize", type=int, default=60)
@option("--batch-size", type=int, default=1000)
@option("--n", type=int, default=10000)
@option("--version", type=int, default=None)
@option("--log/--no_log", default=True)
@option("--time_in_hours", type=float, default=8)
def main(
    niterations,
    maxsize,
    batch_size,
    log,
    time_in_hours,
    n,
    seed,
    loss_fn,
    version,
):

    X, Xflat, yflat = load_data(n)

    loss_f = mse_loss if loss_fn == 'mse' else ll_loss
    jl.seval(loss_f(X.shape[1]))
    kwargs = {'loss_function': 'my_loss'}

    if version is None:
        version = random.randint(0, 100000)
        while os.path.exists(f'sr_results/{version}.pkl'):
            version = random.randint(0, 100000)

        # touch the pkl file so that it exists
        with open(f'sr_results/{version}.pkl', 'wb') as f:
            pkl.dump({}, f)


    try:
        num_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES'))
    except TypeError:
        num_cpus = 10

    config = {
        'equation_file': f'sr_results/{version}.csv',
        ' version': version,
        'pure_sr': True,
        'slurm_id': os.environ.get('SLURM_JOB_ID', None),
        'slurm_name': os.environ.get('SLURM_JOB_NAME', None),
        'time_in_hours': time_in_hours,
        'loss_fn': loss_fn,
        'seed': seed,
    }

    try:
        if log:
            wandb.init(
                entity='bnn-chaos-model',
                project='planets-sr',
                config=config,
            )

        kwargs = {'loss_function': 'my_loss'}

        model = PySRRegressor(
            procs=num_cpus,
            populations=3*num_cpus,
            batching=True,
            batch_size=batch_size,
            equation_file=config['equation_file'],
            niterations=niterations,
            binary_operators=["+", "*", '/', '-', '^'],
            # unary_operators=['sin'],
            maxsize=maxsize,
            timeout_in_seconds=int(60*60*time_in_hours),
            constraints={'^': (-1, 1)},
            # prevent ^ from using complex exponents, nesting power laws is expressive but uninterpretable
            # base can have any complexity, exponent can have max 1 complexity
            # nested_constraints={"sin": {"sin": 0}},
            ncycles_per_iteration=1000,
            random_state=seed,
            **kwargs,
        )

        labels = pkl.load(open("./data/combined.pkl", "rb"))["labels"]
        clean_variables = list(
            map(lambda label: label.replace("+", "p").replace("-", "m"), labels)
        )

        model.fit(Xflat, yflat, variable_names=clean_variables)

        losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]
        if log:
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
