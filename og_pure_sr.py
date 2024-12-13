import pickle as pkl
from pure_sr import load_data as load_data2

import click
import einops as E
import numpy as np
from pysr import PySRRegressor, jl


def load_data():
    data = pkl.load(open("./data/combined.pkl", "rb"))
    X = data["X"]  # (N, 100, 41)
    y = data["y"]  # (N, 2)

    rng = np.random.RandomState(0)
    factor = 0.005

    mask = rng.choice(a=[False, True], size=X.shape[0], p=[1 - factor, factor])

    X = X[mask, ::10]
    y = y[mask]
    # (1142, 10, 41)

    X = E.repeat(X, "batch time feature -> (batch shadow) time feature", shadow=2)
    y = E.rearrange(y, "batch shadow -> (batch shadow)", shadow=2)

    Xflat = E.rearrange(
        X,
        "batch time feature -> (batch time) feature",
    )
    yflat = E.repeat(y, "batch -> (batch time)", time=X.shape[1])

    return X, Xflat, yflat


def option(*param_decls, **attrs):
    attrs.setdefault("show_default", True)
    return click.option(*param_decls, **attrs)


@click.command()
@option("--procs", type=int, default=1)
@option("--niterations", type=int, default=None)
@option("--populations", type=int, default=None)
@option("--population-size", type=int, default=33)
@option("--ncycles-per-iteration", type=int, default=550)
@option("--distributed/--no-distributed", default=False)
@option("--multithreading/--no-multithreading", default=False)
@option("--weight-optimize", type=float, default=0.001)
@option("--parsimony", type=float, default=0.01)
@option("--maxsize", type=int, default=50)
@option("--heap-size-hint-in-bytes", type=int, default=300_000_000)
@option(
    "--binary-operators",
    type=str,
    default="+;*;/",
    help="List of binary operators to use.",
)
@option(
    "--unary-operators",
    type=str,
    default="cos;sqrt;cbrt;square;cube;exp;log",
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
):

    if niterations is None:
        niterations = int(1e12)
    if populations is None:
        if procs is not None:
            populations = 3 * procs
        else:
            populations = 15

    binary_operators = binary_operators.split(";")
    unary_operators = unary_operators.split(";")

    X, Xflat, yflat, labels = load_data()

    jl.seval(
        """
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
    """.replace(
            "NUM_TIMES", str(X.shape[1])
        )
    )

    default_nested = {op: 1 for op in unary_operators}
    default_interactions = {
        "cos": {**default_nested, "cos": 0},
        "sqrt": {**default_nested, "sqrt": 0, "cbrt": 0},
        "cbrt": {**default_nested, "cbrt": 0, "sqrt": 0},
        "square": {**default_nested, "square": 1, "cube": 1},
        "cube": {**default_nested, "cube": 1, "square": 1},
        "exp": {**default_nested, "exp": 0, "log": 0},
        "log": {**default_nested, "log": 0, "exp": 0, "cos": 0, "cbrt": 0},
    }
    nested_constraints = {}
    for op in unary_operators:
        nested_constraints[op] = {
            nest_op: nestedness
            for nest_op, nestedness in default_interactions[op].items()
            if nest_op in unary_operators
        }

    model = PySRRegressor(
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
        turbo=True,
        bumper=True,
        nested_constraints=nested_constraints,
        loss_function="my_loss",
        heap_size_hint_in_bytes=heap_size_hint_in_bytes,
    )

    labels = pkl.load(open("./data/combined.pkl", "rb"))["labels"]
    clean_variables = list(
        map(lambda label: label.replace("+", "p").replace("-", "m"), labels)
    )

    model.fit(Xflat, yflat, variable_names=clean_variables)


if __name__ == "__main__":
    main()
