from matplotlib import pyplot as plt
from interpret import paretoize, overall_complexity
from utils import load_pickle, load_json
import argparse

SPLIT = 'test'


def get_k_results(k_version_dict, overall=True):
    k_results = {}
    for k, v in k_version_dict.items():
        k = int(k)
        # load the original PySR table
        pysr_version = v['pysr_version']
        version = v['version']
        pysr_results = load_pickle(f'sr_results/{pysr_version}.pkl').equations_
        all_rmses = load_pickle(f'pickles/pysr_results_all_{version}_{pysr_version}.pkl')[SPLIT]

        results = {}
        for comp in all_rmses.keys():
            eq = pysr_results[pysr_results['complexity'] == comp].iloc[0]
            if overall:
                overall_comp = overall_complexity(eq, k)
                results[overall_comp] = all_rmses[comp]
            else:
                results[comp] = all_rmses[comp]

        k_results[k] = results
    return k_results


def nn_test_rmse(version):
    return load_pickle(f'pickles/nn_results_all_{version}.pkl')[SPLIT]


def overall_complexity_f2_linear(n_features):
    '''
    - each feature is mean/std of a k=2 input feature, which has complexity 2.
    - we do a linear combination of the features, so that is complexity 3 * n_features (2 for each feature, 1 for the product)
    - also we have a bias term, which adds 1
    '''
    return 3 * n_features + 1


def get_f2_linear_results(f2_linear_models):
    return {
        overall_complexity_f2_linear(int(k)): nn_test_rmse(v) for k, v in f2_linear_models.items()
    }


def plot_combined_pareto(
    k_results,
    f2_linear_results,
    pure_sr_results,
    pure_sr2_results,
    path,
):
    """
    Combine the `plot_k` and `plot_all` panels into one stacked figure and save it.

    Parameters
    ----------
    k_results
        Mapping k -> {complexity: rmse}.
    f2_linear_results
        Mapping {complexity: rmse} for the “Linear ψ” baseline.
    pure_sr_results
        Mapping {complexity: rmse} for the “Pure SR” baseline.
    pure_sr2_results
        Mapping {complexity: rmse} for the “Pure SR (no intermediate features)” baseline.
    path
        Destination path.
    """
    plt.rcParams["font.family"] = "serif"

    # --- Layout ------------------------------------------------------------
    fig, axs = plt.subplots(
        nrows = 2,
        ncols = 1,
        figsize = (6, 8),
        sharex = False,
        # constrained_layout = True,
        gridspec_kw={"hspace": 0.25}
    )

    # ---------- Top panel: all k -------------------------------------------
    ax = axs[0]
    for k, result in k_results.items():
        x, y = zip(*result.items())
        x, y = paretoize(x, y, replace=False)
        label = f"$k = {k}$" if k != 2 else "$k = 2$ (Ours)"
        ax.plot(x, y, marker = "^", label = label)

    first_line = ax.lines[0]     # or keep a handle returned by ax.plot
    first_line.set_zorder(10)    # any value larger than the others

    ax.set_xlabel("Overall complexity", fontsize = 12, labelpad = 6)
    ax.set_ylabel("RMSE (Resonant)", fontsize = 12, labelpad = 6)
    # ax.set_title("Top‑k Pareto fronts", fontsize = 14, pad = 8)
    ax.legend()

    # ---------- Bottom panel: comparison -----------------------------------
    ax = axs[1]

    # k = 2 (ours)
    x, y = zip(*k_results[2].items())
    x, y = paretoize(x, y, replace=False)
    ax.plot(x, y, marker = "^", label = "Ours")

    # linear ψ baseline
    x, y = zip(*f2_linear_results.items())
    x, y = paretoize(x, y, replace=False)
    ax.plot(x, y, marker = "o", label = "Linear $\\psi$")

    # pure SR 2 baseline
    x, y = zip(*pure_sr2_results.items())
    x, y = paretoize(x, y, replace=False)
    ax.plot(x, y, marker = "o", label = "Pure SR (no intermediate features)")

    # pure SR baseline
    x, y = zip(*pure_sr_results.items())
    x, y = paretoize(x, y, replace=False)
    ax.plot(x, y, marker = "o", label = "Pure SR")


    first_line = ax.lines[0]     # or keep a handle returned by ax.plot
    first_line.set_zorder(10)    # any value larger than the others

    ax.set_xlabel("Overall complexity", fontsize=12, labelpad=6)
    ax.set_ylabel("RMSE (Resonant)", fontsize=12, labelpad=6)
    ax.legend()

    # --- Save & return ------------------------------------------------------
    fig.savefig(path, bbox_inches = "tight", dpi = 400)
    print('Saved combined figure to', path)
    plt.close(fig)
    return fig, axs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='graphics/pareto_comparison.pdf', help='Path to save the combined figure')
    parser.add_argument('--version_json', type=str, default='official_model_versions.json', help='Path to the JSON file containing model versions')
    return parser.parse_args()


def main():
    args = get_args()
    version_dict = load_json(args.version_json)
    k_results = get_k_results(version_dict['k'])
    f2_linear_results = get_f2_linear_results(version_dict['f2_linear'])
    pure_sr_results = load_pickle(f'pickles/pure_sr_results_all_{version_dict["mse_pure_sr_version"]}.pkl')[SPLIT]
    # get rid of entries with rmse over 2.0, because they were probably invalid equations
    pure_sr_results = {k: v for k, v in pure_sr_results.items() if v < 2.0}
    pure_sr2_results = load_pickle(f'pickles/pysr_results_all_28114_{version_dict["mse_pure_sr2_version"]}.pkl')[SPLIT]
    plot_combined_pareto(k_results, f2_linear_results, pure_sr_results, pure_sr2_results, args.path)


if __name__ == "__main__":
    main()
