from matplotlib import pyplot as plt
import pandas as pd
from interpret import paretoize, number_of_variables_in_expression
from utils import load_pickle, load_json, save_pickle
import argparse
import numpy as np
from matplotlib.ticker import MultipleLocator
plt.style.use('seaborn-darkgrid')

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
            overall_comp = get_complexity(eq, k, overall=overall)
            results[overall_comp] = all_rmses[comp]

        k_results[k] = results
    return k_results


def nn_test_rmse(version):
    return load_pickle(f'pickles/nn_results_all_{version}.pkl')[SPLIT]


def f2_linear_complexity(n_features, overall=True):
    '''

    - each feature has complexity 2, because it's a k=2 input feature.
    - each feature used in the linear combination has complexity 3: 2 for the complexity of the feature itself, and 1 for the linear combination value.
    - the bias term adds 1 to the overall complexity.
    - so overall commplexity 3 * n_features + 1

    Example: x1 = a1 i1 + a2 i2 has complexity 2 (each
        y = b1 x1 + b2 x2 + c has complexity 3 ignoring the features themselves, plus 4 (complexity 2 for each feature x_i)
    '''

    if overall:
        return 2 * n_features + 1
    else:
        # learned features have complexity 1 instead.
        # so overall complexity is just 2 * n_features + 1
        return n_features + 1


def get_f2_linear_results(f2_linear_models, overall=True):
    return {
        f2_linear_complexity(int(k), overall=overall): nn_test_rmse(v)
        for k, v in f2_linear_models.items()
    }


def get_complexity(entry: pd.Series, k: int, overall=True):
    '''
    - each feature has complexity k, because it's a k input features (k free parameters).
    - overall complexity is Pysr equattion complexity + complexity of the features
    - but the features are already counted in the pysr complexity as complexity 1, so subtract 1
    '''
    if overall:
        complexity = entry['complexity'].item()
        num_variables = number_of_variables_in_expression(entry.equation)
        return complexity + (k - 1) * num_variables
    else:
        return entry['complexity'].item()


def plot_combined_pareto(
    k_results,
    f2_linear_results,
    pure_sr_results,
    pure_sr2_results,
    path,
):
    plt.rcParams["font.family"] = "serif"

    # --- Layout ------------------------------------------------------------
    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6, 8),
        sharex=False,
        # constrained_layout = True,
        gridspec_kw={"hspace": 0.25}
    )

    # ---------- Top panel: all k -------------------------------------------
    ax = axs[0]

    for k, result in k_results.items():
        x, y = zip(*_extract_metric(result, 'rmse').items())
        x, y = paretoize(x, y, replace=False)
        label = f"$k = {k}$" if k != 2 else "$k = 2$ (Ours)"
        ax.plot(x, y, marker = "^", label = label)

    first_line = ax.lines[0]     # or keep a handle returned by ax.plot
    first_line.set_zorder(10)    # any value larger than the others

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("Overall complexity", fontsize = 12, labelpad = 6)
    ax.set_ylabel("Resonant RMSE (dex)", fontsize = 12, labelpad = 6)
    ax.legend(framealpha=1)

    # ---------- Middle panel: comparison -----------------------------------
    ax = axs[1]

    # k = 2 (ours)
    x, y = zip(*_extract_metric(k_results[2], 'rmse').items())
    x, y = paretoize(x, y, replace=False)
    ax.plot(x, y, marker = "^", label = "Ours")

    # linear ψ baseline
    x, y = zip(*_extract_metric(f2_linear_results, 'rmse').items())
    x, y = paretoize(x, y, replace=False)
    ax.plot(x, y, marker = "o", label = "Linear $\\psi$")

    # pure SR 2 baseline
    x, y = zip(*_extract_metric(pure_sr2_results, 'rmse').items())
    x, y = paretoize(x, y, replace=False)
    ax.plot(x, y, marker = "o", label = "Pure SR (no intermediate features)")

    # pure SR baseline
    x, y = zip(*_extract_metric(pure_sr_results, 'rmse').items())
    x, y = paretoize(x, y, replace=False)
    ax.plot(x, y, marker = "o", label = "Pure SR")

    first_line = ax.lines[0]     # or keep a handle returned by ax.plot
    first_line.set_zorder(10)    # any value larger than the others

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("Overall complexity", fontsize = 12, labelpad = 6)
    ax.set_ylabel("Resonant RMSE (dex)", fontsize = 12, labelpad = 6)
    ax.legend(framealpha=1)

    # --- Save & return ------------------------------------------------------
    fig.savefig(path, bbox_inches = "tight", dpi = 400)
    print('Saved combined figure to', path)
    plt.close(fig)
    return fig, axs


def compute_results(version_json='official_versions.json'):
    """Compute all results needed for pareto plots on the resonant test split.

    Uses evaluation.calculate_metrics which caches truths/preds automatically.
    Returns dict with keys 'k', 'f2_linear', 'pure_sr', 'pure_sr2',
    each mapping complexity -> metrics dict.
    """
    from evaluation import calculate_metrics
    from types import SimpleNamespace
    from interpret import get_pysr_results as _get_pysr_results
    from pure_sr_evaluation import get_pure_sr_results as _get_pure_sr_results

    version_dict = load_json(version_json)

    # --- k-variant PySR results ---
    print("=" * 60)
    print("Computing k-variant PySR results...")
    k_results = {}
    for k_str, v in version_dict['k'].items():
        k = int(k_str)
        pysr_version = v['pysr_version']
        version = v['version']
        pysr_table = _get_pysr_results(pysr_version)
        complexities = sorted(pysr_table['complexity'].tolist())
        print(f"  k={k} (version={version}, pysr={pysr_version}): {len(complexities)} complexities")

        k_result = {}
        for comp in complexities:
            args = SimpleNamespace(
                eval_type='pysr', version=version,
                pysr_version=pysr_version,
                pysr_model_selection=str(comp),
                dataset='test',
            )
            metrics = calculate_metrics(args)
            eq = pysr_table[pysr_table['complexity'] == comp].iloc[0]
            overall_comp = get_complexity(eq, k, overall=True)
            k_result[overall_comp] = metrics
            # print(f"    complexity {comp} (overall {overall_comp}): ll={metrics['ll']:.4f}")
        k_results[k] = k_result

    # --- f2 linear baselines ---
    print("=" * 60)
    print("Computing f2 linear baselines...")
    f2_linear_results = {}
    for n_feat_str, version in version_dict['f2_linear'].items():
        n_feat = int(n_feat_str)
        comp = f2_linear_complexity(n_feat, overall=True)
        print(f"  n_features={n_feat} (version={version}): complexity={comp}")
        args = SimpleNamespace(
            eval_type='nn', version=version, dataset='test',
            pysr_version=None, pysr_model_selection=None,
        )
        metrics = calculate_metrics(args)
        f2_linear_results[comp] = metrics
        # print(f"    ll={metrics['ll']:.4f}")

    # --- Pure SR ---
    print("=" * 60)
    print("Computing pure SR results...")
    pure_sr_version = version_dict['pure_sr_version']
    pure_sr_table = _get_pure_sr_results(pure_sr_version)
    pure_sr_complexities = sorted(pure_sr_table['complexity'].tolist())
    print(f"  version={pure_sr_version}: {len(pure_sr_complexities)} complexities")
    pure_sr_results = {}
    for comp in pure_sr_complexities:
        args = SimpleNamespace(
            eval_type='pure_sr', pysr_version=pure_sr_version,
            pysr_model_selection=str(comp), dataset='test',
            version=None,
        )
        metrics = calculate_metrics(args)
        # skip invalid equations (very high RMSE)
        if metrics['rmse'] > 2.0:
            continue
        pure_sr_results[comp] = metrics
        # print(f"    complexity {comp}: ll={metrics['ll']:.4f}")

    # --- Pure SR2 (no intermediate features) ---
    print("=" * 60)
    print("Computing pure SR2 results...")
    pure_sr2_version = version_dict['pure_sr2_version']
    pure_sr2_nn_version = 28114
    pure_sr2_table = _get_pysr_results(pure_sr2_version)
    pure_sr2_complexities = sorted(pure_sr2_table['complexity'].tolist())
    print(f"  version={pure_sr2_version}: {len(pure_sr2_complexities)} complexities")
    pure_sr2_results = {}
    for comp in pure_sr2_complexities:
        args = SimpleNamespace(
            eval_type='pysr', version=pure_sr2_nn_version,
            pysr_version=pure_sr2_version,
            pysr_model_selection=str(comp), dataset='test',
        )
        metrics = calculate_metrics(args)
        pure_sr2_results[comp] = metrics
        # print(f"    complexity {comp}: ll={metrics['ll']:.4f}")

    print("=" * 60)
    print("All results computed.")

    return {
        'k': k_results,
        'f2_linear': f2_linear_results,
        'pure_sr': pure_sr_results,
        'pure_sr2': pure_sr2_results,
    }


def _extract_metric(results_dict, metric='ll'):
    """Extract a single metric from a {complexity: metrics_dict} mapping."""
    return {comp: m[metric] for comp, m in results_dict.items()}


def _paretoize_max(x, y, replace=False):
    """Paretoize for a metric where higher is better (e.g., log likelihood)."""
    neg_y = [-yi for yi in y]
    x_p, neg_y_p = paretoize(x, neg_y, replace=replace)
    return x_p, [-yi for yi in neg_y_p]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='graphics/pareto_comparison.pdf', help='Path to save the combined figure')
    parser.add_argument('--version_json', type=str, default='official_versions.json', help='Path to the JSON file containing model versions')
    return parser.parse_args()


def get_results(version_dict):
    k_results = get_k_results(version_dict['k'], overall=True)
    f2_linear_results = get_f2_linear_results(version_dict['f2_linear'], overall=True)
    pure_sr_results = load_pickle(f'pickles/pure_sr_results_all_{version_dict["mse_pure_sr_version"]}.pkl')[SPLIT]
    # get rid of entries with rmse over 2.0, because they were probably invalid equations
    pure_sr_results = {k: v for k, v in pure_sr_results.items() if v < 2.0}
    pure_sr2_results = load_pickle(f'pickles/pysr_results_all_28114_{version_dict["mse_pure_sr2_version"]}.pkl')[SPLIT]

    return k_results, f2_linear_results, pure_sr_results, pure_sr2_results

def main():
    args = get_args()
    # if you want to recompute the results, uncomment this line
    # results = compute_results(version_json=args.version_json)
    # save_pickle(results, 'pickles/ablation_results.pkl')
    results = load_pickle('pickles/ablation_results.pkl')
    plot_combined_pareto(results['k'], results['f2_linear'], results['pure_sr'], results['pure_sr2'], path=args.path)


if __name__ == "__main__":
    main()
