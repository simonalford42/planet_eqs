import numpy as np
import spock_reg_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import re
import sympy as sp
import pickle
from pysr.export_latex import sympy2latextable, sympy2latex
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset']='dejavuserif'
import pandas as pd

def get_feature_nn(version):
    # load when on the cluster
    model = spock_reg_model.load(version)
    feature_nn = model.feature_nn

    input_linear = feature_nn.linear.weight * feature_nn.mask
    feature_nn.input_linear = input_linear.detach().numpy()
    if feature_nn.linear.bias is not None:
        feature_nn.input_bias = feature_nn.linear.bias.detach().numpy()
    else:
        feature_nn.input_bias = np.zeros(input_linear.shape[0])

    return feature_nn


def get_pysr_results(pysr_version, include_ssx=False, feature_nn=None):
    # load pysr f2 equations
    results = pickle.load(open(f'sr_results/{pysr_version}.pkl', 'rb'))
    results = results.equations_[0]

    # remove 'lambda_format' column
    results.drop(columns=['lambda_format'], inplace=True)

    if include_ssx:
        add_scalar_to_pysr_results(results, feature_nn)

    # rmse_values maps complexity to rmse
    rmse_values = get_k_rmse_values()[2]
    # add rmse column, matching complexity with existing 'complexity' column
    results['rmse'] = results['complexity'].map(rmse_values)

    return results

LABELS = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

# not all of these labels are actually used. for training, these inputs are zeroed out, but still passed in as zeroes.
SKIPPED = ['nan_mmr_near', 'nan_mmr_far', 'nan_megno', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno']


def convert_to_latex(label):
    # 1. if it ends in a number, add an underscore
    if label[-1].isdigit():
        label = label[:-1] + '_' + label[-1]
    # 2. replace sin/cos with \sin/\cos
    label = label.replace('sin', '\\sin')
    label = label.replace('cos', '\\cos')
    label = label.replace('_Omega', '\\Omega')
    label = label.replace('_pomega', '\\omega')
    label = label.replace('_theta', '\\theta')
    return label

LATEX_LABELS = [convert_to_latex(label) for label in LABELS]


def get_scalar():
    ssX = StandardScaler()
    ssX.scale_ = np.array([2.88976974e+03, 6.10019661e-02, 4.03849732e-02, 4.81638693e+01,
               6.72583662e-02, 4.17939679e-02, 8.15995339e+00, 2.26871589e+01,
               4.73612029e-03, 7.09223721e-02, 3.06455099e-02, 7.10726478e-01,
               7.03392022e-01, 7.07873597e-01, 7.06030923e-01, 7.04728204e-01,
               7.09420909e-01, 1.90740659e-01, 4.75502285e-02, 2.77188320e-02,
               7.08891412e-01, 7.05214134e-01, 7.09786887e-01, 7.04371833e-01,
               7.04371110e-01, 7.09828420e-01, 3.33589977e-01, 5.20857790e-02,
               2.84763136e-02, 7.02210626e-01, 7.11815232e-01, 7.10512240e-01,
               7.03646004e-01, 7.08017286e-01, 7.06162814e-01, 2.12569430e-05,
               2.35019125e-05, 2.04211110e-05, 7.51048890e-02, 3.94254400e-01,
               7.11351099e-02])
    ssX.mean_ = np.array([ 4.95458585e+03,  5.67411891e-02,  3.83176945e-02,  2.97223474e+00,
               6.29733979e-02,  3.50074471e-02,  6.72845676e-01,  9.92794768e+00,
               9.99628430e-01,  5.39591547e-02,  2.92795061e-02,  2.12480714e-03,
              -1.01500319e-02,  1.82667162e-02,  1.00813201e-02,  5.74404197e-03,
               6.86570242e-03,  1.25316320e+00,  4.76946516e-02,  2.71326280e-02,
               7.02054326e-03,  9.83378673e-03, -5.70616748e-03,  5.50782881e-03,
              -8.44213953e-04,  2.05958338e-03,  1.57866569e+00,  4.31476211e-02,
               2.73316392e-02,  1.05505555e-02,  1.03922250e-02,  7.36865006e-03,
              -6.00523246e-04,  6.53016990e-03, -1.72038113e-03,  1.24807860e-05,
               1.60314173e-05,  1.21732696e-05,  5.67292645e-03,  1.92488263e-01,
               5.08607199e-03])
    ssX.var_ = ssX.scale_**2
    return ssX


# m_i is the mean of the i'th feature, s_i is the standard deviation
# get the linear transformation that creates feature i
def linear_transformation(feature_nn, i):
    return feature_nn.input_linear[i]

# let's make the linear transformation a bit easier to read
def format_num(x, latex=False):
    if abs(x) > 1000:
        x2 = 100 * (x // 100)
        return str(x2)
    # if abs(x) > 10:
        # return f'{x:.0f}'
    # if abs(x) > 1:
        # return f'{x:.2f}'
    # if abs(x) > 0.1:
        # return f'{x:.2f}'
    # if abs(x) > 0.01:
        # return f'{x:.3f}'
    # elif abs(x) > 0.001:
        # return f'{x:.4f}'
    else:
        return f'{x:.3g}'

format_vec = np.vectorize(format_num)

sym_vars = {lbl: sp.Symbol(lbl, real=True) for lbl in LABELS}

ssX = get_scalar()

def simplify_scaled_feature(transformation, bias=0, include_ssx_bias=True):
    # Create symbolic variables for each feature

    expr = bias

    # Add each transformed feature (unscaled)
    for f_idx in range(len(LABELS)):
        c = transformation[f_idx]
        if c != 0:
            label = LABELS[f_idx]
            mean_j = ssX.mean_[f_idx] if include_ssx_bias else 0.0
            scale_j = ssX.scale_[f_idx]
            expr += c * (sym_vars[label] - mean_j) / scale_j

    expr = sp.simplify(expr)
    return expr

def format_sympy_expr(expr, latex=False):
    # replace labels with latex labels (change character from labels[i] to latex_labels[i])
    if latex:
        for lbl, sym in sym_vars.items():
            i = LABELS.index(lbl)
            new_lbl = LATEX_LABELS[i]
            expr = expr.subs(sym, sp.Symbol(new_lbl, real=True))

    coeffs = expr.as_coefficients_dict()

    terms_str = []
    const_str = None
    for var, coef in coeffs.items():
        if var == 1:
            const_str = format_num(coef, latex)
        else:
            times = '' if latex else '*'
            terms_str.append(f'{format_num(coef, latex)} {times} {var}')

    if const_str is not None:
        terms_str.append(const_str)

    return ' + '.join(terms_str)


def format_transformation(transformation, bias, latex):
    sorted_ixs = np.argsort(np.abs(transformation))[::-1]
    times = '' if latex else '*'
    used_labels = LATEX_LABELS if latex else LABELS
    features = [f'{format_num(transformation[i], latex)} {times} {used_labels[i]}' for i in sorted_ixs if transformation[i] != 0]
    if bias != 0:
        features = [format_num(bias, latex)] + features
    return ' + '.join(features)


def get_scaled_feature_bias(feature_nn, i):
    transformation = linear_transformation(feature_nn, i)
    bias = feature_nn.input_bias[i]
    expr = simplify_scaled_feature(transformation, bias)
    return expr.as_coefficients_dict().get(1, 0)


def add_bias_to_mean_terms(feature_nn, expr):
    replacements = {}
    for symbol in expr.free_symbols:
        if symbol.name.startswith('m') and symbol.name[1:].isdigit():
            i = symbol.name[1:]  # get the number after 'm'
            replacements[symbol] = symbol + get_scaled_feature_bias(feature_nn, int(i))
    return expr.xreplace(replacements)


def get_important_complexities(results, loss_gap = 0.25):
    complexities = list(results['complexity'])
    losses = list(results['loss'])
    assert sorted(losses) == losses[::-1]

    # important complexities are those that decrease the loss by more than loss_gap since the previous important complexity.
    important_complexities = [complexities[0]]
    current_loss = losses[0]

    for i in range(1, len(complexities)):
        if current_loss - losses[i] > loss_gap:
            important_complexities.append(complexities[i])
            current_loss = losses[i]

    # automatically include the highest complexity too
    if complexities[-1] != important_complexities[-1]:
        important_complexities.append(complexities[-1])

    return important_complexities


def remap_latex_str(s, mapping_dict):
    # use regex so that we don't replace multiple times. from o3-mini-high
    mapping_str = {str(old): str(new) for old, new in mapping_dict.items()}
    pattern = re.compile(r'([ms])_\{([^}]+)\}')
    def repl(match):
        prefix, key = match.groups()
        return f"{prefix}_{{{mapping_str[key]}}}" if key in mapping_str else match.group(0)
    return pattern.sub(repl, s)


def feature_string(feature_nn, i, include_ssx=False, latex=False, include_ssx_bias=True):
    transformation = linear_transformation(feature_nn, i)
    bias = feature_nn.input_bias[i]

    if include_ssx:
        expr = simplify_scaled_feature(transformation, bias, include_ssx_bias=include_ssx_bias)
        s = format_sympy_expr(expr, latex)
    else:
        s = format_transformation(transformation, bias, latex)

    # change + -'s to -'s
    s = s.replace(' + -', ' - ')
    return s


def feature_coeffs(feature_nn, i, include_ssx=False, latex=False, include_ssx_bias=True):
    transformation = linear_transformation(i)
    bias = feature_nn.input_bias[i]

    if include_ssx:
        expr = simplify_scaled_feature(transformation, bias, include_ssx_bias=include_ssx_bias)
        s = format_sympy_expr(expr, latex)
    else:
        s = format_transformation(transformation, bias, latex)

    # change + -'s to -'s
    s = s.replace(' + -', ' - ')
    return s


def add_scalar_to_pysr_results(results, feature_nn):
    # mutates
    def add_bias_to_mean_terms2(expr):
        return add_bias_to_mean_terms(feature_nn, expr)

    results['sympy_format'] = results['sympy_format'].apply(add_bias_to_mean_terms2)


def get_variables_in_str(e):
    return [e[i:i+2] for i in range(len(e) - 1) if e[i] in ['m', 's'] and e[i+1].isdigit()]


def get_mapping_dict(results, important_complexities):
    # make a "renaming map" of old_ix: new_ix where variables are mapped to
    # their order they appear in the equations of complexity in important_complexities
    # variables are m{i} or s{i}

    # find the variables present by extracting from string of all equations combined
    all_eqs = '\n'.join(results['equation'])

    # convert string of combined equations to list of variables
    def vars_in_str(e):
        return [int(e[i+1]) for i in range(len(e)-1) if e[i] in ['m', 's'] and e[i+1].isdigit()]

    all_vars = get_variables_in_str(all_eqs)
    all_vars = [int(s[1]) for s in all_vars]
    all_vars = list(dict.fromkeys(all_vars)) # remove duplicates, but keep order

    # get vars just in the important equations
    important_ixs = get_important_ixs(results, important_complexities)
    important_eqs = '\n'.join(results['equation'][important_ixs])
    important_vars = vars_in_str(important_eqs)

    # remove duplicates, but keep order
    vars = list(dict.fromkeys(important_vars))

    # put unused vars at the end (maintaining numerical order)
    for i in all_vars:
        if i not in vars:
            vars.append(i)

    return dict(zip(vars, range(len(vars))))


def get_important_ixs(results, important_complexities):
    return [i for i, c in enumerate(results['complexity']) if c in important_complexities]


def paretoize(x, y, replace=False):
    # if replace=False, just removes values not on pareto frontier
    # if replace=True, if (x, y) isn't on frontier, replaces y value with best y "so far"
    xy_sorted = sorted(zip(x, y), key=lambda t: t[0])
    transformed = []
    best_so_far = float('inf')

    for xx, yy in xy_sorted:
        if yy < best_so_far:
            transformed.append((xx, yy))
            best_so_far = yy

        if replace:
            transformed.append((xx, best_so_far))
        elif best_so_far == yy:
            transformed.append((xx, best_so_far))

    x_pareto, y_pareto = zip(*transformed)
    return list(x_pareto), list(y_pareto)


def get_k_rmse_values():
    k_results = pickle.load(open(f'pickles/k_results_test.pkl', 'rb'))
    # k_results maps k value to dict mapping complexity to rmse
    return k_results


def f1_latex_string(feature_nn, include_ssx=False, include_ssx_bias=True, mapping_dict=None):
    s = ('\\begin{align*}\n'
        + 'f_1& \\text{ features:} \\\\ \n')

    # if mapping_dict is provided, (1) only print variables in the dict, (2) remap the variable numbers
    if mapping_dict is None:
        mapping_dict = {i: i for i in range(feature_nn.input_linear.shape[0])}

    for i in mapping_dict:
        s += f'    &{mapping_dict[i]}: {feature_string(feature_nn, i, include_ssx, latex=True, include_ssx_bias=include_ssx_bias)} \\\\ \n'

    s += '''\end{align*}'''
    return s


def number_of_variables_in_expression(equation: str):
    # assumes each variable is m{i} or s{i}
    # so we can just count the number of s's and count the number of m's
    # checking for a number after so "sin" doesn't get counted
    # assume max feature dim from f1 is 100
    count = 0
    for i in range(100):
        if 'm' + str(i) in equation:
            count += 1
        if 's' + str(i) in equation:
            count += 1

    return count


def overall_complexity(entry: pd.Series, k: int):
    complexity = entry['complexity']
    # return complexity
    num_variables = number_of_variables_in_expression(entry.equation)
    return complexity + (3 * k - 2) * num_variables


def f1_latex_string2(feature_nn, include_ssx=False, include_ssx_bias=True, pysr_results=None, important_complexities=None):
    if pysr_results is not None:
        if important_complexities == None:
            equations = pysr_results['equation']
        else:
            important_ixs = get_important_ixs(pysr_results, important_complexities)
            equations = pysr_results['equation'][important_ixs]

        # only print variables used in an important equation in the pysr results
        all_vars = [get_variables_in_str(e) for e in equations]
        # go from list of lists to just one big list
        all_vars = [item for sublist in all_vars for item in sublist]
        all_vars = list(dict.fromkeys(all_vars)) # remove duplicates, but keep order
        mu_vars = [int(s[1]) for s in all_vars if s[0] == 'm']
        std_vars = [int(s[1]) for s in all_vars if s[0] == 's']
        mu_vars = sorted(mu_vars)
        std_vars = sorted(std_vars)
    else:
        mu_vars = range(feature_nn.input_linear.shape[0])
        std_vars = mu_vars


    s = ('\\begin{align*}\n'
        + 'f_1& \\text{ features:} \\\\ \n')

    # means
    for i in mu_vars:
        feature_str = feature_string(feature_nn, i, include_ssx, latex=True, include_ssx_bias=include_ssx_bias)
        s += '    &\mu_{' + str(i) + '} = \mathbb{E}\\left [' + feature_str + '\\right ] \\\\ \n'

    # stds
    for i in std_vars:
        feature_str = feature_string(feature_nn, i, include_ssx, latex=True, include_ssx_bias=include_ssx_bias)
        s += '    &\sigma_{' + str(i) + '} = \\text{Std} \\left (' + feature_str + '\\right ) \\\\ \n'

    s += '''\end{align*}'''
    return s


def f1_latex_strings(feature_nn, include_ssx=False, include_ssx_bias=True, pysr_results=None, important_complexities=None, std_biases=False):
    if pysr_results is not None:
        if important_complexities == None:
            equations = pysr_results['equation']
        else:
            important_ixs = get_important_ixs(pysr_results, important_complexities)
            equations = pysr_results['equation'][important_ixs]

        # only print variables used in an important equation in the pysr results
        all_vars = [get_variables_in_str(e) for e in equations]
        # go from list of lists to just one big list
        all_vars = [item for sublist in all_vars for item in sublist]
        all_vars = list(dict.fromkeys(all_vars)) # remove duplicates, but keep order
        mu_vars = [int(s[1]) for s in all_vars if s[0] == 'm']
        std_vars = [int(s[1]) for s in all_vars if s[0] == 's']
        mu_vars = sorted(mu_vars)
        std_vars = sorted(std_vars)
    else:
        mu_vars = range(feature_nn.input_linear.shape[0])
        std_vars = mu_vars

    strings = []

    # means
    for i in mu_vars:
        feature_str = feature_string(feature_nn, i, include_ssx, latex=True, include_ssx_bias=include_ssx_bias)
        s = '\mu_{' + str(i) + '} = { \\rm \mathbb{E}} \\left [' + feature_str + ' \\right ]'
        strings.append(s)

    # stds
    for i in std_vars:
        feature_str = feature_string(feature_nn, i, include_ssx, latex=True, include_ssx_bias=include_ssx_bias and std_biases)
        s = r'\sigma_{' + str(i) + r'} = {\rm Std}\left(' + feature_str + r'\right)'
        strings.append(s)

    return strings


def f2_latex_str(results, important_complexities=None, mapping_dict=None, add_rmse=True):
    if important_complexities is None:
        important_complexities = results['complexity'].values

    important_ixs = get_important_ixs(results, important_complexities)

    s = sympy2latextable(results, precision=2, columns=['equation', 'complexity'], indices=important_ixs)

    if mapping_dict:
        s = remap_latex_str(s, mapping_dict)

    s = s.replace('m_{', '\\mu_{')
    s = s.replace('s_{', '\\sigma_{')
    s = s.replace('y = ', r'\log_{10} T_{\text{inst}} = ')

    if add_rmse:
        # add column of rmse scores
        s = s.replace('cc@', 'ccc@')
        s = s.replace('Complexity \\\\', 'Complexity & RMSE \\\\')

        for i in range(len(results)):
            complexity = results.iloc[i]['complexity']
            rmse = results.iloc[i]['rmse']
            s = s.replace(f'${complexity}$ \\\\', f'${complexity}$ & ${rmse:.2f}$ \\\\')

    return s

def save_latex(latex_str, output_file, format='svg', font_family='serif'):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    text = ax.text(0.5, 0.5, latex_str, fontsize=20, ha='center', va='center', family=font_family)

    fig.canvas.draw()  # update text positions
    renderer = fig.canvas.get_renderer()
    bbox = text.get_window_extent(renderer=renderer).transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(output_file, format=format, bbox_inches=bbox, pad_inches=0, transparent=True, dpi=400)
    plt.close(fig)


def f2_latex_strings(results, important_complexities=None, mapping_dict=None):
    if important_complexities is None:
        important_complexities = results['complexity'].values

    important_ixs = get_important_ixs(results, important_complexities)

    latex_strs = [sympy2latex(expr, prec=2) for expr in results['sympy_format'][important_ixs]]
    latex_strs = ['\\log_{10} T_{\\mathrm{inst}} = ' + s for s in latex_strs]

    def transform(s):
        if mapping_dict:
            s = remap_latex_str(s, mapping_dict)

        s = s.replace('m_{', '\\mu_{')
        s = s.replace('s_{', '\\sigma_{')
        return s

    latex_strs = [transform(s) for s in latex_strs]
    return {c: s for c, s in zip(important_complexities, latex_strs)}




def plot_period_ratio_rmse():
    period_ratio_rmse = pickle.load(open('pickles/period_ratio_rmse.pkl', 'rb'))
    eqs = [int(s[2:]) for s in period_ratio_rmse.keys() if s[0:2] == 'eq']
    rmses = [period_ratio_rmse[f'eq{eq}'] for eq in eqs]
    plt.plot(eqs, rmses, 'o', label='equations')
    plt.xlabel('Complexity')
    plt.ylabel('Period ratio sweep RMSE')
    # add horizontal line at period_ratio_rmse['nn']
    plt.axhline(period_ratio_rmse['nn'], color='r', linestyle='--', label='nn')
    plt.legend()
    # add minor x axis tick marks at integers
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1))
    plt.show()
    return plt
