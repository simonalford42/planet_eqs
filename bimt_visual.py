import pickle
import matplotlib.pyplot as plt
import numpy as np
import textwrap

def plot_pareto(path, title: str):
    results = pickle.load(open(path, 'rb'))
    equations = results.equations_[0] #only the first feature, we can include all 20 features on one plot
    x = equations['complexity']
    y = equations['loss']
    eqs = equations['equation']

    # Sort data based on complexity
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    eqs_sorted = eqs[sorted_indices]

    # Calculate AUC
    auc = np.trapz(y_sorted, x_sorted)

    # Find the lowest loss and its corresponding equation
    min_loss_index = np.argmin(y)
    min_loss = y[min_loss_index]
    min_loss_eq = eqs[min_loss_index]

    # Wrap the equation text
    wrapped_eq = textwrap.fill(min_loss_eq, width=50)  # Adjust 'width' as needed

    # plot the pareto frontier
    plt.figure()  # create a new figure
    plt.scatter(x, y)
    plt.plot(x_sorted, y_sorted)  # connect the dots
    plt.xlabel('Complexity')
    plt.ylabel('Loss')
    plt.title(f'Pareto Frontier for {title} (AUC: {auc:.2f})')

    # Display the lowest loss and its equation
    plt.text(0.95, 0.95, f'Lowest Loss: {min_loss:.2f}\nEquation: {wrapped_eq}', 
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

    # save the plot
    plt.savefig(f'pareto_{title}.png')

def main():
    # Paths to the PySR results files
    paths = [
        # ('/home/elg227/bnn_chaos_model/sr_results/hall_of_fame_19307_0.pkl', "BIMT_19307"),
        # ('/home/elg227/bnn_chaos_model/sr_results/hall_of_fame_29002_0.pkl', "BIMTD_29002"),
        # ('/home/elg227/bnn_chaos_model/sr_results/hall_of_fame_476_0.pkl', "BIMT_476"),
        # ('/home/elg227/bnn_chaos_model/sr_results/hall_of_fame_16760_0.pkl', "BIMT_16760"),
        ('/home/elg227/bnn_chaos_model/sr_results/hall_of_fame_21650_0.pkl', "Default_21650"),
        ('/home/elg227/bnn_chaos_model/sr_results/hall_of_fame_32286_0.pkl', "Default_32286")
    ]

    for path, title in paths:
        plot_pareto(path, title)

if __name__ == '__main__':
    main()

    """
    Runs:

    16760 (epoch 1948)
    476 (epoch 1948)
    19307 (epoch 1548)
    29002 (epoch 1548)

    27223 (f1:linear, f2:bimt)
    11891 (f1:linear, f2:bimt)
    
    2547731 (f1: linear, f2: pysr_residual (base - bimt)) (7501 epochs) --> 250 (2563532) val_loss_no_reg = 2.10

    2826851 (f1: linear, f2: mlp (40 hidden dim)) (1250 epochs) --> 17530 val_loss_no_reg = 2.12
    2547714 (f1: linear, f2: bimt (40 hidden dim)) (1250 epochs) --> 27548 (2563531) val_loss_no_reg = 2.12
    2826853 (f1: linear, f2: bimt (120 hidden dim)) (1250 epochs) --> 28437 val_loss_no_reg = 2.13

    2882295 (f1: linear, f2: mlp (20 hidden dim)) (1250 epochs) --> 
    
    3141997 (residual) --> 33796.pkl --> 3232141
    3141996 (direct) --> 51084.pkl --> 3232142
    """

