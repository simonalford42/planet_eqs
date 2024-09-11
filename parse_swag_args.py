import argparse

def get_checkpoint_filename_old(args, glob=False):
    checkpoint_filename = (
            "results/steps=%d_megno=%d_angles=%d_power=%d_hidden=%d_latent=%d_nommr=%d" %
            (args.total_steps, args.megno, args.angles, args.power_transform, args.hidden, args.latent, args.no_mmr)
        + '_nonan=1_noeplusminus=1_v' + str(args.version) + '_'
    )
    if not glob:
        checkpoint_filename += '%d' %(args.seed,)

    return checkpoint_filename


def parse():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    # when importing from jupyter nb, it passes an arg to --f which we should just ignore
    parser.add_argument('--f', type=str, default=None, help='scapegoat for jupyter notebook')

    ########## logging, etc ##########
    parser.add_argument('--no_log', action='store_true', default=False, help='disable wandb logging')
    parser.add_argument('--no_plot', action='store_true', default=False, help='disable plotting after training')
    parser.add_argument('--version', type=int, help='', default=1278)
    parser.add_argument('--seed', type=int, default=0, help='default=0')

    ########## basic run args ##########
    parser.add_argument('--total_steps', type=int, default=300000, help='default=300000')
    parser.add_argument('--hidden_dim', type=int, default=40, help='regress nn and feature nn hidden dim')
    parser.add_argument('--latent', type=int, default=10, help='number of features f1 outputs')
    parser.add_argument('--swa_steps', type=int, default=50000, help='default=50000')
    # remember that f2_depth = 1 is one hidden layer of (h, h) shape, plus the input and output dim layers.
    parser.add_argument('--batch_size', type=int, default=2000, help='swag batch size')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval', action='store_true', help='disables optimizer step so no weights are changed')

    ########## quick experimenting options, many not used anymore ##########
    parser.add_argument('--sr_f1', action='store_true', default=False, help='do misc. stuff with f1 and SR')
    parser.add_argument('--loss_ablate', default='default', type=str, choices=['no_classification', 'no_normalize', 'default', 'no_normalize_no_classification'], help='ablate loss things')
    parser.add_argument('--zero_theta', type=int,  nargs='+', default=0, help='ix or ixs of sin/cos theta1-3 to zero, doing at 1-6')
    parser.add_argument('--no_summary_sample', action='store_true', default=False, help='dont sample the summary stats')
    parser.add_argument('--init_special', action='store_true', default=False, help='init special linear layer')
    parser.add_argument('--no_std', action='store_true')
    parser.add_argument('--no_mean', action='store_true')
    parser.add_argument('--f2_ablate', type=int, default=None) # ix to drop from f2 input
    parser.add_argument('--f2_dropout', type=float, default=None) # dropout p for f2 input
    parser.add_argument('--mean_var', action='store_true')
    parser.add_argument('--tsurv', action='store_true')
    parser.add_argument('--n_predicates', default=10, type=int, help='number predictates for if then f2')
    parser.add_argument('--fix_variance', action='store_true', help='fix the variance prediction to be one')
    parser.add_argument('--K', type=int, default=30, help='run swag K choice')
    parser.add_argument('--calc_scores', action='store_true')
    parser.add_argument('--petit', action='store_true')
    # example: 24880_feature_nn_simplified.pt
    parser.add_argument('--load_f1_feature_nn', default=None, type=str)
    parser.add_argument('--deterministic_summary_stats', action='store_true', help='deterministic summary stats')
    parser.add_argument('--nn_pred_std', action='store_true')
    parser.add_argument('--disable_grad', action='store_true')

    ########## architecture variant args ##########
    parser.add_argument('--f1_variant', type=str, default='linear',
                        choices=['zero', 'identity', 'random_features', 'linear', 'mean_cov', 'mlp', 'random', 'bimt', 'biolinear', 'products', 'products2', 'products3']) # to do pysr, use --pysr_f1 arg
    parser.add_argument('--f2_variant', type=str, default='mlp', choices=['ifthen', 'mlp', 'linear', 'bimt', 'ifthen2', 'new']) # to do pysr, use --pysr_f2 arg

    parser.add_argument('--f2_depth', type=int,  default=1, help='regress nn number of hidden layers')

    parser.add_argument('--l1_reg', type=str, choices=['inputs', 'weights', 'f2_weights', 'both_weights'], default=None)
    parser.add_argument('--l1_coeff', type=float, default=None)

    parser.add_argument('--prune_f1_topk', type=int, default=None, help='number of input features per latent feature')
    parser.add_argument('--prune_f2_topk', type=int, default=None, help='number of features to keep when pruning f2 linear')

    parser.add_argument('--freeze_f1', action='store_true')
    parser.add_argument('--freeze_f2', action='store_true')

    parser.add_argument('--load_f1', type=int, default=None, help='version for loading f1')
    parser.add_argument('--load_f2', type=int, default=None, help='version for loading f2')
    parser.add_argument('--load_f1_f2', type=int, default=None, help='version for loading both f1 and f2')

    parser.add_argument('--pysr_f1', type=str, default=None, help='PySR model to load and replace f1 with, e.g. sr_results/hall_of_fame_9723_0.pkl')
    parser.add_argument('--pysr_f1_model_selection', type=str, default='accuracy', help='best, accuracy, score, or complexity')

    parser.add_argument('--pysr_f2', type=str, default=None) # PySR model to load and replace f2 with, e.g. 'sr_results/hall_of_fame_f2_21101_0_1.pkl'
    parser.add_argument('--pysr_f2_model_selection', type=str, default='best', help='"best", "accuracy", "score", or an integer of the "complexity"')

    parser.add_argument('--f2_residual', type=str, default=None, choices=[None, 'pysr', 'mlp'])
    parser.add_argument('--pysr_f2_residual', type=str, default=None)# PySR model to load for f2 residual , e.g. 'sr_resuls/hall_of_fame_f2_21101_0_1.pkl'
    parser.add_argument('--pysr_f2_residual_model_selection', type=str, default=None)# PySR model to load for f2 residual , e.g. 'sr_resuls/hall_of_fame_f2_21101_0_1.pkl'

    args = parser.parse_args()

    if args.f1_variant == 'identity':
        # just hard coding for the n features with the default arguments..
        args.latent = 41

    return args
