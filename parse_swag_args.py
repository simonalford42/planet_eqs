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

    parser.add_argument('--no_log', action='store_true', default=False, help='disable wandb logging')
    parser.add_argument('--no_swag', action='store_true')
    parser.add_argument('--slurm_id', type=int, default=-1, help='slurm job id')
    parser.add_argument('--slurm_name', type=str, default='', help='slurm job name')
    parser.add_argument('--version', type=int, help='', default=1278)
    parser.add_argument('--seed', type=int, default=0, help='default=0')

    parser.add_argument('--total_steps', type=int, default=300000, help='default=300000')
    parser.add_argument('--hidden_dim', type=int, default=40, help='regress nn and feature nn hidden dim')
    parser.add_argument('--latent', type=int, default=20, help='number of features f1 outputs')

    parser.add_argument('--swa_steps', type=int, default=50000, help='default=50000')

    parser.add_argument('--pysr_model', type=str, default=None, help='PySR model to load and replace f1 with, e.g. sr_results/hall_of_fame_9723_0.pkl')
    parser.add_argument('--pysr_model_selection', type=str, default='best', help='best, accuracy, score, or ix')
    parser.add_argument('--sr_f1', action='store_true', default=False, help='do misc. stuff with f1 and SR')
    parser.add_argument('--f1_variant', type=str, default='linear',
                        choices=['zero', 'identity', 'pysr', 'pysr_frozen', 'random_features', 'linear', 'mean_cov', 'mlp', 'random', 'random_frozen', 'bimt', 'biolinear', 'products'])
    parser.add_argument('--l1_reg', type=str, choices=['inputs', 'weights', 'f2_weights', 'both_weights'], default=None)
    parser.add_argument('--l1_coeff', type=float, default=None)
    parser.add_argument('--cyborg_max_pysr_ix', default=None, type=int, help='indices up to and including the max index will be replaced with the pysr features')
    parser.add_argument('--loss_ablate', default='default', type=str, choices=['no_classification', 'no_normalize', 'default', 'no_normalize_no_classification'], help='ablate loss things')
    # remember that f2_depth = 1 is one hidden layer of (h, h) shape, plus the input and output dim layers.
    parser.add_argument('--f2_depth', type=int,  default=1, help='regress nn number of hidden layers')
    parser.add_argument('--zero_theta', type=int,  nargs='+', default=0, help='ix or ixs of sin/cos theta1-3 to zero, doing at 1-6')
    parser.add_argument('--batch_size', type=int, default=2000, help='swag batch size')

    parser.add_argument('--no_summary_sample', action='store_true', default=False, help='dont sample the summary stats')
    parser.add_argument('--init_special', action='store_true', default=False, help='init special linear layer')
    parser.add_argument('--no_std', action='store_true')
    parser.add_argument('--no_mean', action='store_true')
    # string of args that would be passed into load_model.load(-), example 'version=1278'
    parser.add_argument('--load_f1', type=str, default=None)

    parser.add_argument('--f2_variant', type=str, default='mlp', choices=['pysr', 'pysr_residual', 'ifthen', 'mlp', 'linear', 'pysr_frozen', 'bimt', 'ifthen2', 'new'])
    parser.add_argument('--f2_ablate', type=int, default=None) # ix to drop from f2 input
    parser.add_argument('--f2_dropout', type=float, default=None) # dropout p for f2 input

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--mean_var', action='store_true')
    parser.add_argument('--load', type=str, default=None, help='ckpt path to load, e.g. model.ckpt')

    parser.add_argument('--pysr_f2', type=str, default=None) # PySR model to load and replace f2 with, e.g. 'sr_resuls/hall_of_fame_f2_21101_0_1.pkl'
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--prune_f1_topk', type=int, default=None)
    parser.add_argument('--prune_f1_threshold', type=float, default=None)
    parser.add_argument('--no_bias', action='store_true')
    parser.add_argument('--n_predicates', default=10, type=int)
    parser.add_argument('--pruned_debug', default=None, type=str, choices=['1','2','3','4','5','6'])

    args = parser.parse_args()

    if args.f1_variant == 'identity':
        # just hard coding for the n features with the default arguments..
        args.latent = 41

    return args
