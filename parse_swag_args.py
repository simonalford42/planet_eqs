import argparse
import random

def default_parser():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    # Required positional argument
    parser.add_argument('--no_log', action='store_true', default=False, help='disable wandb logging')
    parser.add_argument('--version', type=int, help='', default=1278)
    parser.add_argument('--total_steps', type=int, default=300000, help='default=300000')
    parser.add_argument('--swa_steps', type=int, default=50000, help='default=50000')
    parser.add_argument('--hidden', type=int, default=40, help='default=40')
    parser.add_argument('--seed', type=int, default=0, help='default=0')
    parser.add_argument('--latent', type=int, default=20, help='default=20')
    parser.add_argument('--beta', type=float, default=0.001, help='default=0.001')
    # changed the defaults so we dont have to pass in the cl args each time
    parser.add_argument('--angles', action='store_true', default=True, help='default=False')
    parser.add_argument('--megno', action='store_true', default=False, help='default=False')
    parser.add_argument('--no_mmr', action='store_true', default=True, help='default=False')
    parser.add_argument('--no_nan', action='store_true', default=True, help='default=False')
    parser.add_argument('--no_eplusminus', action='store_true', default=True, help='default=False')
    parser.add_argument('--power_transform', action='store_true', default=False, help='default=False')
    parser.add_argument('--plot', action='store_true', default=False, help='default=False')
    parser.add_argument('--plot_random', action='store_true', default=False, help='default=False')
    parser.add_argument('--train_all', action='store_true', default=False, help='default=False')
    parser.add_argument('--lower_std', action='store_true', default=False, help='default=False')
    parser.add_argument('--slurm_id', type=int, default=-1, help='slurm job id')
    parser.add_argument('--pysr_model', type=str, default=None, help='PySR model to load and replace f1 with, e.g. sr_results/hall_of_fame_9723_0.pkl')
    parser.add_argument('--pysr_model_selection', type=str, default='best', choices=['best', 'accuracy', 'score'], help='')
    parser.add_argument('--sr_f1', action='store_true', default=False, help='do misc. stuff with f1 and SR')
    parser.add_argument('--f1_variant', type=str, default='default',
                        choices=['zero', 'identity', 'pysr', 'pysr_frozen', 'random_features', 'linear', 'mean_cov', 'default'])
    parser.add_argument('--l1_reg', action='store_true', default=False)
    parser.add_argument('--l1_coeff', type=float, default=0.01)
    parser.add_argument('--cyborg_max_pysr_ix', default=None, type=int, help='indices up to and including the max index will be replaced with the pysr features')
    parser.add_argument('--loss_ablate', default='default', type=str, choices=['no_classification', 'no_normalize', 'default', 'no_normalize_no_classification'], help='ablate loss things')
    parser.add_argument('--no_summary_sample', action='store_true', default=False, help='dont sample the summary stats')
    parser.add_argument('--f2_depth', type=int,  default=1, help='regress nn number of hidden layers')
    parser.add_argument('--zero_theta', type=int,  nargs='+', default=0, help='ix or ixs of sin/cos theta1-3 to zero, doing at 1-6')
    parser.add_argument('--init_special', action='store_true', default=False, help='init special linear layer')
    return parser

def parse(glob=False):
    parser = default_parser()
    args = parser.parse_args()
    extra = ''
    if args.no_nan:
        extra += '_nonan=1' 
    if args.no_eplusminus:
        extra += '_noeplusminus=1' 
    if args.train_all:
        extra += '_train_all=1' 

    if args.f1_variant == 'identity':
        # just hard coding for the n features with the default arguments..
        args.latent = 41

    if args.pysr_model is not None and args.f1_variant == 'default':
        args.f1_variant = 'pysr'

    checkpoint_filename = (
            "results/steps=%d_megno=%d_angles=%d_power=%d_hidden=%d_latent=%d_nommr=%d" %
            (args.total_steps, args.megno, args.angles, args.power_transform, args.hidden, args.latent, args.no_mmr)
        + extra + '_v' + str(args.version)
    )
    if not glob:
        checkpoint_filename += '_%d' %(args.seed,)

    return args, checkpoint_filename

def parse_sr(glob=False):
    parser = default_parser()
    # for pysr only
    parser.add_argument('-t', '--time_in_hours', type=float, default=1)
    parser.add_argument('-m', '--max_size', type=int, default=60)

    args = parser.parse_args()
    extra = ''
    if args.no_nan:
        extra += '_nonan=1' 
    if args.no_eplusminus:
        extra += '_noeplusminus=1' 
    if args.train_all:
        extra += '_train_all=1' 

    if args.f1_variant == 'identity':
        # just hard coding for the n features with the default arguments..
        args.latent = 41
    assert (args.f1_variant in ['pysr', 'pysr_frozen']) == (args.pysr_model is not None)

    checkpoint_filename = (
            "results/steps=%d_megno=%d_angles=%d_power=%d_hidden=%d_latent=%d_nommr=%d" %
            (args.total_steps, args.megno, args.angles, args.power_transform, args.hidden, args.latent, args.no_mmr)
        + extra + '_v' + str(args.version)
    )
    if not glob:
        checkpoint_filename += '_%d' %(args.seed,)

    return args, checkpoint_filename
