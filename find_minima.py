import argparse
import spock_reg_model
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np
import sys
import utils
import os

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
    # remember that f2_depth = 1 is one hidden layer of (h, h) shape, plus the input and output dim layers.
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
    parser.add_argument('--max_pred', type=float, default=12.0, help='max inst time NN can predict, default is 12')

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
    parser.add_argument('--combined_mass_feature', action='store_true')  # add M = m1 + m2 + m3 as input feature
    parser.add_argument('--predict_eq_uncertainty', action='store_true')  # load pysr net separately and predict uncertainty of that.
    parser.add_argument('--mse_loss', action='store_true')  # use mse loss for neural network

    args = parser.parse_args()

    if args.f1_variant == 'identity':
        # just hard coding for the n features with the default arguments..
        args.latent = 41

    return args

rand = lambda lo, hi: np.random.rand()*(hi-lo) + lo
irand = lambda lo, hi: int(np.random.rand()*(hi-lo) + lo)
log_rand = lambda lo, hi: 10**rand(np.log10(lo), np.log10(hi))
ilog_rand = lambda lo, hi: int(10**rand(np.log10(lo), np.log10(hi)))

parse_args = parse()
checkpoint_filename = utils.ckpt_path(parse_args.version, parse_args.seed)

# Fixed hyperparams:
TOTAL_STEPS = parse_args.total_steps
if TOTAL_STEPS == 0:
    import sys
    sys.exit(0)

TRAIN_LEN = 78660
batch_size = 2000 #ilog_rand(32, 3200)
steps_per_epoch = int(1+TRAIN_LEN/batch_size)
epochs = int(1+TOTAL_STEPS/steps_per_epoch)
if parse_args.eval:
    epochs = 1

command = utils.get_script_execution_command()
print(command)
print(f'Training for {epochs} epochs')

args = {
    'seed': parse_args.seed,
    'batch_size': batch_size,
    'f1_depth': 1,
    'swa_lr': parse_args.lr/2,
    'f2_depth': parse_args.f2_depth,
    'samp': 5,
    'swa_start': epochs//2,
    'weight_decay': 1e-14,
    'to_samp': 1,
    'epochs': epochs,
    'scheduler': True,
    'scheduler_choice': 'swa',
    'steps': TOTAL_STEPS,
    'beta_in': 1e-5,
    'beta_out': 0.001,
    'act': 'softplus',
    'noisy_val': False,
    'gradient_clip': 0.1,
    # Much of these settings turn off other parameters tried:
    'fix_megno': False, #avg,std of megno
    'fix_megno2': True, #Throw out megno completely
    'include_angles': True,
    'include_mmr': False,
    'include_nan': False,
    'include_eplusminus': False,
    'power_transform': False,
    # moving some parse args to here to clean up
    'plot': False,
    'plot_random': False,
    'train_all': False,
    'lower_std': False,
    'slurm_id': os.environ.get('SLURM_JOB_ID', None),
    'slurm_name': os.environ.get('SLURM_JOB_NAME', None),
}

# by default, parsed args get sent as hparams
for k, v in vars(parse_args).items():
    args[k] = v

name = 'full_swag_pre_' + checkpoint_filename
logger = WandbLogger(project='bnn-chaos-model', entity='bnn-chaos-model', name=name, mode='disabled' if args['no_log'] else 'online')
checkpointer = ModelCheckpoint(filepath=checkpoint_filename + '/{version}')

model = spock_reg_model.VarModel(args)
model.make_dataloaders()

max_l2_norm = args['gradient_clip']*sum(p.numel() for p in model.parameters() if p.requires_grad)
trainer = Trainer(
    gpus=1, num_nodes=1, max_epochs=args['epochs'],
    logger=logger if not args['no_log'] else False,
    checkpoint_callback=checkpointer, benchmark=True,
    terminate_on_nan=True, gradient_clip_val=max_l2_norm,
)

# do this early too so they show up while training, or if the run crashes before finishing
logger.log_hyperparams(params=args)

try:
    trainer.fit(model)
except ValueError as e:
    print('Error while training')
    print(e)

logger.log_hyperparams(params=model.hparams)
logger.log_metrics({'val_loss': checkpointer.best_model_score.item()})
# in case we load the model, we want to override the hparams from the model with the args actually passed in
logger.log_hyperparams(params=args)
logger.experiment.config['val_loss'] = checkpointer.best_model_score.item()
logger.save()
logger.finalize('success')
print(f'Finished training neural network with version {args["version"]}')
