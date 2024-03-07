"""This file trains a model to minima, then saves it for run_swag.py"""
import seaborn as sns
sns.set_style('darkgrid')
import spock_reg_model
spock_reg_model.HACK_MODEL = True
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np
from scipy.stats import truncnorm
import sys
from parse_swag_args import parse
import utils
import modules
from modules import mlp
import torch.nn as nn

rand = lambda lo, hi: np.random.rand()*(hi-lo) + lo
irand = lambda lo, hi: int(np.random.rand()*(hi-lo) + lo)
log_rand = lambda lo, hi: 10**rand(np.log10(lo), np.log10(hi))
ilog_rand = lambda lo, hi: int(10**rand(np.log10(lo), np.log10(hi)))

parse_args = parse()
checkpoint_filename = utils.ckpt_path(parse_args.version, parse_args.seed)

# Fixed hyperparams:
TOTAL_STEPS = parse_args.total_steps
TRAIN_LEN = 78660
batch_size = 2000 #ilog_rand(32, 3200)
steps_per_epoch = int(1+TRAIN_LEN/batch_size)
epochs = int(1+TOTAL_STEPS/steps_per_epoch)
print(f"epochs: {epochs}")

command = utils.get_script_execution_command()
print(command)

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
}

# by default, parsed args get sent as hparams
for k, v in vars(parse_args).items():
    args[k] = v

name = 'full_swag_pre_' + checkpoint_filename
# logger = TensorBoardLogger("tb_logs", name=name)
logger = WandbLogger(project='bnn-chaos-model', entity='bnn-chaos-model', name=name, mode='disabled' if args['no_log'] else 'online')
checkpointer = ModelCheckpoint(filepath=checkpoint_filename + '/{version}')
if args['load']:
    model = spock_reg_model.load(args['load'])
    # spock_reg_model.update_l1_model(model)

    if 'prune_f1_topk' in args and args['prune_f1_topk'] is not None and args['f1_variant'] != 'pruned_products':
        model.feature_nn = modules.pruned_linear(model.feature_nn, top_k=args['prune_f1_topk'])
        model.l1_reg_weights = args['l1_reg'] == 'weights'
    elif 'prune_f1_threshold' in args and args['prune_f1_threshold'] is not None:
        model.feature_nn = modules.pruned_linear(model.feature_nn, threshold=args['prune_f1_threshold'])
        model.l1_reg_weights = args['l1_reg'] == 'weights'

    if args['pysr_f2']:
        model.regress_nn = modules.PySRNet(args['pysr_f2'], args['pysr_model_selection'])
        if args['f1_variant'] == 'pysr_frozen':
            utils.freeze_module(model.regress_nn)

    if args['f2_variant'] == 'pysr_residual':
        pysr_net = modules.PySRNet(args['pysr_f2'], args['pysr_model_selection'])
        utils.freeze_module(pysr_net)
        base_net = mlp(args['latent'] * 2, 2, args['hidden_dim'], args['f2_depth'])
        model.regress_nn = modules.SumModule(pysr_net, base_net)
        model.l1_reg_f2_weights = args['l1_reg'] in ['f2_weights', 'both_weights']
    elif args['f2_variant'] == 'new':
        model.regress_nn = modules.mlp(model.regress_nn[0].in_features, 2, model.hparams['hidden_dim'], model.hparams['f2_depth'])
        model.l1_reg_f2_weights = args['l1_reg'] in ['f2_weights', 'both_weights']

    if args['eval']:
        model.disable_optimization()

    # trying to prevent nans from happening
    model.input_noise_logvar = torch.nn.Parameter(torch.zeros(model.input_noise_logvar.shape)-2)
    model.summary_noise_logvar = torch.nn.Parameter(torch.zeros(model.summary_noise_logvar.shape) - 2) # add to summaries, not direct latents


else:
    model = spock_reg_model.VarModel(args)
    model.make_dataloaders()

labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

max_l2_norm = args['gradient_clip']*sum(p.numel() for p in model.parameters() if p.requires_grad)
trainer = Trainer(
    gpus=1, num_nodes=1, max_epochs=args['epochs'],
    logger=logger if not args['no_log'] else False,
    checkpoint_callback=checkpointer, benchmark=True,
    terminate_on_nan=True, gradient_clip_val=max_l2_norm,
)

# torch.autograd.set_detect_anomaly(True)
# do this early too so they show up while training, or if the run crashes before finishing
logger.log_hyperparams(params=args)

try:
    trainer.fit(model)
except ValueError as e:
    print('Error while training')
    print(e)
    model.load_state_dict(torch.load(checkpointer.best_model_path)['state_dict'])

logger.log_hyperparams(params=model.hparams)
logger.log_metrics({'val_loss': checkpointer.best_model_score.item()})
# in case we load the model, we want to override the hparams from the model with the args actually passed in
logger.log_hyperparams(params=args)

logger.experiment.config['val_loss'] = checkpointer.best_model_score.item()

logger.save()
logger.finalize('success')

logger.save()

model.load_state_dict(torch.load(checkpointer.best_model_path)['state_dict'])
model.make_dataloaders()

# loading models with pt lightning sometimes doesnt work, so lets also save the feature_nn and regress_nn directly
if 'pysr' not in args['f1_variant']:
    torch.save(model.feature_nn, f'models/{args["version"]}_feature_nn.pt')
if 'pysr' not in args['f2_variant']:
    torch.save(model.regress_nn, f'models/{args["version"]}_regress_nn.pt')
if args['f2_variant'] == 'pysr_residual':
    torch.save(model.regress_nn.module1, f'models/{args["version"]}_pysr_nn.pt')
    torch.save(model.regress_nn.module2, f'models/{args["version"]}_base_nn.pt')

print('Finished running')
