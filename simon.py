
import pysr
pysr.julia_helpers.init_julia()

import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import pyplot as plt
import spock_reg_model
spock_reg_model.HACK_MODEL = True
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np
from scipy.stats import truncnorm
import sys
from parse_swag_args import parse
from einops import rearrange

def compute_features(inputs):
    model = PySRRegressor.from_file('results/hall_of_fame.pkl')
    assert inputs.shape[-1] == 31


def import_inputs_and_nn_features():
    args, checkpoint_filename = parse()
    seed = args.seed

    # Fixed hyperparams:
    lr = 5e-4
    TOTAL_STEPS = args.total_steps
    TRAIN_LEN = 78660
    batch_size = 2000 #ilog_rand(32, 3200)
    steps_per_epoch = int(1+TRAIN_LEN/batch_size)
    epochs = int(1+TOTAL_STEPS/steps_per_epoch)
    epochs, epochs//10
    args = {
        'seed': seed,
        'batch_size': batch_size,
        'hidden': args.hidden,#ilog_rand(50, 1000),
        'in': 1,
        'latent': args.latent, #2,#ilog_rand(4, 500),
        'lr': lr,
        'swa_lr': lr/2,
        'out': 1,
        'samp': 5,
        'swa_start': epochs//2,
        'weight_decay': 1e-14,
        'to_samp': 1,
        'epochs': epochs,
        'scheduler': True,
        'scheduler_choice': 'swa',
        'steps': TOTAL_STEPS,
        'beta_in': 1e-5,
        'beta_out': args.beta,#0.003,
        'act': 'softplus',
        'noisy_val': False,
        'gradient_clip': 0.1,
        # Much of these settings turn off other parameters tried:
        'fix_megno': args.megno, #avg,std of megno
        'fix_megno2': (not args.megno), #Throw out megno completely
        'include_angles': args.angles,
        'include_mmr': (not args.no_mmr),
        'include_nan': (not args.no_nan),
        'include_eplusminus': (not args.no_eplusminus),
        'power_transform': args.power_transform,
        'lower_std': args.lower_std,
        'train_all': args.train_all,
    }

    name = 'full_swag_pre_' + checkpoint_filename
    checkpoint_path = checkpoint_filename + '/version=0-v0.ckpt'
    model = spock_reg_model.VarModel.load_from_checkpoint(checkpoint_path)
    model.make_dataloaders()
    model.eval()

    # just takes a random batch of inputs and passes them through the neural network
    batch = next(iter(model.train_dataloader()))
    inputs, targets = model.generate_f1_inputs_and_targets(batch, batch_idx=0)
    print('inputs shape: ', inputs.shape)
    print('targets shape: ', targets.shape)

    N = 100
    X = rearrange(inputs, 'B T F -> (B T) F')
    y = rearrange(targets, 'B T F -> (B T) F')
    indices = np.random.choice(X.shape[0], size=N, replace=False)
    X, y = X[indices], y[indices]
    X, y = X.detach().numpy(), y.detach().numpy()


    all_labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

    skipped = ['nan_mmr_near', 'nan_mmr_far', 'nan_megno', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno']
    assert len(skipped) == 10

    included_ixs = [i for i in range(len(all_labels)) if all_labels[i] not in skipped]
    labels = [all_labels[i] for i in included_ixs]

    assert np.all(X[..., [i for i in range(len(all_labels)) if i not in included_ixs]] == 0)

    # X = X[..., included_ixs]
    # assert X.shape[-1] == 31

    return X, y


def run_regression(X, y):
    model = pysr.PySRRegressor(
        equation_file='results/hall_of_fame.csv',
        niterations=5,  # < Increase me for better results
        binary_operators=["+", "*", '/', '-', '^'],
        unary_operators=[
            "square",
            "cube",
            "exp",
            "log",
            'abs',
            'sqrt',
            'sin',
            'cos',
            'tan',
        ],
    )

    model.fit(X, y)
    torch_model = model.pytorch()

if __name__ == '__main__':
    X, y = import_inputs_and_nn_features()
    run_regression(X, y)
