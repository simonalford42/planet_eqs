from modules import SpecialLinear
import torch
import torch.nn as nn
from load_model import load
from utils import assert_equal
import utils
from spock_reg_model import VarModel

model = load(version= 21101)

model.make_dataloaders()
model.eval()
batch = next(iter(model.train_dataloader()))
inputs, targets = model.generate_f1_inputs_and_targets(batch, batch_idx=0)

# i0 = inputs[0] # [100, 41]
# t0 = targets[0] # [100, 20]
# t0_means = t0.mean(dim=0) # [20]
# t0_vars = t0.std(dim=0) ** 2 # [20]
# covs = utils.batch_cov(i0.unsqueeze(0))[0] # shape: [41, 41]
# means = i0.mean(dim=0)
# inp = torch.cat((means, covs.flatten()))
# assert_equal(inp.shape, (41 + 41 * 41, ))

linear = model.feature_nn
assert_equal(type(linear), nn.Linear)
net = SpecialLinear.from_linear(linear)
# print(type(net))
# out = net(inp)
# if torch.allclose(out, torch.cat((t0_means, t0_vars))):
#     print('we are good')

# l0 = model.feature_nn.weight.data # shape: [41, 20]
# b0 = model.feature_nn.bias.data # shape: [20]
# i0 = inputs[0] # [100, 41]
# t0 = targets[0] # [100]
# t0.mean(dim=0) # 2.1679
# t0.std(dim=0) ** 2 # 0.0045
# assert torch.equal(t0, i0 @ l0.transpose(0,1) + b0) # True
# assert torch.allclose(t0.mean(dim=0), i0.mean(dim=0) @ l0.transpose(0, 1) + b0) # True
# covs = utils.batch_cov(i0.unsqueeze(0))[0] # shape: [41, 41]
# assert torch.allclose(t0.std(dim=0)**2, torch.einsum('ni, ij, nj -> n', l0, covs, l0)) # True

import parse_swag_args

parse_args, checkpoint_filename = parse_swag_args.parse()
seed = parse_args.seed

# Fixed hyperparams:
lr = 5e-4
TOTAL_STEPS = parse_args.total_steps
TRAIN_LEN = 78660
batch_size = 2000 #ilog_rand(32, 3200)
steps_per_epoch = int(1+TRAIN_LEN/batch_size)
epochs = int(1+TOTAL_STEPS/steps_per_epoch)
args = {
    'slurm_id': parse_args.slurm_id,
    'version': parse_args.version,
    'seed': seed,
    'batch_size': batch_size,
    'hidden': parse_args.hidden,#ilog_rand(50, 1000),
    'in': 1,
    'latent': parse_args.latent, #2,#ilog_rand(4, 500),
    'lr': lr,
    'swa_lr': lr/2,
    # 'out': 1,
    'out': parse_args.f2_depth,
    'samp': 5,
    'swa_start': epochs//2,
    'weight_decay': 1e-14,
    'to_samp': 1,
    'epochs': epochs,
    'scheduler': True,
    'scheduler_choice': 'swa',
    'steps': TOTAL_STEPS,
    'beta_in': 1e-5,
    'beta_out': parse_args.beta,#0.003,
    'act': 'softplus',
    'noisy_val': False,
    'gradient_clip': 0.1,
    # Much of these settings turn off other parameters tried:
    'fix_megno': parse_args.megno, #avg,std of megno
    'fix_megno2': (not parse_args.megno), #Throw out megno completely
    'include_angles': parse_args.angles,
    'include_mmr': (not parse_args.no_mmr),
    'include_nan': (not parse_args.no_nan),
    'include_eplusminus': (not parse_args.no_eplusminus),
    'power_transform': parse_args.power_transform,
}

# by default, parsed args get sent as hparams
for k, v in vars(parse_args).items():
    if k not in ['beta', 'megno', 'angles', 'no_mmr', 'no_nan', 'no_eplusminus']:
        args[k] = v

args['f1_variant'] = 'mean_cov'
args['no_summary_sample'] = True
model2 = VarModel(args)

args['f1_variant'] = 'identity'
id_model = VarModel(args)

batch = next(iter(model2.train_dataloader()))
x, y = batch

model2.training_step(batch, batch_idx=0)

# [B, 41 + 41*41] statistics of the inputs
input_stats = model2.compute_mean_cov_stats(x)
print(f'input_stats: {input_stats.shape}')

id_stats = id_model.compute_summary_stats2(x)

# [B, 40] statistics of the linear network
# linear_stats = model.compute_summary_stats2(x)
# print(f'linear_stats: {linear_stats.shape}')

# SpecialLinear initialized from model's linear feature_nn
# linear_approx_stats = net(input_stats)
# print(f'linear_approx_stats: {linear_approx_stats.shape}')

# misses = [i for i in range(linear_stats.shape[1]) if not torch.allclose(linear_stats[i], linear_approx_stats[i])]
# print(misses)

n = 41
input_means = input_stats[:, :n]
id_means = id_stats[:, :n]
assert torch.allclose(input_means, id_means)
input_covs = input_stats[:, n:]
id_vars = id_stats[:, n:]
from einops import rearrange
input_covs2 = rearrange(input_covs, 'b (n1 n2) -> b n1 n2', n1=n, n2=n)
input_vars = input_covs2[:, torch.arange(n), torch.arange(n)]
assert torch.allclose(input_vars, id_vars)
