from modules import SpecialLinear
import torch
import torch.nn as nn
from load_model import load
from utils import assert_equal
import utils

model = load(version= 21101)

model.make_dataloaders()
model.eval()
batch = next(iter(model.train_dataloader()))
inputs, targets = model.generate_f1_inputs_and_targets(batch, batch_idx=0)

i0 = inputs[0] # [100, 41]
t0 = targets[0] # [100, 20]
t0_means = t0.mean(dim=0) # [20]
t0_vars = t0.std(dim=0) ** 2 # [20]
covs = utils.batch_cov(i0.unsqueeze(0))[0] # shape: [41, 41]
means = i0.mean(dim=0)
inp = torch.cat((means, covs.flatten()))
assert_equal(inp.shape, (41 + 41 * 41, ))

linear = model.feature_nn
assert_equal(type(linear), nn.Linear)
nn = SpecialLinear.from_linear(linear)
