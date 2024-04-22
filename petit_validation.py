from spock_reg_model import load
from sklearn.preprocessing import StandardScaler
import torch
from petit20_survival_time import Tsurv
import numpy as np
from utils import assert_equal

LABELS = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

def tsurv_inputs(x):
    '''
    X: [41] tensor of inputs
    returns: tuple of inputs (nu12, nu23, masses)
    '''
    # semimajor axes at each time step
    # petit paper is the average over the 10k orbits
    ixs = {'a1': 8, 'a2': 17, 'a3': 26, 'm1': 35, 'm2': 36, 'm3': 37}
    a1, a2, a3 = x[ixs['a1']], x[ixs['a2']], x[ixs['a3']]
    nu12 = (a1 / a2) ** (3 / 2)
    nu23 = (a2 / a3) ** (3 / 2)
    masses = [x[ixs['m1']], x[ixs['m2']], x[ixs['m3']]]
    return (nu12, nu23, masses)


def tsurv(x):
    '''
    x: [B, T, 41] batch of inputs
    returns: [B, ] prediction of instability time for each inputs
    '''
    # we only need locations at T=0
    x = x[:, 0, :] # [B, 41]
    preds = [Tsurv(*tsurv_inputs(xi)) for xi in x]
    preds = np.array(preds)
    preds = np.nan_to_num(preds, posinf=1e9, neginf=1e9, nan=1e9)

    # also threshold at 1e4 and 1e9
    preds = np.clip(preds, 1e4, 1e9)

    preds = np.log10(preds)
    return torch.tensor(preds)


def tsurv_with_std(x):
    '''
    x: [B, T, 41]
    returns [B, 2] prediction of instability time for each inputs, and dummy std 0
    '''
    t = tsurv(x)
    return torch.stack([t, torch.zeros_like(t)], dim=-1)


def safe_log_erf(x):
    base_mask = x < -1
    value_giving_zero = torch.zeros_like(x, device=x.device)
    x_under = torch.where(base_mask, x, value_giving_zero)
    x_over = torch.where(~base_mask, x, value_giving_zero)

    f_under = lambda x: (
         0.485660082730562*x + 0.643278438654541*torch.exp(x) +
         0.00200084619923262*x**3 - 0.643250926022749 - 0.955350621183745*x**2
    )
    f_over = lambda x: torch.log(1.0+torch.erf(x))

    return f_under(x_under) + f_over(x_over)

def _lossfnc(testy, y):
    mu = testy[:, [0]]
    std = testy[:, [1]]

    var = std**2
    t_greater_9 = y >= 9

    regression_loss = -(y - mu)**2/(2*var)
    regression_loss += -torch.log(std)

    regression_loss += -safe_log_erf(
                (mu - 4)/(torch.sqrt(2*var))
            )

    classifier_loss = safe_log_erf(
                (mu - 9)/(torch.sqrt(2*var))
        )

    safe_regression_loss = torch.where(
            ~torch.isfinite(regression_loss),
            -torch.ones_like(regression_loss)*100,
            regression_loss)
    safe_classifier_loss = torch.where(
            ~torch.isfinite(classifier_loss),
            -torch.ones_like(classifier_loss)*100,
            classifier_loss)

    total_loss = (
        safe_regression_loss * (~t_greater_9) +
        safe_classifier_loss * ( t_greater_9)
    )

    return -total_loss.sum(1)

def tsurv_val_loss(batch):
    X, y = batch
    testy = tsurv_with_std(X)
    assert_equal(testy.shape, y.shape)
    loss = _lossfnc(testy, y).sum()
    return loss

def tsurv_rmse(batch):
    X, y = batch
    testy = tsurv(X)
    y = y[:, 0]
    assert_equal(testy.shape, y.shape)
    return (testy - y).pow(2).sum()


# to access the validation set and compute baseline
model = load(19698)
no_op_scaler = StandardScaler(with_mean=False, with_std=False)
model.make_dataloaders(ssX=no_op_scaler, train_ssX=True)
validation_set = model.val_dataloader()

# maybe some predictions are completely off.
# inspect closer.

val_loss =  0
model_val_loss = 0

rmse = 0
model_rmse = 0

y_pred_tsurv_list = []
for batch in validation_set:

    X, y = batch
    model_preds = model(X, noisy_val=False)
    rmse = ((model_preds[:, 0] - y[:, 0])**2).sum().item()
    model_rmse = model_rmse + rmse

    testy = tsurv_with_std(X)
    assert_equal(testy.shape, y.shape)
    loss = _lossfnc(testy, y).sum()

    val_loss = val_loss + loss

    rmse = rmse + tsurv_rmse(batch).item()

    y_pred_tsurv_list += [(y, model__preds, testy)

print('val loss: ', val_loss)
print('rmse: ', rmse)


