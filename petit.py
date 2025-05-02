from spock_reg_model import load
from sklearn.preprocessing import StandardScaler
import torch
from petit20_survival_time import Tsurv
import numpy as np
from utils import assert_equal
import modules

LABELS = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

def tsurv_inputs(x):
    '''
    X: [T, 41] tensor of inputs
    returns: tuple of inputs (nu12, nu23, masses)
    '''
    # semimajor axes at each time step
    # petit paper is the average over the 10k orbits
    ixs = {'a1': 8, 'a2': 17, 'a3': 26, 'm1': 35, 'm2': 36, 'm3': 37}
    a1, a2, a3 = x[:, ixs['a1']].mean(), x[:, ixs['a2']].mean(), x[:, ixs['a3']].mean()
    # alternative: in multiswag_5_planet.py, a1/a2/a3 are not averaged, but just the initial value.
    # a1, a2, a3 = x[0, ixs['a1']], x[0, ixs['a2']], x[0, ixs['a3']]
    # alternative gets the same rmse when running main_figures.py --petit ...
    nu12 = (a1 / a2) ** (3 / 2)
    nu23 = (a2 / a3) ** (3 / 2)
    masses = [x[0, ixs['m1']], x[0, ixs['m2']], x[0, ixs['m3']]]
    return (nu12, nu23, masses)


'''
nu12 = np.array((data['avg_a1']/data['avg_a2'])**(3./2))
nu23 = np.array((data['avg_a2']/data['avg_a3'])**(3./2))
m1 = np.array(data['m1'])
m2 = np.array(data['m2'])
m3 = np.array(data['m3'])
petit_result = np.array([Tsurv(nu12[i], nu23[i], [m1[i], m2[i], m3[i]]) for i in range(len(m1))])
data['petit'] = petit_result
data['petit'] = np.nan_to_num(data['petit'], posinf=1e9, neginf=1e9, nan=1e9)
'''
def tsurv(x):
    '''
    x: [B, T, 41] batch of inputs
    returns: [B, ] prediction of instability time for each inputs
    '''
    assert_equal(len(x.shape), 3)
    assert_equal(x.shape[-1], 41)
    x = x[:, :] # [B, 41]
    preds = [Tsurv(*tsurv_inputs(xi)) for xi in x]
    preds = np.array(preds)
    preds = np.nan_to_num(preds, posinf=1e12, neginf=1e12, nan=1e12)

    # also threshold at 1e4 and 1e12
    preds = np.clip(preds, 1e4, 1e12)

    preds = np.log10(preds)
    return torch.tensor(preds)


def petit_prediction_fn():
    return tsurv

# copied from spock_reg_model.py
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

# copied from spock_reg_model.py
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


def petit_dataloader(validation=False):
    # to access the validation set and compute baseline
    model = load(12646)
    no_op_scaler = StandardScaler(with_mean=False, with_std=False)
    # the no op scalar will not do anything even if we train it,
    # but we need to train it so we can use it
    model.make_dataloaders(ssX=no_op_scaler, train_ssX=True)
    if validation:
        return model.val_dataloader()
    else:
        return model.train_dataloader()


def calc_stuff():
    tsurv_validation_set = petit_dataloader(validation=True)
    # remake the dataloaders with data of correct scaling for NN
    model = load(22040)
    model.make_dataloaders()
    eq_model = load(24880)
    eq_model.regress_nn = modules.PySRNet('sr_results/11003.pkl', model_selection=30)
    model = eq_model
    model_validation_set = model.val_dataloader()


    # maybe some predictions are completely off.
    # inspect closer.

    val_loss =  0
    model_val_loss = 0

    rmse = 0
    model_rmse = 0
    const9_rmse = 0
    const9_val_loss = 0

    # store a list of (x, y_actual, y_model_pred, y_tsurv_pred)
    x_store, y_store, y_model_store, y_tsurv_store = [], [], [], []
    N = 0
    for tsurv_batch, model_batch in zip(tsurv_validation_set, model_validation_set):
        N += len(model_batch[0])
        model_X, model_y = model_batch
        tsurv_X, tsurv_y = tsurv_batch
        torch.testing.assert_close(model_y, tsurv_y)

        model_preds = model(model_X, noisy_val=False)
        rmse = (model_preds[:, 0] - model_y[:, 0]).pow(2).sum().item()
        model_rmse = model_rmse + rmse
        model_val_loss += _lossfnc(model_preds, model_y).sum()

        # tsurv_pred = tsurv(tsurv_X)
        # # add dummt std 0
        # testy = torch.stack([tsurv_pred, model_preds[:, 1].clone()], dim=-1)
        # assert_equal(testy.shape, tsurv_y.shape)
        # loss = _lossfnc(testy, tsurv_y).sum()

        # val_loss = val_loss + loss

        # rmse = rmse + (tsurv_pred - tsurv_y[:, 0]).pow(2).sum().item()
        const9_rmse = const9_rmse + (9 - tsurv_y[:, 0]).pow(2).sum().item()
        const9_val_loss = const9_val_loss + _lossfnc(torch.stack([torch.full_like(tsurv_y[:, 0], 9), model_preds[:, 1].clone()], dim=-1), tsurv_y).sum()

        for i in range(len(model_X)):
            x_store.append(model_X[i])
            y_store.append(model_y[i])
            y_model_store.append(model_preds[i])
            # y_tsurv_store.append(tsurv_pred[i])

    x_store = torch.stack(x_store)
    y_store = torch.stack(y_store)
    y_model_store = torch.stack(y_model_store)
    # y_tsurv_store = torch.stack(y_tsurv_store)

    # torch.save(x_store, 'x.pt')
    # torch.save(y_store, 'y.pt')
    # torch.save(y_model_store, 'y_model.pt')
    # torch.save(y_tsurv_store, 'y_tsurv.pt')

    print('val loss: ', val_loss / N)
    print('model voss: ', model_val_loss / N)
    print('rmse: ', rmse / N)
    print('model rmse: ', model_rmse / N)
    print('const9 rmse: ', const9_rmse / N)
    print('const9 val loss: ', const9_val_loss / N)



