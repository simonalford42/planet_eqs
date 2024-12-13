from spock_reg_model import load, load_with_pysr_f2
import torch
import numpy as np
from utils import assert_equal


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


def val_loss(version=24880, pysr_version=None, model_selection='accuracy'):
    if pysr_version is not None:
        model = load_with_pysr_f2(version=version, seed=0, pysr_version=pysr_version, pysr_model_selection=model_selection)
    else:
        model = load(version=version)

    model.make_dataloaders()
    validation_set = model.val_dataloader()

    val_loss =  0
    total_rmse = 0

    N = 0
    for batch in validation_set:
        N += len(batch[0])
        model_X, model_y = batch

        model_preds = model(model_X, noisy_val=False)
        rmse = (model_preds[:, 0] - model_y[:, 0]).pow(2).sum().item()
        total_rmse = total_rmse + rmse
        val_loss += _lossfnc(model_preds, model_y).sum()

    print('val loss: ', val_loss / N)
    print('rmse: ', rmse / N)
