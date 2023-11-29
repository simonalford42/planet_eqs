import parse_swag_args
import spock_reg_model
import torch

ARGS, CHECKPOINT_FILENAME = parse_swag_args.parse_sr()

def load(version=1278, **kwargs):
    kwargs['version'] = version
    update_args(**kwargs)
    # Fixed hyperparams:
    name = 'full_swag_pre_' + CHECKPOINT_FILENAME
    checkpoint_path = CHECKPOINT_FILENAME + '/version=0-v0.ckpt'
    try:
        model = spock_reg_model.VarModel.load_from_checkpoint(checkpoint_path)
    except FileNotFoundError:
        checkpoint_path = CHECKPOINT_FILENAME + '/version=0.ckpt'
        model = spock_reg_model.VarModel.load_from_checkpoint(checkpoint_path)

    return model

def update_args(version=1278, seed=0, total_steps=300000, megno=False, angles=True, power_transform=False,
        hidden=40, latent=20, no_mmr=True, no_nan=True, no_eplusminus=True, train_all=False):
    extra = ''
    if no_nan:
        extra += '_nonan=1' 
    if no_eplusminus:
        extra += '_noeplusminus=1' 
    if train_all:
        extra += '_train_all=1' 
    checkpoint_filename = (
            "results/steps=%d_megno=%d_angles=%d_power=%d_hidden=%d_latent=%d_nommr=%d" %
            (total_steps, megno, angles, power_transform, hidden, latent, no_mmr)
        + extra + '_v' + str(version)
    )
    checkpoint_filename += '_%d' %(seed,)

    global ARGS, CHECKPOINT_FILENAME
    ARGS.version = version
    ARGS.seed = seed
    CHECKPOINT_FILENAME = checkpoint_filename

where_constant = torch.tensor([False,  True,  True,  True,  True,  True,  True,  True, False, False,
                               False, False, False, False, False, False, False, False, False, False,
                               False, False, False, False, False, False, False, False, False, False,
                               False, False, False, False, False, False, False, False, True,  True,
                               True])


def mask(model):
    return model.inputs_mask.mask.data

def const_avg(model):
    m = mask(model)
    return m[where_constant].mean()

def nonconst_avg(model):
    m = mask(model)
    return m[~where_constant].mean()
