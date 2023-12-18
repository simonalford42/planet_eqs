import spock_reg_model
import utils

def load(version, seed=0):
    path = utils.ckpt_path(version, seed)
    try:
        f = path + '/version=0-v0.ckpt'
        model = spock_reg_model.VarModel.load_from_checkpoint(f)
    except FileNotFoundError:
        f = path + '/version=0.ckpt'
        model = spock_reg_model.VarModel.load_from_checkpoint(f)

    return model

