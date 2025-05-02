#!/usr/bin/env python
# coding: utf-8
import argparse
import numpy as np
import spock_reg_model
from tqdm.notebook import tqdm
import petit
import pickle
from pure_sr_evaluation import pure_sr_predict_fn, get_pure_sr_results
from interpret import get_pysr_results
from utils import assert_equal

from sklearn.preprocessing import StandardScaler
from spock_reg_model import get_data


###############################################################################
# HOW `get_data` SPLITS & SERVES THE DATA
#
# ┌───────────────────────────┬──────────────────────────────────────────────┐
# │ argument combination      │ what it loads / how it splits               │
# ├───────────────────────────┼──────────────────────────────────────────────┤
# │ train=True, plot=False    │                                           │
# │ (default)                 │ • load *resonant_dataset.pkl*               │
# │                           │ • two nested train_test_split calls         │
# │                           │      1) 90 % / 10 % → hold-out → *finalX*   │
# │                           │      2) 90 % of 90 % → 81 % train           │
# │                           │                        9 % validation       │
# │                           │ • returns:                                  │
# │                           │     dataloader  → 81 % “train”              │
# │                           │     test_loader →  9 % “val”                │
# ├───────────────────────────┼──────────────────────────────────────────────┤
# │ train=True, plot=True     │ same as above **but** validation loader is  │
# │                           │ swapped with the 10 % *finalX* hold-out →   │
# │                           │ returned as “test”.                         │
# ├───────────────────────────┼──────────────────────────────────────────────┤
# │ train=False, plot_random=True│ • load *random_dataset.pkl*              │
# │                              │ • no splitting; whole file becomes       │
# │                              │   first returned loader (“random test”). │
# │                              │ • second loader is empty.                │
# └───────────────────────────┴──────────────────────────────────────────────┘
#
# Anything with train_all=True loads the combined resonant+random set and
# treats the “random” flag vector `r` to mask out subsets when plotting.
###############################################################################

def get_dataloader(
    split: str = "val",
    petit=False,
):
    """
    Parameters
    ----------
    split : {'train', 'val', 'test', 'random'}
        Which loader to obtain (see table above).
    petit : bool
        If True, loads data for use by Petit

    Returns
    -------
    torch.utils.data.DataLoader
    """
    model = spock_reg_model.load(24880)

    if petit:
        kwargs = {
            # no op scalar
            'ssX' :  StandardScaler(with_mean=False, with_std=False),
            # the no op scalar will not do anything even if we train it,
            # but we need to train it so we can use it
            'train_ssX' : True,
        }
    else:
        kwargs = {}

    if split == "train":
        model.make_dataloaders(**kwargs)
        return model.train_dataloader()
    if split == "val":
        model.make_dataloaders(**kwargs)
        return model.val_dataloader()
    if split == "test":
        model.make_dataloaders(train=True, plot=True, **kwargs)
        return model._val_dataloader
    if split == "random":
        if not petit:
            # need to load ssX for the resonant dataset
            model.make_dataloaders(train=True)
            from copy import deepcopy as copy
            assert model.ssX is not None
            tmp_ssX = copy(model.ssX)
            kwargs = {'ssX': tmp_ssX}

        model.make_dataloaders(train=False, plot_random=True, **kwargs)
        return model._dataloader

    raise ValueError("split must be one of 'train', 'val', 'test', or 'random'")


def nn_prediction_fn(model):
    model.eval()
    model.cuda()
    def predict_fn(x):
        pred = model(x.cuda(), noisy_val=False).cpu().detach().numpy()
        # just the mean
        return pred[..., 0]

    return predict_fn


def calculate_rmse(args, clip=True, within_range=True):
    petit = args.eval_type == 'petit'
    dataloader = get_dataloader(args.dataset, petit=petit)
    predict_fn = get_prediction_fn(args)

    truths = []
    preds = []
    for x, y in tqdm(dataloader):
        truths.append(y.numpy())
        preds.append(predict_fn(x))

    truths = np.concatenate(truths)
    preds = np.concatenate(preds)
    assert_equal(truths.shape[0], preds.shape[0])

    print('num nan:', np.sum(np.isnan(preds)), '/', len(preds))
    preds[np.isnan(preds)] = 4

    if clip:
        preds = np.clip(preds, 4, 9)

    truths = np.average(truths, 1)  # avg the two ground truths for each sim

    if within_range:
        preds = preds[truths < 9]
        truths = truths[truths < 9]

    rmse = np.average(np.square(truths - preds))**0.5
    return rmse


def const_predict_fn(const):
    return lambda *_: const


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved {filename}')


def calculate_all_results(args):
    results = {}
    for dataset in ['val', 'test', 'random']:
        args.dataset = dataset

        if args.eval_type in ['nn', 'petit']:
            rmse = calculate_rmse(args)
            results[dataset] = rmse
        else:
            if args.eval_type == 'pure_sr':
                sr_results = get_pure_sr_results(args.pysr_version)
            else:
                assert args.eval_type == 'pysr'
                sr_results = get_pysr_results(args.pysr_version)

            dataset_results = {}
            for c in sr_results['complexity']:
                args.pysr_model_selection = c
                rmse = calculate_rmse(args)
                dataset_results[c] = rmse

            results[dataset] = dataset_results

    results_tag = ''
    if args.version is not None:
        results_tag += f'_{args.version}'
    if args.pysr_version is not None:
        results_tag += f'_{args.pysr_version}'

    filename = f'pickles/{args.eval_type}_results_all{results_tag}.pkl'
    save_pickle(results, filename)


def calculate_k_results():
    d = {
        2:  {
            'version': 24880,
            'pysr_version': 11003,
            'val_loss': 1.62794,
        },
        3:  {
            'version': 74649,
            'pysr_version': 83278,
            'val_loss': 1.67736,
        },
        4:  {
            'version': 11566,
            'pysr_version': 51254,
            'val_loss': 1.63825,
        },
        5:  {
            'version': 72646,
            'pysr_version': 55894,
            'val_loss': 1.66181,
        }
    }

    overall_results = {}
    for k in d:
        version = d[k]['version']
        pysr_version = d[k]['pysr_version']
        reg = pickle.load(open(f'sr_results/{pysr_version}.pkl', 'rb'))
        results = reg.equations_[0]
        complexities = results['complexity']
        k_results = {}
        for c in complexities:
            args = get_args()
            args.version = version
            args.pysr_version = pysr_version
            args.pysr_model_selection = c
            args.just_rmse = True
            rmse = main(args)
            k_results[c] = rmse
            print(f'k={k}, c={c}, rmse={rmse}')
        overall_results[k] = k_results

    pickle.dump(overall_results, open('pickles/k_results_test.pkl', 'wb'))



def get_prediction_fn(args):
    if args.eval_type == 'petit':
        return petit.tsurv
    elif args.eval_type == 'pure_sr':
        pure_sr_results = get_pure_sr_results(args.pysr_version)
        return pure_sr_predict_fn(pure_sr_results, int(args.pysr_model_selection))
    elif args.eval_type == 'nn':
        model = spock_reg_model.load(version=args.version)
        return nn_prediction_fn(model)
    elif args.eval_type == 'pysr':
        model = spock_reg_model.load_with_pysr_f2(version=args.version, pysr_version=args.pysr_version, pysr_model_selection=args.pysr_model_selection)
        return nn_prediction_fn(model)
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=None)
    parser.add_argument('--pysr_version', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='val', choices=['train','val','test','random', 'all'])
    parser.add_argument('--eval_type', type=str, default='pysr', choices=['pure_sr', 'pysr', 'nn', 'petit'])
    parser.add_argument('--pysr_model_selection', type=str, default='accuracy', help='"best", "accuracy", "score", or an integer of the pysr equation complexity.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    calculate_all_results(args)
    # rmse = calculate_rmse(args)
    # print('RMSE:', rmse)
