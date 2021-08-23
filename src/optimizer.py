import itertools
import tensorflow as tf

import numpy as np
import pandas as pd

from ann_model import TimePerCall

tf.get_logger().setLevel('ERROR')
pd.options.mode.chained_assignment = None  # To avoid ugly warngings


def optimize(model, nel, nsp, nat, ntrans, nlanth, nks, ngsmooth,
             nk, nbetas, nnodes, corespernode, arch=0):

    if type(model) == str:
        model = TimePerCall.load(model)

    tpns = np.array([1, 2, 4, 6, 8])
    ncores = nnodes*corespernode
    npools = np.array([2**x for x in np.arange(10)])
    npools = npools[npools <= nnodes]
    x_ = np.array([a for a in itertools.product(tpns, npools)])
    X = np.zeros((len(x_), 13+x_.shape[1]))
    X[:, 12] = x_[:, 0]
    X[:, 13] = x_[:, 1]
    X[:, :12] = np.array([nel, nel**3, nsp, nat, ntrans, nlanth, nks,
                          ngsmooth, nk, nbetas, ncores, nnodes])
    X[:, -1] = arch
    X = pd.DataFrame(X)
    X.columns = ['n_el', 'n_el^3', 'n_species', 'n_at', 'n_transition',
                 'n_lanthanid',
                 'n_ks', 'n_g_smooth', 'n_k', 'n_betas', 'n_cores', 'n_nodes',
                 'threads_per_node', 'n_pool', 'arch']
    Y = model.predict_normed(X)[:, 0]
    t = Y*(nel**3)*nk/ncores
    tmin = np.min(t)
    opt_tpn = x_[np.argmin(t), 0]
    opt_npool = x_[np.argmin(t), 1]

    return tmin, opt_tpn, opt_npool


def optimize_from_df(model, df):
    df['corespernode'] = df['n_cores']//df['n_nodes']
    cols = ['n_el', 'n_species', 'n_at', 'n_transition',
            'n_lanthanid', 'n_ks', 'n_g_smooth', 'n_k',
            'n_betas', 'n_nodes', 'corespernode', 'arch']
    if type(df) == pd.DataFrame:
        return optimize(model, *df[:, cols].values)
    elif type(df) == pd.Series:
        return optimize(model, *df[cols].values)
    elif type(df) == dict:
        return optimize(model, *df[cols])
