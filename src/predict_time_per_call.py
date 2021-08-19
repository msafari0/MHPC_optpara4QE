from ann_model import TimePerCall
import numpy as np
import itertools
import pandas as pd


def optimize(model, nel, nsp, nat, ntrans, nlanth, nks, ngsmooth,
             nk, nbetas, nnodes, corespernode, arch=0):

    if type(model) == str:
        model = TimePerCall.load(model)

    tpns = np.array([1, 2, 4, 6, 8])
    ncores = nnodes*corespernode
    npools = 1+np.arange(nnodes)
    x_ = np.array([a for a in itertools.product(tpns, npools)])
    x = np.array([nel, nel**3, nsp, nat, ntrans, nlanth, nks,
                  ngsmooth, nk, nbetas, ncores, nnodes, arch])
    X = np.empty((len(x_), len(x)+x_.shape[1]))
    X[:, 12:14] = x_
    X[:, :12] = x[:-1]
    X[:, :-1] = x[-1]

    X = pd.DataFrame(X)
    X.columns = ['n_el', 'n_el^3', 'n_species', 'n_at', 'n_transition',
                 'n_lanthanid',
                 'n_ks', 'n_g_smooth', 'n_k', 'n_betas', 'n_cores', 'n_nodes',
                 'threads_per_node', 'n_pool', 'arch']
    Y = model.predict_normed(X)
    t = Y*nel**3*nk/ncores
    tmin = np.min(t)
    opt_tpn = x_[np.argmin(t), 0]
    opt_npool = x_[np.argmin(t), 1]

    return tmin, opt_tpn, opt_npool
