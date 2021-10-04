import itertools
import joblib
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # To avoid ugly warngings


def optimize(model, nel, nk, nbetas, nnodes, cores_per_node):

    if type(model) == str:
        model = joblib.load(model)

    tpns = np.array([2, 4, 8])
    ncores = nnodes*cores_per_node
    nodes_per_pool = np.array([1, 2, 4, 8, 16])
    x_ = np.array([a for a in itertools.product(tpns, nodes_per_pool)])
    X = np.zeros((len(x_), 5+x_.shape[1]))
    X[:, 0] = x_[:, 0]
    X[:, -1] = x_[:, 1]
    X[:, 1] = nel
    X[:, 2] = nk
    X[:, 3] = nbetas
    X[:, 4] = ncores
    X[:, 5] = nnodes
    scores = model.predict(np.log(X))
    scores[0] = scores[0]
    opt_tpns = X[np.argmax(scores)][0]
    opt_npp = X[np.argmax(scores)][-1]
    return opt_tpns, opt_npp


def optimize_from_df(model, df):
    df['corespernode'] = df['n_cores']//df['n_nodes']
    cols = ['n_el', 'n_k',
            'n_betas', 'n_nodes', 'corespernode']
    if type(df) == pd.DataFrame:
        return optimize(model, *df[:, cols].values)
    elif type(df) == pd.Series:
        return optimize(model, *df[cols].values)
    elif type(df) == dict:
        return optimize(model, *df[cols])
