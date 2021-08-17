import numpy as np
eps = 1e-12


def train_test_split(df_, split=0.8, yname=None):
    # Create training and test set
    msk = np.random.rand(len(df_)) < split
    df_train = df_[msk]
    df_val = df_[~msk]

    X_tr = df_train.drop([yname], axis=1)
    Y_tr = df_train[yname]

    X_val = df_val.drop([yname], axis=1)
    Y_val = df_val[yname]

    return X_tr, Y_tr, X_val, Y_val


def delog(y):
    return np.exp(y) - eps


def log(y):
    return np.log(y + eps)


def norm(y, m=None, s=None):
    if m is None:
        m = np.mean(y)
    if s is None:
        s = np.std(y) + eps
    return (y-m)/s, m, s


def denorm(y, m, s):
    return y*(s - 1e-12)+m


def normlog(y, m=None, s=None):
    y_mod = log(y)
    norm_y, m, s = norm(y_mod, m, s)
    return norm_y, m, s


def denormlog(y, m, s):
    norm_y = denorm(y, m, s)
    norm_y = delog(norm_y)
    return norm_y
