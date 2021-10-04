import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import json


def train_test_indices(df, split=0.8):
    # Create training and test set
    inds = np.random.choice(np.array(df.index), len(df), replace=False)
    ntr = int(len(df)*0.8)
    ind_tr = inds[:ntr]
    ind_val = inds[ntr:]
    return ind_tr, ind_val


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class TimePerCall():
    def __init__(self, activation='swish', l1=1e-5,
                 l2=1e-4, lr=0.00005, loss='mae', nvars=15):

        self.model = keras.Sequential([
            layers.Dense(30, activation=activation, input_shape=[nvars],
                         kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
            layers.Dropout(0.2),
            layers.Dense(20, activation=activation,
                         kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
            layers.Dense(15, activation=activation,
                         kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
            layers.Dense(10, activation=activation,
                         kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
            layers.Dense(1)
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.lr = lr
        self.model.compile(loss=loss,
                           optimizer=self.optimizer,
                           metrics=[loss])
        self.eps = 1e-12
        self.mx, self.my = None, None
        self.sx, self.sy = None, None

    def train(self, X, Y, epochs=25, validation_split=0.2):
        self.history = self.model.fit(
            X, Y,
            epochs=epochs, validation_split=validation_split, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

    def train_normed(self, x, y,
                     epochs=25, validation_split=0.2):
        df_normx = self.normalize_x(x)
        df_normy = self.normalize_y(y)
        self.train(df_normx, df_normy, epochs, validation_split)

    def predict_normed(self, x):
        Y_hat = self.predict(self.normalize_x(x))
        return self.denormalize_y(Y_hat)

    def normalize_x(self, x):
        logx = np.log(x + self.eps)
        if self.mx is None:
            self.mx = np.mean(logx)
        if self.sx is None:
            self.sx = np.std(logx)
        return (logx - self.mx)/self.sx

    def normalize_y(self, y):
        logy = np.log(y + self.eps)
        if self.my is None:
            self.my = np.mean(logy)
        if self.sy is None:
            self.sy = np.std(logy)
        return (logy - self.my)/self.sy

    def denormalize_x(self, x):
        normx = x*self.sx + self.mx
        return np.exp(normx) - self.eps

    def denormalize_y(self, y):
        normy = y*self.sy + self.my
        return np.exp(normy) - self.eps

    def save(self, filename):
        self.model.save(filename)
        params = {
            'eps': self.eps,
            'mx': self.mx.values,
            'my': self.my,
            'sx': self.sx.values,
            'sy': self.sy,
            'lr': self.lr,
            'model_name': filename}
        with open(filename + ".json", "w") as f:
            json.dump(params, f, cls=NumpyEncoder)

    @classmethod
    def load(cls, filename):
        model = TimePerCall()
        with open(filename + ".json") as f:
            params = json.load(f)
        model.model = keras.models.load_model(filename)
        model.eps = params['eps']
        model.mx = params['mx']
        model.sx = params['sx']
        model.my = params['my']
        model.sy = params['sy']
        model.lr = params['lr']
        return model
