import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import json
from matplotlib import pyplot as plt

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

        # Explicit Input layer
        inputs = keras.Input(shape=(nvars,))

        # Define the model architecture
        x = layers.Dense(30, activation=activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(20, activation=activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
        x = layers.Dense(15, activation=activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
        x = layers.Dense(10, activation=activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
        outputs = layers.Dense(1)(x)

        # Build the model
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.lr = lr
        self.model.compile(loss=loss,
                       optimizer=self.optimizer,
                       metrics=[loss])
        self.eps = 1e-12
        self.mx, self.my = None, None
        self.sx, self.sy = None, None


    def train(self, X, Y, epochs=50, validation_split=0.2, plot=1, save_path="training_plot.png"):
        train_history = self.history = self.model.fit(
            X, Y,
            epochs=epochs, validation_split=validation_split, batch_size=20, verbose=1)
        total_training_samples = len(X)
        steps_per_epoch = self.history.params['steps']
        calculated_batch_size = total_training_samples // steps_per_epoch
        print(f"Calculated Batch Size: {calculated_batch_size}")
        if plot == 1:
            # Plot training and validation loss
            plt.figure(figsize=(8, 6))
            plt.plot(np.arange(epochs), train_history.history['loss'], label='Train Loss')
            plt.plot(np.arange(epochs), train_history.history['val_loss'], label='Validation Loss')
            plt.legend(fontsize=12)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Loss', fontsize=14)
            plt.title('Training vs Validation Loss', fontsize=16)
            plt.grid(True)
        
            # Save the plot
            plt.savefig(save_path)
            print(f"Plot saved as {save_path}")
            plt.show()

    def predict(self, X):
        return self.model.predict(X)

    def train_normed(self, x, y,
                     epochs=50, validation_split=0.2, plot=1, save_path="training_plot.png"):
        df_normx = self.normalize_x(x)
        df_normy = self.normalize_y(y)
        self.train(df_normx, df_normy, epochs, validation_split, plot=plot, save_path=save_path)

    def predict_normed(self, x):
        Y_hat = self.predict(self.normalize_x(x))
        return self.denormalize_y(Y_hat)
        
    #In orther to evaluate the losses! without the effects of L1/L2 and dropout
    def evaluate_losses(self, X_train, Y_train, X_val, Y_val):
    	"""
    	Compute average loss on train and validation sets in eval mode.
    	This avoids dropout/regularization noise and last-batch effects.
    	"""
    	train_loss, train_metric = self.model.evaluate(X_train, Y_train, verbose=0)
    	val_loss, val_metric = self.model.evaluate(X_val, Y_val, verbose=0)
    	print(f"Train Loss (eval mode): {train_loss:.4f}")
    	print(f"Val Loss   (eval mode): {val_loss:.4f}")
    	return train_loss, val_loss


    def normalize_x(self, x):
        logx = np.log(x + self.eps)
        if self.mx is None:
            self.mx = np.mean(logx)
        if self.sx is None:
            self.sx = np.std(logx) + self.eps
        return (logx - self.mx)/self.sx

    def normalize_y(self, y):
        logy = np.log(y + self.eps)
        if self.my is None:
            self.my = np.mean(logy)
        if self.sy is None:
            self.sy = np.std(logy) + self.eps
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
            'mx': float(self.mx),
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
