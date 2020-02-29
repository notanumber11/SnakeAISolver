import math

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from tensorflow import keras
from tensorflow.keras import layers

from solvers.basic_dnn import constants
from solvers.basic_dnn.constants import LABELS

print(tf.__version__)


def get_data(path):
    # Read data
    dataset_path = path
    raw_dataset = pd.read_csv(dataset_path, names=LABELS,
                              na_values="?", header=0,
                              sep='\t', skipinitialspace=True)

    dataset_ = raw_dataset.copy()
    train_dataset = dataset_.sample(frac=0.8, random_state=0)
    test_dataset = dataset_.drop(train_dataset.index)
    train_labels = train_dataset.pop("reward")
    test_labels = test_dataset.pop("reward")
    last5elements = train_dataset.tail()
    print(last5elements)
    return train_dataset, test_dataset, train_labels, test_labels


def build_model(train_dataset):
    model = keras.Sequential([
        layers.Dense(125, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(125, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    # Print useful info
    model.summary()
    return model


def train_model(model, train_dataset, train_labels):
    epochs = 1000
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        train_dataset, train_labels,
        epochs=epochs, validation_split=0.2, verbose=0,
        callbacks=[early_stop, tfdocs.modeling.EpochDots()])
    return history


def plot(history):
    hist_full = pd.DataFrame(history.history)
    hist_full['epoch'] = history.epoch
    hist = hist_full.tail()
    print("\nHistory\n")
    print(hist)

    # MAE output
    plotter = tfdocs.plots.HistoryPlotter()
    plotter.plot({'Basic': history}, metric="mae")
    plt.ylabel('MAE [MPG]')
    plt.ylim([0, max(max(hist_full.mae), max(hist_full.val_mae))*1.2])
    plt.show()

    # MSE output
    plotter.plot({'Basic': history}, metric="mse")
    plt.ylim([0, max(max(hist_full.mse), max(hist_full.val_mse))*1.2])
    plt.ylabel('MSE [MPG^2]')
    plt.show()


def save_model(model, name: str, samples: int):
    mse = model.history.history["val_mse"][-1]
    mse = '{:.2E}'.format(mse)
    path = "{}{}_mse_{}_samples_{}".format(constants.DATA_DIR.replace("/", "\\"), name, mse, samples)
    model.save(path)
