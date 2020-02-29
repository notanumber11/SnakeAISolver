import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print(tf.__version__)

import constants

column_names = ["up", "down", "left", "right", "up available", "down available", "left available", "right available",
                "angle to apple", "reward"]

# Read data
dataset_path = constants.DATA_DIR + "training_data.csv"
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?",
                          sep='\t', skipinitialspace=True)

dataset = raw_dataset.copy()
# dataset["angle to apple"] = dataset["angle to apple"].round(2)
result = dataset.tail()
print(result)
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_labels = train_dataset.pop("reward")
test_label = test_dataset.pop("reward")

# build model
def build_model():
    model = keras.Sequential([
        layers.Dense(125, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()


example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
print(example_result)

EPOCHS = 1000

history = model.fit(
    train_dataset, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()])

hist_full = pd.DataFrame(history.history)
hist_full['epoch'] = history.epoch
hist = hist_full.tail()
print("\nHistory\n")
print(hist)

# manual history
# hist_mae = pd.DataFrame(hist_full["mae"], index = history.epoch)
# lines = hist_mae.plot.line()
# plt.show()

# tutorial history
plotter = tfdocs.plots.HistoryPlotter()
plotter.plot({'Basic': history}, metric = "mae")
plt.ylabel('MAE [MPG]')
plt.ylim([0, 1])
plt.show()
