import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import solvers.training.basic_training_data_generator

print(tf.__version__)


def get_data(dataset_path):
    # Read data
    raw_dataset = pd.read_csv(dataset_path, names=solvers.training.basic_training_data_generator.LABELS,
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
    import tensorflow_docs as tfdocs
    import tensorflow_docs.modeling
    epochs = 1000
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        train_dataset, train_labels,
        epochs=epochs, validation_split=0.2, verbose=0,
        callbacks=[early_stop, tfdocs.modeling.EpochDots()])
    return history


def plot_training_validation(history):
    import tensorflow_docs as tfdocs
    import tensorflow_docs.plots
    hist_full = pd.DataFrame(history.history)
    hist_full['epoch'] = history.epoch
    hist = hist_full.tail()
    print("\nHistory\n")
    print(hist)

    # MAE output
    plotter = tfdocs.plots.HistoryPlotter()
    plotter.plot({'Basic': history}, metric="mae")
    plt.ylabel('MAE [MPG]')
    plt.ylim([0, max(max(hist_full.mae), max(hist_full.val_mae)) * 1.2])
    plt.show()

    # MSE output
    plotter.plot({'Basic': history}, metric="mse")
    plt.ylim([0, max(max(hist_full.mse), max(hist_full.val_mse)) * 1.2])
    plt.ylabel('MSE [MPG^2]')
    plt.show()


def save_model(model, name: str, samples: int):
    mse = model.history.history["val_mse"][-1]
    mse = '{:.2E}'.format(mse)
    path = "{}{}_mse_{}_samples_{}".format(solvers.training.basic_training_data_generator.DATA_DIR.replace("/", "\\"),
                                           name, mse, samples)
    model.save(path)


def test_model(model, test_dataset, test_labels):
    print("Testing the game...")
    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
    print("Testing set MAE: {:.2E} MSE: {:.2E}".format(mae, mse))

    test_predictions = model.predict(test_dataset).flatten()
    plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [-1.2, 1.2]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()
