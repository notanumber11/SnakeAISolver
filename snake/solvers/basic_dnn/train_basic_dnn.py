import solvers.basic_dnn.basic_dnn as basic_dnn
from solvers.training_data_generators.data_utils import load_model
from solvers.training_data_generators.regression.short_vision_regression import ShortVisionRegression, DATA_DIR, \
    TRAINING_DATA_BASIC_DNN

short_vision_regression = ShortVisionRegression()


def create_basic_dnn_model():
    samples = 10000
    short_vision_regression.generate_random_training_data(6, 5, samples)
    path_ = "{}{}_samples_{}.csv".format(DATA_DIR,
                                         TRAINING_DATA_BASIC_DNN,
                                         samples)
    train_dataset, test_dataset, train_labels, test_labels = basic_dnn.get_data(path_)
    model = basic_dnn.build_model(test_dataset)
    history = basic_dnn.train_model(model, train_dataset, train_labels)
    basic_dnn.plot_training_validation(history)
    basic_dnn.test_model(model, test_dataset, test_labels)
    model.summary()
    basic_dnn.save_model(model, "model_basic_dnn", samples)


def load_basic_dnn_model():
    path_model = r"models/basic_dnn/model_basic_dnn_mse_8.20E-03_samples_10000"
    model = load_model(path_model)
    return model


def test_model(model):
    training_path = r"models/basic_dnn/training_data_basic_dnn_samples_10000.csv"
    train_dataset, test_dataset, train_labels, test_labels = basic_dnn.get_data(training_path)
    basic_dnn.test_model(model, test_dataset, test_labels)


def train_basic_dnn():
    create_basic_dnn_model()
    # model = load_basic_dnn_model()
    # test_model(model)
