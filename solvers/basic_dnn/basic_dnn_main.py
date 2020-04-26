import solvers.basic_dnn.training_data_generator
import solvers.basic_dnn.basic_dnn as dnn_solver
import solvers.basic_dnn.constants


def create_basic_dnn_model():
    samples = 1000
    solvers.basic_dnn.training_data_generator.generate_random_training_data(6, 5, samples)
    path_ = "{}{}_samples_{}.csv".format(solvers.basic_dnn.constants.DATA_DIR,
                                         solvers.basic_dnn.constants.TRAINING_DATA_BASIC_DNN, samples)
    train_dataset, test_dataset, train_labels, test_labels = dnn_solver.get_data(path_)
    model = dnn_solver.build_model(test_dataset)
    weights_princ = model.layers[1].get_weights()
    history = dnn_solver.train_model(model, train_dataset, train_labels)
    dnn_solver.plot_training_validation(history)
    test_model(model, test_dataset, test_labels)
    model.summary()
    # dnn_solver.save_model(model, constants.MODEL_BASIC_DNN, samples)


def load_basic_dnn_model():
    path_model = r"C:\Users\Denis\Desktop\SnakePython\data\basic_dnn\mode_basic_dnn_mse_7.12E-03_samples_10000"
    training_path = r"C:\Users\Denis\Desktop\SnakePython\data\basic_dnn\training_data_basic_dnn_samples_10000.csv"
    train_dataset, test_dataset, train_labels, test_labels = dnn_solver.get_data(training_path)
    model = dnn_solver.load_model(path_model)
    # model.summary()
    # test_model(model, test_dataset, test_labels)
    return model


def test_model(model, test_dataset, test_labels):
    dnn_solver.test_model(model, test_dataset, test_labels)


if __name__ == '__main__':
    # load_basic_dnn_model()
    create_basic_dnn_model()
