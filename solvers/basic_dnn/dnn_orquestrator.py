from solvers.basic_dnn import training_data_generator, constants, dnn_solver

#


def create_model():
    samples = 1000
    training_data_generator.generate_training_data(6, samples)
    path_ = "{}{}_samples_{}.csv".format(constants.DATA_DIR, constants.TRAINING_DATA_BASIC_DNN, samples)
    train_dataset, test_dataset, train_labels, test_labels = dnn_solver.get_data(path_)
    model = dnn_solver.build_model(test_dataset)
    history = dnn_solver.train_model(model, train_dataset, train_labels)
    dnn_solver.plot(history)
    dnn_solver.save_model(model, constants.MODEL_BASIC_DNN, samples)


create_model()
