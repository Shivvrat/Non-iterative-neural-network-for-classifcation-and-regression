import sys
import warnings

import activation_function
import comparision_regression
import elm_classifier
import elm_regressor
import import_data_classification
import import_data_regression
import rvfl_classifier
import comparision_classification
import rvfl_regressor

warnings.filterwarnings("ignore")


# Here we take the inputs given in the command line
arguments = list(sys.argv)
try:
    algorithm_name = str(arguments[1])
except:
    print("Please provide enough arguments")
    exit()

try:
    data_set_name = str(arguments[2])
    num_of_hidden_nodes = int(str(arguments[3]))

except:
    print "You are trying to run the regression algorithm or number of arguments are wrong please check "

try:
    threshold = 0.5
    threshold = int(str(arguments[4]))
except:
    print("You are trying to run the classification algorithm or number of arguments are wrong please check ")
def main():
    """
    This is the main function which is used to run all the algorithms
    :return:
    """
    try:
        if algorithm_name == '-elmc':
            if data_set_name == '-banknote_authentication':
                X_train, X_test, y_train, y_test = import_data_classification.import_dataset_banknote_authentication()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = elm_classifier.train_classifier(X_train, y_train, num_of_hidden_nodes, activation_function.tanh)
                predicted_output = elm_classifier.test_classifier(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                  activation_function.tanh, threshold)
                accuracy, precision, recall, f1_score = elm_classifier.evaluate_classifier(predicted_output, y_test)
                print(accuracy, precision, recall, f1_score)
            elif data_set_name == '-htru':
                X_train, X_test, y_train, y_test = import_data_classification.import_dataset_htru()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = elm_classifier.train_classifier(X_train, y_train, num_of_hidden_nodes, activation_function.tanh)
                predicted_output = elm_classifier.test_classifier(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                  activation_function.tanh, threshold)
                accuracy, precision, recall, f1_score = elm_classifier.evaluate_classifier(predicted_output, y_test)
                print(accuracy, precision, recall, f1_score)
            elif data_set_name == '-sonar':
                X_train, X_test, y_train, y_test = import_data_classification.import_dataset_sonar()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = elm_classifier.train_classifier(X_train, y_train, num_of_hidden_nodes, activation_function.tanh)
                predicted_output = elm_classifier.test_classifier(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                  activation_function.tanh, threshold)
                accuracy, precision, recall, f1_score = elm_classifier.evaluate_classifier(predicted_output, y_test)
                print(accuracy, precision, recall, f1_score)
            elif data_set_name == '-ionosphere':
                X_train, X_test, y_train, y_test = import_data_classification.import_dataset_ionosphere()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = elm_classifier.train_classifier(X_train, y_train, num_of_hidden_nodes, activation_function.tanh)
                predicted_output = elm_classifier.test_classifier(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                  activation_function.tanh, threshold)
                accuracy, precision, recall, f1_score = elm_classifier.evaluate_classifier(predicted_output, y_test)
                print(accuracy, precision, recall, f1_score)
        elif algorithm_name == '-rvflc':
            if data_set_name == '-banknote_authentication':
                X_train, X_test, y_train, y_test = import_data_classification.import_dataset_banknote_authentication()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = rvfl_classifier.train_classifier(X_train, y_train, num_of_hidden_nodes, activation_function.tanh)
                predicted_output = rvfl_classifier.test_classifier(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                  activation_function.tanh, threshold)
                accuracy, precision, recall, f1_score = rvfl_classifier.evaluate_classifier(predicted_output, y_test)
                print(accuracy, precision, recall, f1_score)
            elif data_set_name == '-htru':
                X_train, X_test, y_train, y_test = import_data_classification.import_dataset_htru()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = rvfl_classifier.train_classifier(X_train, y_train, num_of_hidden_nodes, activation_function.tanh)
                predicted_output = rvfl_classifier.test_classifier(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                  activation_function.tanh, threshold)
                accuracy, precision, recall, f1_score = rvfl_classifier.evaluate_classifier(predicted_output, y_test)
                print(accuracy, precision, recall, f1_score)
            elif data_set_name == '-sonar':
                X_train, X_test, y_train, y_test = import_data_classification.import_dataset_sonar()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = rvfl_classifier.train_classifier(X_train, y_train, num_of_hidden_nodes, activation_function.tanh)
                predicted_output = rvfl_classifier.test_classifier(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                  activation_function.tanh, threshold)
                accuracy, precision, recall, f1_score = rvfl_classifier.evaluate_classifier(predicted_output, y_test)
                print(accuracy, precision, recall, f1_score)
            elif data_set_name == '-ionosphere':
                X_train, X_test, y_train, y_test = import_data_classification.import_dataset_ionosphere()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = rvfl_classifier.train_classifier(X_train, y_train, num_of_hidden_nodes, activation_function.tanh)
                predicted_output = rvfl_classifier.test_classifier(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                  activation_function.tanh, threshold)
                accuracy, precision, recall, f1_score = rvfl_classifier.evaluate_classifier(predicted_output, y_test)
                print(accuracy, precision, recall, f1_score)
        elif algorithm_name == '-elmr':
            if data_set_name == '-wine':
                X_train, X_test, y_train, y_test = import_data_regression.import_dataset_winequality_white()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = elm_regressor.train_regressor(X_train, y_train, num_of_hidden_nodes, activation_function.relu)
                predicted_output = elm_regressor.test_regressor(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                activation_function.relu)
                mse, mae = elm_regressor.evaluate_regressor(predicted_output, y_test)
                print(mse, mae)
            elif data_set_name == '-airfoil':
                X_train, X_test, y_train, y_test = import_data_regression.import_dataset_airfoil_self_noise()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = elm_regressor.train_regressor(X_train, y_train, num_of_hidden_nodes, activation_function.relu)
                predicted_output = elm_regressor.test_regressor(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                activation_function.relu)
                mse, mae = elm_regressor.evaluate_regressor(predicted_output, y_test)
                print(mse, mae)
            elif data_set_name == '-abalone':
                X_train, X_test, y_train, y_test = import_data_regression.import_dataset_abalone()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = elm_regressor.train_regressor(X_train, y_train, num_of_hidden_nodes, activation_function.relu)
                predicted_output = elm_regressor.test_regressor(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                activation_function.relu)
                mse, mae = elm_regressor.evaluate_regressor(predicted_output, y_test)
                print(mse, mae)
        elif algorithm_name == '-rvflr':
            if data_set_name == '-wine':
                X_train, X_test, y_train, y_test = import_data_regression.import_dataset_winequality_white()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = rvfl_regressor.train_regressor(X_train, y_train, num_of_hidden_nodes, activation_function.relu)
                predicted_output = rvfl_regressor.test_regressor(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                activation_function.relu)
                mse, mae = rvfl_regressor.evaluate_regressor(predicted_output, y_test)
                print(mse, mae)
            elif data_set_name == '-airfoil':
                X_train, X_test, y_train, y_test = import_data_regression.import_dataset_airfoil_self_noise()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = rvfl_regressor.train_regressor(X_train, y_train, num_of_hidden_nodes, activation_function.relu)
                predicted_output = rvfl_regressor.test_regressor(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                activation_function.relu)
                mse, mae = rvfl_regressor.evaluate_regressor(predicted_output, y_test)
                print(mse, mae)
            elif data_set_name == '-abalone':
                X_train, X_test, y_train, y_test = import_data_regression.import_dataset_abalone()
                weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer = rvfl_regressor.train_regressor(X_train, y_train, num_of_hidden_nodes, activation_function.relu)
                predicted_output = rvfl_regressor.test_regressor(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                                                                activation_function.relu)
                mse, mae = rvfl_regressor.evaluate_regressor(predicted_output, y_test)
                print(mse, mae)
        elif algorithm_name == '-allc':
            X_train, X_test, y_train, y_test = import_data_classification.import_dataset_banknote_authentication()
            print(comparision_classification.classification_models(X_train, X_test, y_train, y_test))
        elif algorithm_name == '-allr':
            X_train, X_test, y_train, y_test = import_data_regression.import_dataset_winequality_white()
            evaluation_metrics2 = comparision_regression.regression_models(X_train, X_test, y_train, y_test)
            print(evaluation_metrics2)
        else:
            print("Please provide a valid algorithm name")
    except:
        print("The arguments were wrong")


if __name__ == "__main__":
    main()