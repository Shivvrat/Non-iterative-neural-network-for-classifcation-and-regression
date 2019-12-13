import numpy as np
import pandas
from sklearn import metrics


def train_regressor(X_train, y_train, num_of_hidden_nodes, activation_function):
    X_with_bias = np.column_stack([X_train, np.ones([X_train.shape[0], 1])])
    weights_between_input_and_hidden_layer = np.random.randn(np.shape(X_with_bias)[1], num_of_hidden_nodes)
    hidden_layer = activation_function(X_with_bias.dot(weights_between_input_and_hidden_layer))
    weights_between_output_and_hidden_layer = np.linalg.pinv(hidden_layer).dot(y_train)
    return weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer


def test_regressor(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                   activation_function_for_hidden_layer):
    X_test_with_bias = np.column_stack([X_test, np.ones([X_test.shape[0], 1])])
    hidden_layer = activation_function_for_hidden_layer(X_test_with_bias.dot(weights_between_input_and_hidden_layer))
    predicted_output = hidden_layer.dot(weights_between_output_and_hidden_layer)
    predicted_output = pandas.array(predicted_output)
    return predicted_output


def evaluate_regressor(predicted_output, y_test):
    y_test = y_test.array
    mse = metrics.mean_squared_error(y_test, predicted_output)
    mae = metrics.mean_absolute_error(y_test, predicted_output)
    return mse, mae
