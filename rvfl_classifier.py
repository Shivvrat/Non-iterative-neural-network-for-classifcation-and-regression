import numpy as np
import pandas
import activation_function
import evaluation_metrics


def train_classifier(X_train, y_train, num_of_hidden_nodes, activation_function1):
    X_with_bias = np.column_stack([X_train, np.ones([X_train.shape[0], 1])])
    weights_between_input_and_hidden_layer = np.random.randn(np.shape(X_with_bias)[1], num_of_hidden_nodes)
    hidden_layer = activation_function1(X_with_bias.dot(weights_between_input_and_hidden_layer))
    hidden_layer = np.concatenate((X_with_bias, hidden_layer), axis=1)
    weights_between_output_and_hidden_layer = np.linalg.pinv(hidden_layer).dot(y_train)
    return weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer


def test_classifier(X_test, weights_between_output_and_hidden_layer, weights_between_input_and_hidden_layer,
                    activation_function_for_hidden_layer, threshold):
    X_test_with_bias = np.column_stack([X_test, np.ones([X_test.shape[0], 1])])
    hidden_layer = activation_function_for_hidden_layer(X_test_with_bias.dot(weights_between_input_and_hidden_layer))
    hidden_layer = np.concatenate((X_test_with_bias, hidden_layer), axis=1)
    predicted_output = hidden_layer.dot(weights_between_output_and_hidden_layer)
    predicted_output = activation_function.tanh(predicted_output)
    predicted_output = pandas.array(predicted_output)
    predicted_output = predicted_output > threshold
    return predicted_output


def evaluate_classifier(predicted_output, y_test):
    y_test = y_test.array
    accuracy = evaluation_metrics.accuracy(y_test, predicted_output)
    precision = evaluation_metrics.precision(y_test, predicted_output)
    recall = evaluation_metrics.recall(y_test, predicted_output)
    f1_score = evaluation_metrics.f1_score(recall, precision)
    return accuracy, precision, recall, f1_score
