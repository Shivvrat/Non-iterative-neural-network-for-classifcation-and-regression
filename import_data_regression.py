import pandas as pd
from sklearn.model_selection import train_test_split


def import_dataset_airfoil_self_noise():
    dataset = pd.read_csv("data/airfoil_self_noise.dat", sep=",", header=None, dtype=float)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test


def import_dataset_abalone():
    dataset = pd.read_csv("data/abalone.data", sep=",", header=None)
    X = dataset.iloc[:, 1:-1]
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test


def import_dataset_winequality_white():
    dataset = pd.read_csv("data/wine_quality.data", sep=";", dtype=float,header=None)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test