import pandas as pd
from sklearn.model_selection import train_test_split


def import_dataset_banknote_authentication():
    dataset = pd.read_csv("data/data_banknote_authentication.txt", header=None)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test


def import_dataset_htru():
    dataset = pd.read_csv("data/HTRU_2.csv", header=None)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test

def import_dataset_sonar():
    dataset = pd.read_csv("data/sonar.all-data", header=None)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    y = (y == 'R')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test

def import_dataset_ionosphere():
    dataset = pd.read_csv("data/ionosphere.data", header=None)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    y = (y == 'g')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test


