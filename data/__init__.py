from enum import Enum

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Datafile(Enum):

    # (Regression) Simulated 10-features regression data
    SimulatedLm10 = 'lm_10.csv'

    # (Regression) Predict forest fire area
    # http://archive.ics.uci.edu/ml/datasets/Forest+Fires
    ForestFire = 'forestfires.csv'

    # (Classification) Predict iris type
    # https://archive.ics.uci.edu/ml/datasets/Iris
    Iris = 'iris.csv'

    # (Classification) Predict whether or not a client
    # will respond to advertisement, homework data.
    Marketing = 'marketing.csv'

    # (Classification) Predict digit labels (1, 4)
    # http://yann.lecun.com/exdb/mnist/
    BinaryMNIST17 = 'MNIST_17.csv'
    FullMNIST = 'MNIST.csv'


def load_data(data_file, test_config=(0,), random_state=42):
    if data_file is Datafile.ForestFire:
        X, y = load_forest_fires()
    elif data_file is Datafile.Iris:
        X, y = load_iris()
    elif data_file is Datafile.Marketing:
        X, y = load_marketing()
    elif data_file is Datafile.BinaryMNIST17:
        X, y = load_mnist_17()
    elif data_file is Datafile.FullMNIST:
        X, y = load_full_mnist()
    else:
        raise ValueError("Data set does not exist.")
    if type(test_config) in [int, float]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_config, random_state=random_state)
    elif type(test_config) in [list, tuple, np.ndarray]:
        test_config = np.array(test_config)
        n, _ = X.shape
        train_idx = np.ones(n, dtype=bool)
        test_idx = test_config
        train_idx[test_idx] = False
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx, :], y[test_idx, :]
    else:
        raise TypeError("Type {} not allowed for test_config.".format(
            str(type(test_config))))
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    return X_train, X_test, y_train, y_test, test_config


def load_forest_fires():
    df = pd.read_csv('data/{}'.format(Datafile.ForestFire.value))
    n = len(df)
    df_x = df.iloc[:, :12]
    y = df.values[:, -1].reshape(n, 1)
    df_one_hot = pd.get_dummies(df_x)
    X = df_one_hot.values
    return X, y


def load_iris():
    df = pd.read_csv("data/{}".format(
        Datafile.Iris.value), header=None)
    df.columns = ['X0', 'X1', 'X2', 'X3', 'y']
    df = df[df['y'].isin(['Iris-virginica', 'Iris-versicolor'])]
    n = len(df)
    X = df.iloc[:, :4].values
    y = df.iloc[:, -1].values.reshape(n, 1)
    y[y == 'Iris-virginica'] = 0
    y[y == 'Iris-versicolor'] = 1
    return X, y


def load_marketing():
    df = pd.read_csv("data/{}".format(
        Datafile.Marketing.value))
    n = len(df)
    df_x = df.iloc[:, :8]
    y = df.values[:, -1].reshape(n, 1)
    y[y == 'no'] = 0
    y[y == 'yes'] = 1
    df_one_hot = pd.get_dummies(df_x)
    X = df_one_hot.values
    return X, y


def load_mnist_17():
    df = pd.read_csv("data/{}".format(
        Datafile.BinaryMNIST17.value))
    df_1 = df[df['label'] == 1]
    df_7 = df[df['label'] == 7]
    df_17 = pd.DataFrame(pd.concat((df_1, df_7)))
    n = len(df_17)
    df_17 = df_17.sample(
        frac=1, random_state=42).reset_index(drop=True)
    X = df_17.iloc[:, 1:].values
    y = df_17.iloc[:, 0].values.reshape(n, 1)
    y[y == 7] = 0
    return X, y


def load_full_mnist():
    df = pd.read_csv("data/{}".format(
        Datafile.FullMNIST.value))
    n = len(df)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values.reshape(n, 1)
    return X, y
