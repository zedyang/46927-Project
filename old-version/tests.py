import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

from data import Datafile
from influence.emp_risk_optimizer import EmpiricalRiskOptimizer
from influence.plot_utils import compare_with_loo
from models.regularized_regression import LinearRegression2Blocks
from models.regularized_regression import RegularizedRegression
from models.hyperplane_clf import BinaryLogisticRegression
from models.hyperplane_clf import SmoothedSupportVector


def echo_headline(echo):
    print("###################################################")
    print(echo)
    print("###################################################")


def get_data(file_name):
    """

    :param file_name:
    :return:
    """
    df = pd.read_csv('data/{}'.format(file_name.value))
    n = len(df)
    if file_name == Datafile.SimulatedLm10:
        # simulated regression data
        X = df.values[:, 0:10]
        y = df.values[:, -1].reshape(n, 1)
    elif file_name == Datafile.ForestFire:
        df_x = df.iloc[:, :12]
        y = df.values[:, -1].reshape(n, 1)
        df_one_hot = pd.get_dummies(df_x)
        X = df_one_hot.values
    else:
        raise ValueError("Data set does not exist.")
    return X, y


def test_linear_regression(data_file, params=None):
    echo_headline(
        "Linear Regression Test on File <{}>".format(
            data_file.value)
    )
    X, y = get_data(data_file)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for data in [X_train, X_test, y_train, y_test]:
        print(data.shape)
    n, p = X_train.shape

    model = LinearRegression2Blocks(
        model_name='LinearRegression2Blocks',
        eta=1e-6
    )
    model.fit(
        X_train, y_train,
        n_iter=10000,
        verbose=1
    )

    I_loss_bf = model.influence_loss(
        X_test, y_test,
        method='brute-force'
    )

    I_loss_cg = model.influence_loss(
        X_test, y_test,
        method='cg',
        tol=1e-3,
        max_iter=500
    )

    loo_diff = model.leave_one_out_refit(
        X_test, y_test,
        n_iter=3000,
        force_restart=True
    )

    fig, axes = compare_with_loo(
        [I_loss_bf, I_loss_cg],
        loo_diff
    )
    plt.show()


if __name__ == '__main__':
    test_linear_regression(Datafile.ForestFire)
