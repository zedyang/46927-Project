import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

from data import Datafile, load_data
from influence.emp_risk_optimizer import EmpiricalRiskOptimizer
from influence.plot_utils import compare_with_loo
from influence.closed_forms import I_loss_RidgeCf
from models.regularized_regression import RegularizedRegression
from models.hyperplane_clf import BinaryLogisticRegression
from models.hyperplane_clf import SmoothedSupportVector


def echo_headline(echo):
    print("###################################################")
    print(echo)
    print("###################################################")


def test_linear_regression(data_file):
    # np.random.seed(0)
    echo_headline(
        "Ridge Regression Test on File <{}>".format(
            data_file.value)
    )

    X_train, X_test, y_train, y_test = load_data(
        data_file, test_config=6)
    n_tr, p = X_train.shape
    n_te, _ = X_test.shape

    init_eta = 1
    batch_size = 100
    C = 1.0
    train_iter = 50000
    loo_extra_iter = 5000
    decay_epochs = (5000, 8000)
    checkpoint_iter = train_iter-1
    # LOO a on random set of training indices, otherwise too slow
    leave_indices = np.random.choice(
        n_tr, size=100, replace=False)
    print("LOO Indices:", leave_indices)

    model = RegularizedRegression(
        model_name='RidgeRegression',
        init_eta=init_eta,
        decay_epochs=decay_epochs,
        batch_size=batch_size,
        C=C
    )

    model.fit(
        X_train, y_train,
        n_iter=train_iter,
        verbose=1,
        iter_to_switch_to_sgd=np.inf
    )

    I_loss_bf = model.influence_loss(
        X_test, y_test,
        method='brute-force'
    )

    I_loss_close_form = I_loss_RidgeCf(
        X_train, y_train, X_test, y_test, C)

    """
    I_loss_cg = model.influence_loss(
        X_test, y_test,
        method='cg',
        tol=1e-3,
        max_iter=500
    )
    """

    loo_diff = model.leave_one_out_refit(
        X_test, y_test,
        n_iter=loo_extra_iter,
        iter_to_load=checkpoint_iter,
        leave_indices=leave_indices
    )

    print(I_loss_bf[leave_indices, :]/n_tr)
    print(loo_diff)

    fig, axes = compare_with_loo(
        [I_loss_bf[leave_indices, :],
         I_loss_close_form[leave_indices, :]],
        loo_diff, n_samples=n_tr
    )

    a, b = loo_diff.shape
    for j in range(n_te):
        print("Test Point %d Correlation: %.4f" % (j, np.corrcoef(
            I_loss_bf[leave_indices, j], loo_diff[:, j])[0][1]))
    print("Overall Correlation: %.4f" % np.corrcoef(
            I_loss_bf[leave_indices, :].reshape(a*b,),
            loo_diff.reshape(a*b,))[0][1])

    plt.show()


def test_binary_logistic(data_file):
    echo_headline(
        "Binary Logistic Test on File <{}>".format(
            data_file.value)
    )
    X_train, X_test, y_train, y_test, test_indices = load_data(
        Datafile.SmallBinaryMNIST14, test_config=[45, 256])
    n_tr, p = X_train.shape
    n_te, _ = X_test.shape
    print(n_tr, p)

    init_eta = 1e-5
    batch_size = 200
    train_iter = 50000
    loo_extra_iter = 5000
    decay_epochs = (4000, 5000)
    checkpoint_iter = train_iter - 1
    # LOO a on random set of training indices, otherwise too slow
    leave_indices = np.random.choice(n_tr, size=20, replace=False)
    assert not set(test_indices) & set(leave_indices)
    print(test_indices)
    print(leave_indices)

    #tf.reset_default_graph()
    model = BinaryLogisticRegression(
        model_name='BinaryLogistic-Notebook',
        init_eta=init_eta,
        decay_epochs=decay_epochs,
        batch_size=batch_size,
    )

    model.fit(
        X_train, y_train,
        n_iter=train_iter,
        verbose=1,
        iter_to_switch_to_sgd=np.inf
    )

    I_loss_bf = model.influence_loss(
        X_test, y_test,
        method='brute-force'
    )

    loo_diff = model.leave_one_out_refit(
        X_test, y_test,
        n_iter=loo_extra_iter,
        iter_to_load=checkpoint_iter,
        leave_indices=leave_indices
    )

    fig, axes = compare_with_loo(
        [I_loss_bf[leave_indices, :],
         I_loss_bf[leave_indices, :]],
        loo_diff, n_samples=n_tr
    )

if __name__ == "__main__":
    test_linear_regression(Datafile.ForestFire)