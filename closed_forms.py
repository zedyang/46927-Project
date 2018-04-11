import numpy as np


def I_loss_LinearRegressionCf(X_tr, Y_tr, X_te, Y_te):
    n_tr, p = X_tr.shape
    n_te, _ = X_te.shape
    beta_hat = np.linalg.inv(
        X_tr.T.dot(X_tr)).dot(X_tr.T).dot(Y_tr)
    H_inv = (n_tr/2)*np.linalg.inv((X_tr.T.dot(X_tr)))
    I_loss = np.zeros((n_tr, n_te))
    for i in range(n_tr):
        for j in range(n_te):
            x_tr, y_tr = X_tr[i:i+1, :], Y_tr[i:i+1, :]
            x_te, y_te = X_te[j:j+1, :], Y_te[j:j+1, :]
            grad_tr = np.multiply(
                -2, x_tr.T.dot(y_tr - x_tr.dot(beta_hat)))
            grad_te = np.multiply(
                -2, x_te.T.dot(y_te - x_te.dot(beta_hat)))
            I_loss[i,j] = grad_tr.T.dot(H_inv).dot(grad_te)
    return I_loss
