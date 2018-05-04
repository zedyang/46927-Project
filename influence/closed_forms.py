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


def I_loss_RidgeCf(X_tr, Y_tr, X_te, Y_te, C):
    n_tr, p = X_tr.shape
    n_te, _ = X_te.shape
    beta_hat = np.linalg.inv(
        X_tr.T.dot(X_tr) + C*np.eye(p)).dot(X_tr.T).dot(Y_tr)
    H_inv = (n_tr/2)*np.linalg.inv(X_tr.T.dot(X_tr) + C*np.eye(p))
    I_loss = np.zeros((n_tr, n_te))
    for i in range(n_tr):
        for j in range(n_te):
            x_tr, y_tr = X_tr[i:i+1,:], Y_tr[i:i+1,:]
            x_te, y_te = X_te[j:j+1,:], Y_te[j:j+1,:]
            grad_tr = -2*x_tr.T.dot(
                y_tr - x_tr.dot(beta_hat)) + (2*C/n_tr)*beta_hat
            grad_te = -2*x_te.T.dot(
                y_te - x_te.dot(beta_hat)) + (2*C/n_tr)*beta_hat
            I_loss[i,j] = grad_tr.T.dot(H_inv).dot(grad_te)
    return I_loss


def LOO_diff_LinearRegression(X_tr, Y_tr, X_te, Y_te):
    l2_loss = lambda x, y, b: (y-x.dot(b))**2
    n_tr, p = X_tr.shape
    n_te, _ = X_te.shape
    beta_hat = np.linalg.inv(
        X_tr.T.dot(X_tr)).dot(X_tr.T).dot(Y_tr)
    L_full = l2_loss(X_te, Y_te, beta_hat)
    loss_diff = np.zeros((n_tr, n_te))
    for i in range(n_tr):
        rest_indices = [idx for idx in range(n_tr) if idx != i]
        X_loo, Y_loo = X_tr[rest_indices, :], Y_tr[rest_indices, :]
        beta_loo = np.linalg.inv(
            X_loo.T.dot(X_loo)).dot(X_loo.T).dot(Y_loo)
        L_loo = l2_loss(X_te, Y_te, beta_loo)
        loss_diff[i, :] = (L_full - L_loo).T
    return loss_diff