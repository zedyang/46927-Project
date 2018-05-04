import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from influence.emp_risk_optimizer import EmpiricalRiskOptimizer


class LinearRegression2Blocks(EmpiricalRiskOptimizer):
    """
    Only for debug purposes.
    """

    def __init__(self, **kwargs):
        super(LinearRegression2Blocks, self).__init__(**kwargs)
        self.p1 = None

    def get_all_params(self):
        # number of parameters.
        self.n_params = self.p
        # split X into 2 blocks, for debug purpose only
        self.p1 = int(self.p/2)

        # build the flat params container
        all_params = tf.get_variable(
            'flat_params', (self.n_params, 1),
            initializer=tf.zeros_initializer)

        # build the params dict
        self.all_params_dict[
            'beta_block1'] = all_params[0:self.p1, :]
        self.all_params_dict[
            'beta_block2'] = all_params[self.p1:self.n_params, :]

        return all_params

    def get_emp_risk(self):
        # params dict split the params into groups, for convenience
        b1 = self.all_params_dict['beta_block1']
        b2 = self.all_params_dict['beta_block2']

        # y_hat
        y_hat = tf.add(
            tf.matmul(self.X_input[:, 0:self.p1], b1, name='block1'),
            tf.matmul(self.X_input[:, self.p1:self.p], b2, name='block2'),
            name='y_hat'
        )

        # must return individual losses and total empirical risk
        losses = tf.pow(self.y_input - y_hat, 2, name='l2_loss')
        emp_risk = tf.reduce_mean(losses, name='emp_risk')
        return losses, emp_risk

    def predict(self, X_test):
        params = self.get_eval(items=['params'])
        beta1, beta2 = params['beta_block1'], params['beta_block2']
        return X_test.dot(np.vstack((beta1, beta2)))


class RegularizedRegression(EmpiricalRiskOptimizer):
    """
    Regularized regression
    """

    def __init__(self, **kwargs):
        super(RegularizedRegression, self).__init__(**kwargs)
        self.C = kwargs.pop('C')
        self.hyperparams = ['eta', 'C']

    def get_all_params(self):
        """

        :return:
        """
        # number of parameters.
        self.n_params = self.p

        # build the flat params container
        all_params = tf.get_variable(
            'flat_params', (self.n_params, 1),
            dtype=tf.float32,
            initializer=tf.zeros_initializer)

        # build the params dict
        self.all_params_dict[
            'beta'] = all_params

        return all_params

    def get_emp_risk(self):
        """

        :return:
        """
        beta = self.all_params_dict['beta']

        # y_hat
        y_hat = tf.matmul(self.X_input, beta, name='y_hat')

        # L2 regularization
        l2_penalty = self.C*tf.reduce_sum(
            beta**2/self.n, name='l2_penalty')

        losses = tf.add(
            tf.square(self.y_input - y_hat),
            l2_penalty, name='losses')

        emp_risk = tf.reduce_mean(losses, name='emp_risk')
        return losses, emp_risk

    def predict(self, X_test):
        params = self.get_eval(items=['params'])
        beta = params['beta']
        return X_test.dot(beta)
