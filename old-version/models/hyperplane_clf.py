import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from influence.emp_risk_optimizer import EmpiricalRiskOptimizer


class BinaryLogisticRegression(EmpiricalRiskOptimizer):
    """
    BinaryLogisticRegression
    """

    def __init__(self, **kwargs):
        super(BinaryLogisticRegression, self).__init__(**kwargs)

    def get_all_params(self):
        """
        :return:
        """
        # number of parameters.
        self.n_params = self.p

        # build the flat params container
        all_params = tf.get_variable(
            'flat_params', (self.n_params, 1),
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

        # logits
        logits = tf.sigmoid(tf.matmul(self.X_input, beta))

        # loss
        losses = - self.y_input * tf.log(
            logits) - (1.0 - self.y_input) * tf.log(1.0 - logits)
        emp_risk = tf.reduce_mean(losses, name='emp_risk')
        return losses, emp_risk

    def predict(self, X_test):
        params = self.get_eval(items=['params'])
        beta = params['beta']
        logits = 1 / (1 + np.exp(-X_test.dot(beta)))
        return 1*(logits > 0.5)


class SmoothedSupportVector(EmpiricalRiskOptimizer):

    def __init__(self, **kwargs):
        super(SmoothedSupportVector, self).__init__(**kwargs)
        self.C = kwargs.pop('C')
        self.t = kwargs.pop('t')
        self.hyperparams = ['eta', 'C', 't']

    @staticmethod
    def smooth_hinge_loss(x, t):
        """

        :param x:
        :param t:
        :return:
        """
        exponents = (1-x)/t
        exponents_truncate = tf.maximum(
            exponents, tf.zeros_like(exponents))
        return tf.multiply(t, exponents_truncate + tf.log(
            tf.exp(exponents-exponents_truncate) +
            tf.exp(tf.zeros_like(exponents)-exponents_truncate)
        ), name='smooth_hinge_loss')

    def get_all_params(self):
        # number of parameters.
        self.n_params = self.p

        # build the flat params container
        all_params = tf.get_variable(
            'flat_params', (self.n_params, 1), dtype=tf.float32,
            initializer=tf.random_normal_initializer)

        # build the params dict
        self.all_params_dict[
            'beta'] = all_params[0:self.n_params, :]

        return all_params

    def get_emp_risk(self):
        # params dict split the params into groups, for convenience
        beta = self.all_params_dict['beta']

        # unsigned inner product
        logits = tf.matmul(self.X_input, beta, name='logits')
        margin = tf.multiply(
            tf.cast(self.y_input, tf.float32),
            logits, name='margin')

        # L2 regularization
        l2_penalty = self.C*tf.reduce_sum(
            beta**2/self.n, name='l2_penalty')

        # hinge loss
        """
        hinge_loss = tf.subtract(
            tf.constant(1.0),
            tf.multiply(
                self.y_input,
                x_dot_beta),
            name='hinge_loss')
        """

        # smooth hinge
        """
        smooth_hinge_loss = tf.multiply(
            tf.constant(self.t, dtype=tf.float64),
            tf.log(1+tf.exp(-margin/self.t)),
            name='smooth_hinge_loss'
        )
        """

        """
        t = tf.constant(self.t, name='t')
        smooth_hinge = tf.multiply(
            t, tf.log(tf.constant(1.0) + tf.exp(
                (tf.constant(1.0) - tf.reduce_sum(
                    tf.multiply(self.y_input, x_dot_beta))) / t)),
            name='smooth_hinge')
        """

        # must return individual losses and total empirical risk
        losses = tf.add(
            self.smooth_hinge_loss(margin, self.t),
            l2_penalty,
            name='slack_loss')
        emp_risk = tf.reduce_mean(losses, name='emp_risk')

        return losses, emp_risk

    def predict(self, X_test):
        params = self.get_eval(items=['params'])
        beta = params['beta']
        return np.sign(X_test.dot(beta))
