import abc
import sys
import time
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin

import tensorflow as tf
from tensorflow.python.ops import array_ops


class EmpiricalRiskOptimizer(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        self.model_name = 'EmpiricalRiskOptimizer'
        if 'model_name' in kwargs:
            self.model_name = kwargs.pop('model_name')

        # Initialize graph and session
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.graph.as_default()
        self.sess = tf.Session()

        # Config
        self.trained = False
        self.eval_hessian = True

        # Hyperparams
        self.eta = kwargs.pop('eta')
        self.n, self.p = None, None
        self.hyperparams = ['eta']

        # Variables and Inputs
        self.n_params = None
        self.all_params_dict, self.all_params = OrderedDict(), None
        self.X_tr, self.y_tr = None, None
        self.X_te, self.y_te = None, None
        self.feed_dict = None
        self.emp_risk, self.losses = None, None
        self.train_op = None
        self.grad_emp_risk = None
        self.grad_L = None
        if self.eval_hessian:
            self.hessian_emp_risk = None

    def __repr__(self):
        hyperparams_str = ','.join([
            '{}={}'.format(p_, getattr(self, p_))
            for p_ in self.hyperparams])
        __repr = self.model_name + '({})'.format(hyperparams_str)
        return __repr

    def fit(self, X, y, n_iter, verbose=True):
        self.n, self.p = X.shape
        self.X_tr = tf.placeholder(
            tf.float32, (self.n, self.p), name='X_tr')
        self.y_tr = tf.placeholder(
            tf.float32, (self.n, 1), name='y_tr')
        self.all_params = self.get_all_params()
        self.losses, self.emp_risk = self.get_emp_risk()
        self.train_op = self.get_op()
        self.grad_emp_risk = tf.gradients(self.emp_risk, self.all_params)
        self.hessian_emp_risk = tf.hessians(
            self.emp_risk, self.all_params)
        self.grad_L = [tf.placeholder(
            tf.float32, shape=a.get_shape())
            for a in self.all_params]

        # initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # evaluation
        for i in range(n_iter):
            start_time = time.time()
            self.feed_dict = {
                self.X_tr: X,
                self.y_tr: y
            }
            _, loss_val = self.sess.run(
                [self.train_op, self.emp_risk],
                feed_dict=self.feed_dict)

            dur_time = time.time() - start_time
            if verbose:
                if i % 1000 == 0:
                    print('Step %d: loss = %.8f (%.3f sec)' % (
                        i, loss_val, dur_time))

        self.trained = True
        return self

    def get_op(self):
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.eta)
        train_op = optimizer.minimize(self.emp_risk)
        return train_op

    def get_params_by_name(self, key):
        return self.all_params

    def get_all_params(self):
        raise NotImplementedError

    def get_emp_risk(self):
        raise NotImplementedError

    def get_eval(self, all_eval=True):
        assert self.trained is True
        if all_eval is True:
            all_params_val, loss_val = self.sess.run(
                [self.all_params, self.emp_risk],
                feed_dict=self.feed_dict
            )
            grad_loss_val, hessian_loss_val = self.sess.run(
                [self.grad_emp_risk, self.hessian_emp_risk],
                feed_dict=self.feed_dict
            )
            params = {key: val for key, val in zip(
                self.all_params_dict.keys(), all_params_val)}
            grads = {key: val for key, val in zip(
                self.all_params_dict.keys(), grad_loss_val)}
            grads_stacked = np.vstack(grad_loss_val)
            #dim = np.prod(all_params_val.shape)
            hessian = None
            return {
                'params': params,
                'loss': loss_val,
                'grads': grads,
                'grads_stacked': grads_stacked,
                'hessian': hessian_loss_val
            }
        else:
            # only retrieve parameters
            return self.sess.run(
                [self.all_params],
                feed_dict=self.feed_dict
            )

    def predict(self, X_test):
        raise NotImplementedError


class LinearRegressionGD(EmpiricalRiskOptimizer):

    def __init__(self, **kwargs):
        super(LinearRegressionGD, self).__init__(**kwargs)

    def get_all_params(self):
        self.n_params = self.p
        self.all_params_dict['beta'] = tf.get_variable(
            'beta', (self.n_params,1),
            initializer=tf.zeros_initializer)

        return list(self.all_params_dict.values())

    def get_emp_risk(self):
        beta = self.all_params_dict['beta']
        y_hat = tf.matmul(self.X_tr, beta, name='y_hat')
        losses = tf.pow(self.y_tr - y_hat, 2, name='l2_loss')
        emp_risk = tf.reduce_mean(losses, name='emp_risk')
        #stacked_params =
        return losses, emp_risk

    def predict(self, X_test):
        beta = self.get_eval(all_eval=False)
        return X_test.dot(beta)


class LinearRegressionGDSplitParams(EmpiricalRiskOptimizer):
    """
    Only for debug purposes.
    """

    def __init__(self, **kwargs):
        super(LinearRegressionGDSplitParams, self).__init__(**kwargs)

    def get_all_params(self):
        return [tf.get_variable(
            'beta', (self.p, 1),
            initializer=tf.random_normal_initializer)]

    def get_emp_risk(self):
        beta = self.all_params
        y_hat = tf.matmul(self.X_tr, beta, name='y_hat')
        return tf.reduce_sum(
            (self.y_tr - y_hat)**2/self.n, name='emp_risk')

    def predict(self, X_test):
        beta = self.get_eval(all_eval=False)
        return X_test.dot(beta)


if __name__ == '__main__':
    df = pd.read_csv('data/leverage.csv')
    X = df.values[:, 0:2]
    n, p = X.shape
    y = df.values[:, -1].reshape(n, 1)
    model = LinearRegressionGD(
        model_name='LinearRegression',
        eta=0.001)
    model.fit(X,y,n_iter=10000)
    print(model.get_eval())
    #print(json.dumps(model.get_eval(), sort_keys=True,
    #                 indent=4, separators=(',', ': ')))