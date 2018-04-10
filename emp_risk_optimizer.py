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
        # self.all_params_dict contains structured params for convenience
        # self.all_params is a flat (p*1) container that stacks all params
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
            for a in self.all_params_dict.values()]

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

    def get_all_params(self):
        raise NotImplementedError

    def get_emp_risk(self):
        raise NotImplementedError

    def get_eval(self, all_eval=True):
        assert self.trained is True
        if all_eval is True:
            all_params_blocks = self.sess.run(
                list(self.all_params_dict.values()),
                feed_dict=self.feed_dict
            )
            losses_val, emp_risk_val = self.sess.run(
                [self.losses, self.emp_risk],
                feed_dict=self.feed_dict
            )
            grad_loss_val, hessian_loss_val = self.sess.run(
                [self.grad_emp_risk, self.hessian_emp_risk],
                feed_dict=self.feed_dict
            )
            params = {key: val for key, val in zip(
                self.all_params_dict.keys(), all_params_blocks)}
            params_flat = np.concatenate(tuple(params.values()), 0)
            grads_stacked = np.vstack(grad_loss_val)
            hessian = np.vstack(hessian_loss_val).reshape(
                self.n_params, self.n_params)
            return {
                'params': params,
                'params_flat': params_flat,
                'losses': losses_val,
                'emp_risk': emp_risk_val,
                'grads_stacked': grads_stacked,
                'hessian': hessian
            }
        else:
            # only retrieve parameters
            return self.sess.run(
                [self.all_params],
                feed_dict=self.feed_dict
            )

    def predict(self, X_test):
        raise NotImplementedError


class LinearRegression2Blocks(EmpiricalRiskOptimizer):
    """
    Only for debug purposes.
    """

    def __init__(self, **kwargs):
        super(LinearRegression2Blocks, self).__init__(**kwargs)
        self.p1 = None

    def get_all_params(self):
        self.n_params = self.p
        # split X into 2 blocks, for debug purpose only
        self.p1 = int(self.p/2)

        # build the flat params container
        self.all_params = tf.get_variable(
            'flat_params', (self.n_params, 1),
            initializer=tf.zeros_initializer)

        # build the params dict
        self.all_params_dict[
            'beta_block1'] = self.all_params[0:self.p1, :]
        self.all_params_dict[
            'beta_block2'] = self.all_params[self.p1:self.n_params, :]

        return self.all_params

    def get_emp_risk(self):
        # params dict split the params into groups, for convenience
        b1 = self.all_params_dict['beta_block1']
        b2 = self.all_params_dict['beta_block2']

        # y_hat
        y_hat = tf.add(
            tf.matmul(self.X_tr[:, 0:self.p1], b1, name='block1'),
            tf.matmul(self.X_tr[:, self.p1:self.p], b2, name='block1'),
            name='y_hat'
        )

        # must return individual losses and total empirical risk
        losses = tf.pow(self.y_tr - y_hat, 2, name='l2_loss')
        emp_risk = tf.reduce_mean(losses, name='emp_risk')
        return losses, emp_risk

    def predict(self, X_test):
        beta = self.get_eval(all_eval=False)
        return X_test.dot(beta)


if __name__ == '__main__':
    df = pd.read_csv('data/lm_10.csv')
    X = df.values[:, 0:10]
    n, p = X.shape
    y = df.values[:, -1].reshape(n, 1)
    model = LinearRegression2Blocks(
        model_name='LinearRegression2Blocks',
        eta=0.001)
    model.fit(X,y,n_iter=10000)
    print(model.get_eval())
    #print(json.dumps(model.get_eval(), sort_keys=True,
    #                 indent=4, separators=(',', ': ')))