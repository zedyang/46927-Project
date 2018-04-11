import abc
import sys
import time
import json
from collections import OrderedDict
from lazy import lazy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

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
        self.data = None
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
        self.grad_individual = None
        if self.eval_hessian:
            self.hessian_emp_risk = None

    def __repr__(self):
        hyperparams_str = ','.join([
            '{}={}'.format(p_, getattr(self, p_))
            for p_ in self.hyperparams])
        __repr = self.model_name + '({})'.format(hyperparams_str)
        return __repr

    def fit(self, X, y, n_iter, verbose=True, **kwargs):
        self.n, self.p = X.shape
        self.data = {'X': X, 'y': y}
        self.X_tr = tf.placeholder(
            tf.float32, (None, self.p), name='X_tr')
        self.y_tr = tf.placeholder(
            tf.float32, (None, 1), name='y_tr')
        self.all_params = self.get_all_params()
        self.losses, self.emp_risk = self.get_emp_risk()
        self.train_op = self.get_op()
        self.hessian_emp_risk = tf.hessians(
            self.emp_risk, self.all_params, name='H_total_risk')
        # self.grad_individual = [tf.gradients(
        #    self.losses, self.all_params)
        #    for i in range(self.n)]
        # tf.add_to_collection('grads', self.grad_individual)
        self.grad_emp_risk = tf.gradients(
            self.emp_risk, self.all_params, name='grad_total_risk')

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
            learning_rate=self.eta, name='GD')
        train_op = optimizer.minimize(self.emp_risk)
        return train_op

    def get_all_params(self):
        raise NotImplementedError

    def get_emp_risk(self):
        raise NotImplementedError

    @lazy
    def inv_hessian(self):
        try:
            H = self.get_eval(items=['hessian'])
            H_inv = np.linalg.inv(H)
            return H_inv
        except np.linalg.LinAlgError as e:
            print(str(e))

    def influence_params(self, method):
        assert self.trained is True
        if method == 'brute-force':
            H_inv = self.inv_hessian
            grads_train_elems = self.get_per_elem_gradients()
            return [H_inv.dot(v) for v in grads_train_elems]

    def influence_loss(self, X_valid, y_valid, method):
        assert self.trained is True
        if method == 'brute-force':
            H_inv = self.inv_hessian
            grads_train_elems = self.get_per_elem_gradients()
            U = np.stack(grads_train_elems, axis=1)
            grads_valid_elems = self.get_per_elem_gradients(
                z_valid=(X_valid, y_valid))
            influence_loss_val = np.zeros((self.n, X_valid.shape[0]))
            for idx, v in enumerate(grads_valid_elems):
                hvp = H_inv.dot(v)
                I_loss_z = U.T.dot(hvp)
                influence_loss_val[:, idx:idx+1] = I_loss_z
            return influence_loss_val

    def get_per_elem_gradients(self, train_idx=None, **kwargs):
        assert self.trained is True
        grads_per_elem = []

        if 'z_valid' in kwargs:
            # evaluate gradients for validation points
            z_valid = kwargs.pop('z_valid')
            X_valid, y_valid = z_valid

            for idx in range(y_valid.shape[0]):
                single_elem_feed_dict = {
                    self.X_tr: X_valid[idx:idx + 1, :],
                    self.y_tr: y_valid[idx:idx + 1, :]
                }
                grad_emp_risk_val = self.sess.run(
                    self.grad_emp_risk,
                    feed_dict=single_elem_feed_dict
                )
                grads_per_elem.append(
                    np.vstack(grad_emp_risk_val))
        else:
            # evaluate gradients for training points
            if not train_idx:
                train_idx = range(self.n)
            start_time = time.time()
            for counter, idx in enumerate(train_idx):
                single_elem_feed_dict = {
                    self.X_tr: self.data['X'][idx:idx+1, :],
                    self.y_tr: self.data['y'][idx:idx+1, :]
                }
                grad_emp_risk_val = self.sess.run(
                    self.grad_emp_risk,
                    feed_dict=single_elem_feed_dict
                )
                grads_per_elem.append(
                    np.vstack(grad_emp_risk_val))
            dur_time = time.time() - start_time
            print('Fetch training loss gradients (%.3f sec)' % dur_time)

        return grads_per_elem

    def get_eval(self, all_eval=True, **kwargs):
        assert self.trained is True
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

        eval_dict = {
            'params': params,
            'params_flat': params_flat,
            'losses': losses_val,
            'emp_risk': emp_risk_val,
            'grads_stacked': grads_stacked,
            'hessian': hessian
        }

        if all_eval is True and 'items' not in kwargs:
            return eval_dict
        else:
            # only retrieve selected items
            items = kwargs.pop('items')
            if len(items) == 1:
                return eval_dict[items[0]]
            return {key: eval_dict[key] for key in items}

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
    n = len(df)
    X = df.values[:, 0:10]
    y = df.values[:, -1].reshape(n, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for data in [X_train, X_test, y_train, y_test]:
        print(data.shape)
    n, p = X_train.shape

    model = LinearRegression2Blocks(
        model_name='LinearRegression2Blocks',
        eta=0.001)

    model.fit(X_train, y_train, n_iter=10000)
    I_loss = model.influence_loss(
        X_test, y_test, method='brute-force')
    print(I_loss)
