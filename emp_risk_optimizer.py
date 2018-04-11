import abc
import sys
import time
import json
from collections import OrderedDict
from lazy import lazy

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.ops import array_ops

__author__ = 'zed'
__last_update__ = '2018/04/11'


class EmpiricalRiskOptimizer(BaseEstimator, TransformerMixin):
    """
    Base class for empirical risk minimization.
    Derives from sklearn.TransformerMixin to fit into general sklearn APIs.
    Derived classes must implement 3 abstract methods:
        - get_all_params() to specify all parameters.
        - get_emp_risk() to specify a concrete empirical risk function.
        - predict(X_test) to do inference.

    With all of the above specified, a tensorflow graph will be initialized
    on construction. The base class takes over when fit() is executed.
    It will conduct empirical risk minimization with tensorflow framework,
    compute and store all required gradients, and enable influence function
    evaluation.

    The influence function are calculated with methods:
        - influence_params(method) to compute the influence to parameters
          of every training points.
        - influence_loss(X_valid, y_valid, method) to compute, for a given
          validation set, the influence from every training points.

    The base class also implements get_eval(all_eval=True, **kwargs) method
    to retrieve useful intermediate quantities on demands.
    """
    # TODO: Maybe an alternative constructor to build the graph before fit()?
    # TODO: Retraining routine.
    # TODO: Leave-One-Out routine, and comparisons.
    # TODO: A better optimization procedure, maybe mini-batch?
    # TODO: More concrete instances, maybe 2-classes logistic regression?
    # TODO: Only work for (n*1) labels, should implement multi-classes case
    # TODO: Hessian vector product calculation (QP/Taylor)
    # TODO: Should separate predict and inference for classifications.

    def __init__(self, **kwargs):
        """
        Constructor.
        Makes a tensorflow graph and set it as the default graph.
        Declares all the names for tensorflow objects that will be
        (mostly) initialized in the fit(X, y) method.
        :param kwargs:
            - 'model_name': the label of model, will appear in __repr__.
            - 'eta': learning rate.
            - other hyperparameters for concrete models.
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
        self.hyperparams = ['eta'] # for __repr__

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
        """
        Object representation in sklearn style:
        'model_name(hyperparameter_name=hyperparameter_val)'
        :return: the representation.
        """
        hyperparams_str = ','.join([
            '{}={}'.format(p_, getattr(self, p_))
            for p_ in self.hyperparams])
        __repr = self.model_name + '({})'.format(hyperparams_str)
        return __repr

    def fit(self, X, y, n_iter, verbose=True, **kwargs):
        """
        Fit the model by run tensorflow session, evaluating the optimization
        operation. All tensorflow objects are initialized here.
        After training the model, set self.trained flag as True to enable
        inference & influence function calculation.

        :param X: 2-dim np.ndarray, training feature.
        :param y: 2-dim np.ndarray, training labels.
        :param n_iter: int, number of iterations for optimization procedure.
        :param verbose: bool, whether to print the process.
        :param kwargs:
        :return: self, the fitted model.
        """
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
        # TODO: Maybe we'll need this later?
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
        """
        Build the optimizer and optimization operation.
        Can not execute if self.emp_risk has not been initialized.
        :return: optimization operation.
        """
        # TODO: A better optimization procedure, with mini-batch, etc.
        assert self.emp_risk is not None
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.eta, name='GD')
        train_op = optimizer.minimize(self.emp_risk)
        return train_op

    def get_all_params(self):
        """
        Abstract function that must be implemented.
        Specify all parameters that will be trained.
        It must do at least 2 jobs, plus 1 optional job:

            - 1. Assign a value to self.n_params,
                 the total number of parameters to train.

            - 2. Create a rank-2 tensorflow variable
                 with shape (self.n_params, 1), i.e. a flat container of all
                 parameters. This is used to evaluate the full Hessian of
                 empirical risk with respect to all parameters.
                 The method should return this tensorflow variable, and it
                 be assigned to self.all_params.

        `   - 3. (Optional) Set self.all_params_dict items as one's wish.
                 This is mainly for the convenience to keep track of different
                 blocks/groups of parameters that has different meaning by
                 their names, instead of the positions in the flat all_params
                 vector. It allows the user to build the empirical risk
                 function more easily.
                 Note that self.all_params_dict is an OrderedDict, so
                 unpacking self.all_params_dict.value() returns a list that
                 contains tensorflow variables in the same order as they were
                 inserted to the dict.

        :return: tensorflow variable, self.all_params.
        """
        # (1)
        # self.n_params = ...

        # (2)
        # all_params = tf.get_variable(
        #   'flat_params', (self.n_params, 1),
        #   initializer=tf.zeros_initializer)

        # (3), optional
        # self.all_params_dict[name] = all_params[positions]

        # return all_params

        raise NotImplementedError

    def get_emp_risk(self):
        """
        Abstract function that must be implemented.
        Specify the losses and the empirical risk function,
        It must do 2 jobs:

            - 1. The loss tensor L(X, y; params)
                 For Regression:
                 L: R(n*p) * R(n*1) -> R(n*1)
                    (X, y) -> L(X, y; params)

                 For Multi-labels Classification:
                 L: R(n*p) * R(n*G) -> R(n*G)
                    (X, y) -> L(X, y; params)
                    y: G classes, one-hot labels.

                 One has access to retrieve training data with
                 self.X_tr and self.y_tr; and the parameters
                 self.all_params_dict[name] or self.all_params[position]
                 as specified in get_all_params method.

            - 2. The empirical risk function
                 tf.reduce_mean(losses, name='emp_risk')

        :return: two tensorflow variable/operations, (losses, emp_risk)
                 they will then be assigned to self.losses and self.emp_risk.
        """
        # (1)
        # losses = ...

        # (2)
        # emp_risk = tf.reduce_mean(losses, name='emp_risk')
        # TODO: Any other possible forms? If not, should move into fit().

        # return losses, emp_risk

        raise NotImplementedError

    def predict(self, X_test):
        """
        Abstract function that must be implemented.
        sklearn API Prediction method
        :param X_test: 2-dim np.ndarray, testing features.
        :return: 2-dim np.ndarray, the predictions y_pred.
        """
        # return y_pred
        raise NotImplementedError

    def fit_transform(self, X, y=None, **fit_params):
        """
        sklearn API fit_transform method.
        :param X: 2-dim np.ndarray, training/testing features.
        :param y: Optional, 2-dim np.ndarray, training labels.
        :param fit_params: other fitting parameters.
        :return: If trained, return prediction, otherwise fit and
                 return prediction on training set.
        """
        if self.trained is True:
            return self.predict(X)
        else:
            assert y is not None
            self.fit(X, y, **fit_params)
            return self.predict(X)

    @lazy
    def inv_hessian(self):
        """
        Evaluate the hessian inverse.
        Note: Only evaluate once, should create an another object
              for re-evaluation.
        :return: 2-dim np.ndarray, the inverse of
                 the hessian of empirical risk wrt all params.
        """
        try:
            H = self.get_eval(items=['hessian'])
            H_inv = np.linalg.inv(H)
            return H_inv
        except np.linalg.LinAlgError as e:
            print(str(e))

    def influence_params(self, method):
        """
        Compute every training points' influence to parameters.
        :param method: string, the method to be adopted.
            - 'brute-force': Explicitly calculate H^-1 grad(L(z))
               for every training point z.
        :return: a list of n 2-dim np.ndarray with shape (self.n_params, 1).
                 The i-th item is the i-th training point's influence to
                 all parameters.
        """
        assert self.trained is True
        if method == 'brute-force':
            H_inv = self.inv_hessian
            grads_train_elems = self.get_per_elem_gradients()
            return [H_inv.dot(v) for v in grads_train_elems]

    def influence_loss(self, X_valid, y_valid, method):
        """
        Compute every training points' influence to every validation points.
        :param X_valid: 2-dim np.ndarray, validation features.
        :param y_valid: 2-dim np.ndarray, validation labels.
        :param method: the method to be adopted.
            - 'brute-force': Explicitly calculate
               grad(L(z)) (H^-1) grad(L(z_test))
               for every training point z and every test point z_test.
        :return: 2-dim np.ndarray of shape (n_train, n_test), [i,j]-th entry
                 is the i-th training point's influence
                 to the j-th validation point.
        """
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
        """
        Calculate per element (not-aggregated)
        gradients with respect to parameters.
        :param train_idx: list of ints, the training points' indices on which
            the gradients are calculated. If not specified, calculate for
            all training points.
        :param kwargs:
            - z_valid: tuple of two 2-dim np.ndarray.
              The validation data. If specified, calculate per element
              gradient with respect to parameters on the validation set.
        :return:
        """
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
        """
        Evaluate and return quantities on requests.
        :param all_eval: bool, if True, evaluate all items.
        :param kwargs:
            - items: list of strings, if specified, only
              return these items.
        :return:
        """
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

    @staticmethod
    def serialize_eval_dict(eval_dict, pretty=False):
        """
        Serialize an evaluation outcome dictionary to json formatted string.
        Might be saved to disk or nicely printed out later.
        :param eval_dict: a returned dictionary from
            EmpiricalRiskOptimizer.get_eval()
        :param pretty: bool, whether to prettify or not.
        :return: str, in json parse-able format.
        """
        json_dict = dict()
        for k1, val1 in eval_dict.items():
            if type(val1) is np.ndarray:
                if k1 is not 'hessian':
                    json_dict[k1] = val1.reshape(
                        1, val1.shape[0]).tolist()
                else:
                    json_dict[k1] = val1.tolist()
            elif type(val1) is dict:
                json_dict[k1] = dict()
                for k2, val2 in eval_dict[k1].items():
                    if type(val2) is np.ndarray:
                        json_dict[k1][k2] = val2.reshape(
                            1, val2.shape[0]).tolist()
        if pretty:
            return json.dumps(
                json_dict, sort_keys=True,
                indent=4, separators=(',', ': '))
        return json.dumps(json_dict)


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
    X_data = df.values[:, 0:10]
    y_data = df.values[:, -1].reshape(n, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42)
    for data in [X_train, X_test, y_train, y_test]:
        print(data.shape)
    n, p = X_train.shape

    model = LinearRegression2Blocks(
        model_name='LinearRegression2Blocks',
        eta=0.001)

    model.fit(X_train, y_train, n_iter=10000)
    I_loss = model.influence_loss(
        X_test, y_test, method='brute-force')
    d = model.serialize_eval_dict(model.get_eval())

