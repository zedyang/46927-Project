import os
import time
import json
from collections import OrderedDict

import numpy as np
from scipy.optimize import fmin_ncg
from sklearn.base import BaseEstimator, TransformerMixin

import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops

__author__ = 'zed'
__last_update__ = '2018/04/22'


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

    # TODO: Should separate predict and inference for classifications.

    def __init__(self, **kwargs):
        """
        Constructor.
        Makes a tensorflow graph and set it as the default graph.
        Declares all the names for tensorflow objects that will be
        (mostly) initialized in the fit(X, y) method.
        :param kwargs:
            - 'model_name': the label of model, will appear in __repr__.
            - 'eta': float, initial learning rate.
            - 'batch_size': int, size of a minibatch.
            - 'decay_epochs': list of ints,
               the epochs to apply learning rate decay.
            - other hyperparameters for concrete models.
        """
        self.model_name = 'EmpiricalRiskOptimizer'
        if 'model_name' in kwargs:
            self.model_name = kwargs.pop('model_name')
        self.checkpoint_file = os.path.join(
            'output',
            "%s-checkpoint" % self.model_name)

        # Initialize graph and session
        self.graph = tf.Graph()
        self.graph.as_default()
        self.sess = None
        self.saver = None

        # Config
        self.trained = False

        # Hyperparams
        self.init_eta = kwargs.pop('init_eta')
        self.batch_size = kwargs.pop('batch_size')
        self.decay_epochs = kwargs.pop('decay_epochs')
        self.n_samples, self.n_features = None, None
        self.hyperparams = ['init_eta', 'batch_size', 'decay_epochs']
        # for __repr__

        # Variables and Inputs
        self.data = None
        self._data_batch = None
        self.n_classes = None
        self._idx_in_epoch = 0
        self.global_step = None
        self.adaptive_eta = None
        self.eta_input, self.eta_assign = None, None
        self.n_params = None
        self.all_params_dict, self.all_params = OrderedDict(), None
        self.all_params_assign, self.all_params_input = None, None
        # self.all_params_dict contains structured params for convenience
        # self.all_params is a flat
        # (n_features*1) container that stacks all params
        self.X_input, self.y_input = None, None

        # Fitting
        self.emp_risk, self.losses = None, None
        self.train_op_adam = None
        self.train_op_sgd = None
        self.grad_emp_risk = None
        self.grad_individual = None
        self.hessian_emp_risk = None
        # hessian vector prod = Hv, v_placeholder is the placeholder for v
        self.hessian_vector_product = None
        self.v_placeholder = None
        self.sklearn_clf = None
        self.grads_v_products = None

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

    def get_new_feed_dict(self, feed_data):
        """

        :param feed_data:
        :return:
        """
        if 'X' in feed_data and 'y' in feed_data:
            feed_dict = {
                self.X_input: feed_data['X'],
                self.y_input: feed_data['y']}
        else:
            feed_dict = {
                self.X_input: self.data['X'],
                self.y_input: self.data['y']}
        if 'v' in feed_data and feed_data['v'] is not None:
            feed_dict[self.v_placeholder] = feed_data['v']
        return feed_dict

    def get_single_elem_feed_dict(self, idx, feed_data=None):
        """

        :param idx:
        :param feed_data:
        :return:
        """
        if feed_data is None:
            feed_data = self.data
        X, y = feed_data['X'], feed_data['y']
        return {
            self.X_input: X[idx:idx + 1, :],
            self.y_input: y[idx:idx + 1, :]
        }

    def get_batch_feed_dict(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.n_samples
        start = self._idx_in_epoch
        self._idx_in_epoch += batch_size
        # if used for one pass, re-shuffle the data
        if self._idx_in_epoch > self.n_samples:
            perm = np.arange(self.n_samples)
            np.random.shuffle(perm)
            self._data_batch = {
                'X': self._data_batch['X'][perm, :],
                'y': self._data_batch['y'][perm, :]
            }
            start = 0
            self._idx_in_epoch = batch_size
        end = self._idx_in_epoch
        return {
            self.X_input: self._data_batch['X'][start:end],
            self.y_input: self._data_batch['y'][start:end]
        }

    def load(self, iter_to_load):
        checkpoint = "%s-%s" % (self.checkpoint_file, iter_to_load)
        self.saver.restore(self.sess, checkpoint)

    def update_eta(self, current_iter):
        """
        Shrink eta while going deep into high iterations.
        :param current_iter:
        :return:
        """
        assert self.n_samples % self.batch_size
        n_iters_in_epoch = self.n_samples / self.batch_size
        current_epoch = current_iter // n_iters_in_epoch
        if current_epoch < self.decay_epochs[0]:
            decay = 1
        elif current_epoch < self.decay_epochs[1]:
            decay = 0.1
        else:
            decay = 0.01
        self.sess.run(
            self.eta_assign,
            feed_dict={self.eta_input: decay*self.init_eta}
        )
        return decay, current_epoch

    def fit(self, X, y, n_iter,
            iter_to_switch_off_minibatch=20000,
            iter_to_switch_to_sgd=40000,
            verbose=True, show_eval=True,
            lazy_fit=False, **kwargs):
        """
        Fit the model by run tensorflow session, evaluating the optimization
        operation. All tensorflow objects are initialized here.
        After training the model, set self.trained flag as True to enable
        inference & influence function calculation.

        :param X: 2-dim np.ndarray, training feature.
        :param y: 2-dim np.ndarray, training labels.
        :param n_iter: int, number of iterations for optimization procedure.
        :param iter_to_switch_off_minibatch: int
        :param iter_to_switch_to_sgd: int
        :param verbose: bool, whether to print the process.
        :param show_eval: bool, whether to show evaluation upon completion.
        :param lazy_fit: bool, whether to fit now or just initialize containers.
        :param kwargs:
        :return: self, the fitted model.
        """
        if 'refit' in kwargs and kwargs.pop('refit') is True:
            assert self.trained is True
        else:
            # tf.reset_default_graph()
            self.sess = tf.Session()

            # data and minibatch utils
            self.n_samples, self.n_features = X.shape
            _, g = y.shape
            self.data = {'X': X, 'y': y}
            self._data_batch = {'X': np.copy(X), 'y': np.copy(y)}
            self.n_classes = g if g > 1 else 2
            self.X_input = tf.placeholder(
                tf.float32, (None, self.n_features), name='X_input')
            self.y_input = tf.placeholder(
                tf.float32, (None, g), name='y_input')
            self.global_step = tf.Variable(
                0, name='batch_counter', trainable=False)

            # training operations & utils
            self.adaptive_eta = tf.Variable(
                0.0, name='batch_counter', trainable=False, dtype=tf.float32)
            self.eta_input = tf.placeholder(tf.float32)
            self.eta_assign = tf.assign(self.adaptive_eta, self.eta_input)
            self.all_params = self.get_all_params()
            self.all_params_input = tf.placeholder(
                tf.float32, (self.n_params, 1), name='params_input')
            self.all_params_assign = tf.assign(
                self.all_params, self.all_params_input)
            self.losses, self.emp_risk = self.get_emp_risk()
            self.train_op_adam, self.train_op_sgd = self.get_train_op()

            # gradients, hessian, and hvp.
            self.grad_emp_risk = tf.gradients(
                self.emp_risk, self.all_params, name='grad_total_risk')
            if hasattr(self, 'hessian_emp_risk'):
                self.hessian_emp_risk = tf.hessians(
                    self.emp_risk, self.all_params, name='H_total_risk')
            self.v_placeholder = tf.placeholder(
                tf.float32, (None, 1), name='v')
            self.hessian_vector_product = self.get_hessian_vector_product(
                self.emp_risk, [self.all_params], [self.v_placeholder],
                first_grads=self.grad_emp_risk)

            # misc
            self.saver = tf.train.Saver()
        if 'traceback_checkpoint' in kwargs:
            traceback_checkpoint = kwargs.pop('traceback_checkpoint')
        else:
            traceback_checkpoint = n_iter

        # initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # evaluation
        if not lazy_fit:
            for i in range(n_iter):
                start_time = time.time()
                _, epoch = self.update_eta(current_iter=i)

                if i < iter_to_switch_off_minibatch:
                    feed_dict = self.get_batch_feed_dict()
                    _, emp_risk_val = self.sess.run(
                        [self.train_op_adam, self.emp_risk],
                        feed_dict=feed_dict)

                elif i < iter_to_switch_to_sgd:
                    feed_dict = self.get_new_feed_dict(self.data)
                    _, emp_risk_val = self.sess.run(
                        [self.train_op_adam, self.emp_risk],
                        feed_dict=feed_dict)

                else:
                    feed_dict = self.get_new_feed_dict(self.data)
                    _, emp_risk_val = self.sess.run(
                        [self.train_op_sgd, self.emp_risk],
                        feed_dict=feed_dict)

                dur_time = time.time() - start_time
                if verbose:
                    if i % int(1000 / verbose) == 0:
                        print('Step %d, Epoch %d: loss = %.8f (%.3f sec)' % (
                            i, epoch, emp_risk_val, dur_time))

                # checkpoint
                if (i + 1) % 100000 == 0 or (
                            (i + 1) == traceback_checkpoint) or (
                            (i + 1) == n_iter):
                    self.saver.save(
                        self.sess, self.checkpoint_file, global_step=i)

            self.trained = True
            if show_eval:
                self.show_eval()
        return self

    def refit_with_feed_dict(self, feed_dict, n_iter,
                             force_restart=False, verbose=False):
        """


        :param feed_dict:
        :param n_iter:
        :param force_restart:
        :param verbose:
        :return:
        """
        assert self.trained is True

        emp_risk_val = None
        if force_restart:
            # force to reinitialize all variables
            init = tf.global_variables_initializer()
            self.sess.run(init)

        for i in range(n_iter):
            start_time = time.time()
            _, emp_risk_val = self.sess.run(
                [self.train_op_adam, self.emp_risk],
                feed_dict=feed_dict)
            dur_time = time.time() - start_time
            if verbose:
                if i % 1000 == 0:
                    print('Step %d: loss = %.8f (%.3f sec)' % (
                        i, emp_risk_val, dur_time))
        return emp_risk_val

    def fit_with_sklearn(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def refit_with_sklearn(self, feed_dict, **kwargs):
        """

        :param feed_dict:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def get_train_op(self):
        """
        Build the optimizer and optimization operation.
        Can not execute if self.emp_risk has not been initialized.
        :return: optimization operation.
        """
        assert self.emp_risk is not None
        adam_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.adaptive_eta, name='Adam')
        gd_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.adaptive_eta, name='GD')
        train_op_gd = gd_optimizer.minimize(
            self.emp_risk, global_step=self.global_step)
        train_op_adam = adam_optimizer.minimize(
            self.emp_risk, global_step=self.global_step)
        return train_op_adam, train_op_gd

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
                 L: R(n_samples*n_features) * R(n_samples*1) -> R(n_samples*1)
                    (X, y) -> L(X, y; params)

                 For Multi-labels Classification:
                 L: R(n_samples*n_features) * R(n_samples*G) -> R(n_samples*G)
                    (X, y) -> L(X, y; params)
                    y: G classes, one-hot labels.

                 One has access to retrieve training data with
                 self.X_input and self.y_input; and the parameters
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

    def get_hessian_vector_product(self, ys, xs, v, first_grads=None, **kwargs):
        """

        :param ys:
        :param xs:
        :param v:
        :param first_grads:
        :param kwargs:
        :return:
        """
        assert len(xs) == len(v)
        if first_grads is None:
            first_grads = tf.gradients(ys, xs)
        """
        self.grads_v_products = [
            math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
            for grad_elem, v_elem in zip(first_grads, v)
            if grad_elem is not None
        ]
        """

        self.grads_v_products = tf.reduce_sum(
            first_grads*tf.stop_gradient(v), name='grads_dot_v')

        name = kwargs.pop('name') if 'name' in kwargs \
            else 'hessian_vector_product'
        return tf.gradients(self.grads_v_products, xs, name=name)

    def inv_hessian(self, damping=0.0):
        """
        :param damping: float, l2 regularization added to hessian.
        :return: 2-dim np.ndarray, the inverse of
                 the hessian of empirical risk wrt all params.
        """
        try:
            H = self.get_eval(items=['hessian'])
            H += damping*np.eye(H.shape[0])
            H_inv = np.linalg.inv(H)
            return H_inv
        except np.linalg.LinAlgError as e:
            print(str(e))

    def eval_hvp(self, v, damping=0.0, verbose=False):
        """

        :param v:
        :param damping:
        :param verbose:
        :return:
        """
        if v.ndim == 1:
            v = v.reshape((v.shape[0], 1))
        feed_dict = self.get_new_feed_dict({'v': v})
        hessian_vector_product_val = self.sess.run(
            self.hessian_vector_product, feed_dict=feed_dict)
        hessian_vector_product_val = np.concatenate(
            hessian_vector_product_val, axis=0)
        hessian_vector_product_val += damping*v
        if verbose:
            print('Hvp norm:', np.linalg.norm(
                hessian_vector_product_val))
        return hessian_vector_product_val

    def eval_hvp_minibatch(self, v, damping=0.0, verbose=False):
        """

        :param v:
        :param damping:
        :param verbose:
        :return:
        """
        if v.ndim == 1:
            v = v.reshape((v.shape[0], 1))
        n_iters = int(self.n_samples/self.batch_size)
        hessian_vector_product_val = np.array([])
        if verbose:
            print("Evaluating hvp.", end='')
        for i in range(n_iters):
            feed_dict = self.get_batch_feed_dict()
            feed_dict[self.v_placeholder] = v
            hvp_batch = self.sess.run(
                self.hessian_vector_product,
                feed_dict=feed_dict)
            hvp_batch = np.concatenate(hvp_batch, axis=0)
            if i == 0:
                hessian_vector_product_val = hvp_batch / n_iters
            else:
                hessian_vector_product_val += hvp_batch / n_iters
            print(".", end='')
        hessian_vector_product_val += damping * v
        if verbose:
            print('Hvp norm:', np.linalg.norm(
                hessian_vector_product_val))
        return hessian_vector_product_val

    def influence_params(self, method, damping=0.0):
        """
        Compute every training points' influence to parameters.
        :param method: string, the method to be adopted.
            - 'brute-force': Explicitly calculate H^-1 grad(L(z))
               for every training point z.
        :param damping: float, l2 regularization added to hessian.
        :return: a list of n_samples 2-dim np.ndarray with shape
                 (self.n_params, 1). The i-th item is the i-th
                 training point's influence to all parameters.
        """
        assert self.trained is True
        if method == 'brute-force':
            H_inv = self.inv_hessian(damping=damping)
            grads_train_elems = self.get_per_elem_gradients()
            return [H_inv.dot(v) for v in grads_train_elems]

    def influence_loss(self, X_valid, y_valid, method,
                       damping=0.0, minibatch=False,
                       leave_indices=None, **kwargs):
        """
        Compute every training points' influence to every validation points.
        :param X_valid: 2-dim np.ndarray, validation features.
        :param y_valid: 2-dim np.ndarray, validation labels.
        :param method: the method to be adopted.
            - 'brute-force': Explicitly calculate
               grad(L(z)) (H^-1) grad(L(z_test))
               for every training point z and every test point z_test.
        :param damping: float, l2 regularization added to hessian.
        :param minibatch: bool, whether to use minibatch evaluation
               for hessian vector product or not.
        :param leave_indices: list of ints
        :return: 2-dim np.ndarray of shape (n_train, n_test), [i,j]-th entry
                 is the i-th training point's influence
                 to the j-th validation point.
        """
        assert self.trained is True
        grads_train_elems = self.get_per_elem_gradients(
            train_idx=leave_indices)
        U = np.stack(grads_train_elems, axis=1)
        grads_valid_elems = self.get_per_elem_gradients(
            z_valid=(X_valid, y_valid))

        influence_loss_val = np.zeros(
            (self.n_samples if leave_indices is None else len(leave_indices),
             X_valid.shape[0]))

        for idx, v in enumerate(grads_valid_elems):
            if method == 'brute-force':
                H_inv = self.inv_hessian(damping=damping)
                # this is a dim-2 (n_params*1) np.ndarray
                inverse_hvp = H_inv.dot(v)
            elif method == 'cg':
                # this is a dim-1 (n_params,) np.array
                tol, max_iter = 1e-5, 1000
                if 'tol' in kwargs:
                    tol = kwargs['tol']
                if 'max_iter' in kwargs:
                    max_iter = kwargs['max_iter']
                inverse_hvp = self.get_inverse_hvp_cg(
                    v, damping=damping, minibatch=minibatch,
                    tol=tol, max_iter=max_iter)
                inverse_hvp = np.array(
                    inverse_hvp).reshape((self.n_params, 1))
            elif method == 'lissa':
                repeat, scale, depth = 1, 10, 1000
                batch_size = self.batch_size
                verbose = 10
                if 'scale' in kwargs:
                    scale = kwargs['scale']
                if 'repeat' in kwargs:
                    repeat = kwargs['repeat']
                if 'depth' in kwargs:
                    depth = kwargs['depth']
                if 'batch_size' in kwargs:
                    batch_size = kwargs['batch_size']
                if 'verbose' in kwargs:
                    verbose = kwargs['verbose']
                inverse_hvp = self.get_inverse_hvp_lissa(
                    v, damping=damping, repeat=repeat, scale=scale,
                    recursion_depth=depth, batch_size=batch_size,
                    verbose=verbose
                )
            else:
                raise ValueError
            print(f'Influence evaluated for testing point {idx}.')
            I_loss_z = U.T.dot(inverse_hvp)
            influence_loss_val[:, idx:idx + 1] = I_loss_z
        return influence_loss_val

    def get_inverse_hvp_lissa(self, v, damping=0.0, scale=10,
                              batch_size=None, verbose=100,
                              recursion_depth=1000, repeat=1):
        """

        :param v:
        :param damping:
        :param scale:
        :param batch_size:
        :param verbose:
        :param recursion_depth:
        :param repeat:
        :return:
        """
        inverse_hvp = None
        for i in range(repeat):
            print(f'--- Lissa Sample {i} ---')
            curr_estimate = v

            # Taylor recursion
            for j in range(recursion_depth):
                feed_dict = self.get_batch_feed_dict(
                    batch_size=batch_size)
                feed_dict[self.v_placeholder] = curr_estimate
                hessian_vector_val = self.sess.run(
                    self.hessian_vector_product,
                    feed_dict=feed_dict)

                hessian_vector_val = np.concatenate(
                    hessian_vector_val)
                curr_estimate = v + (1-damping)*curr_estimate \
                    - hessian_vector_val/scale
                if j % verbose == 0:
                    norm = np.linalg.norm(
                        np.concatenate(curr_estimate))
                    print(f"Recursion depth: {j}, hvp norm: {norm}")

                    """
                    grad_v_products_val = self.sess.run(
                        self.grads_v_products,
                        feed_dict=feed_dict)
                    print(hessian_vector_val, '***')
                    """

            if inverse_hvp is None:
                inverse_hvp = curr_estimate/scale
            else:
                inverse_hvp += curr_estimate/scale

        # average over r estimates
        inverse_hvp /= repeat
        return inverse_hvp

    def get_inverse_hvp_cg(self, v, damping=0.0, minibatch=False,
                           tol=1e-5, max_iter=1000):
        """

        :param v:
        :param damping:
        :param minibatch:
        :param tol:
        :param max_iter:
        :return:
        """
        assert damping >= 0
        if not minibatch:
            def __cg_objective(x):
                Hx = self.eval_hvp(x)
                obj = np.multiply(0.5, x.T.dot(Hx)) - v.T.dot(x)
                # d0, = obj.shape
                return obj

            def __cg_grad(x):
                Hx = self.eval_hvp(x)
                d0, d1 = Hx.shape
                return (Hx - v).reshape((d0 * d1,))

            def __cg_fHess_p(x, p):
                Hp = self.eval_hvp(p)
                d0, d1 = Hp.shape
                return Hp.reshape((d0 * d1,))
        else:
            def __cg_objective(x):
                Hx = self.eval_hvp_minibatch(x)
                obj = np.multiply(0.5, x.T.dot(Hx)) - v.T.dot(x)
                # d0, = obj.shape
                return obj

            def __cg_grad(x):
                Hx = self.eval_hvp_minibatch(x)
                d0, d1 = Hx.shape
                return (Hx - v).reshape((d0 * d1,))

            def __cg_fHess_p(x, p):
                Hp = self.eval_hvp_minibatch(p)
                d0, d1 = Hp.shape
                return Hp.reshape((d0 * d1,))

        def __cg_callback(x):
            print('CG Objective: %s' % __cg_objective(x)[0])

        cg_min_results = fmin_ncg(
            f=__cg_objective,
            x0=np.concatenate(v),
            fprime=__cg_grad,
            fhess_p=__cg_fHess_p,
            callback=__cg_callback,
            avextol=tol,
            maxiter=max_iter)
        return cg_min_results

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
        :return: A list of (n_features * 1) np.ndarray objects, (list len is m)
            n_features is the total number of params;
            m is the number of elements.
        """
        assert self.trained is True
        grads_per_elem = []

        if 'z_valid' in kwargs:
            # evaluate gradients for validation points
            z_valid = kwargs.pop('z_valid')
            X_valid, y_valid = z_valid

            for idx in range(y_valid.shape[0]):
                grad_emp_risk_val = self.sess.run(
                    self.grad_emp_risk,
                    feed_dict=self.get_single_elem_feed_dict(
                        idx, {'X': X_valid, 'y': y_valid})
                )
                grads_per_elem.append(
                    np.vstack(grad_emp_risk_val))
        else:
            # evaluate gradients for training points
            if train_idx is None:
                train_idx = range(self.n_samples)
            start_time = time.time()
            for counter, idx in enumerate(train_idx):
                grad_emp_risk_val = self.sess.run(
                    self.grad_emp_risk,
                    feed_dict=self.get_single_elem_feed_dict(idx)
                )
                grads_per_elem.append(
                    np.vstack(grad_emp_risk_val))
            dur_time = time.time() - start_time
            print('Fetch training loss gradients (%.3f sec)' % dur_time)

        return grads_per_elem

    def get_eval(self, items):
        """
        Evaluate and return quantities on requests.
        :param items: list of strings, if specified, only
              return these items.
        :return:
        """
        assert self.trained is True
        feed_dict = self.get_new_feed_dict(self.data)
        eval_dict = dict()
        if 'params' in items:
            all_params_blocks = self.sess.run(
                list(self.all_params_dict.values()),
                feed_dict=feed_dict
            )
            params = {key: val for key, val in zip(
                self.all_params_dict.keys(), all_params_blocks)}
            eval_dict['params'] = params

        if 'params_flat' in items:
            all_params_flat_val = self.sess.run(
                self.all_params,
                feed_dict=feed_dict
            )
            eval_dict['params_flat'] = all_params_flat_val

        if 'emp_risk' in items:
            emp_risk_val = self.sess.run(
                self.emp_risk,
                feed_dict=feed_dict
            )
            eval_dict['emp_risk'] = emp_risk_val

        if 'losses' in items:
            losses_val = self.sess.run(
                self.losses,
                feed_dict=feed_dict
            )
            eval_dict['losses'] = losses_val

        if 'grads' in items:
            grad_loss_val = self.sess.run(
                self.grad_emp_risk,
                feed_dict=feed_dict
            )
            grads_stacked = np.vstack(grad_loss_val)
            eval_dict['grads'] = grads_stacked

        if 'hessian' in items:
            hessian_loss_val = self.sess.run(
                self.hessian_emp_risk,
                feed_dict=feed_dict
            )
            hessian = np.vstack(hessian_loss_val).reshape(
                self.n_params, self.n_params)
            eval_dict['hessian'] = hessian

        if len(items) == 1:
            return eval_dict[items[0]]
        else:
            return eval_dict

    def show_eval(self):
        assert self.trained is True
        evals = self.get_eval(
            items=['params_flat', 'emp_risk', 'grads'])
        params_ = evals['params_flat']
        emp_risk_ = evals['emp_risk']
        grads_ = evals['grads']
        print("\nModel Evaluations:")
        print("------------------------------------------")
        print("Empirical Risk: %.6f" % emp_risk_)
        print("Norm of Params: %.6f" % np.linalg.norm(params_))
        print("Norm of Gradient: %.6f" % np.linalg.norm(grads_))

    def leave_one_out_refit(self, X_valid, y_valid,
                            n_iter, iter_to_load, leave_indices=None,
                            verbose=0.1, force_restart=False,
                            sklearn_refit=False, **kwargs):
        """

        :param X_valid:
        :param y_valid:
        :param n_iter:
        :param iter_to_load:
        :param leave_indices:
        :param verbose:
        :param force_restart:
        :param sklearn_refit:
        :return:
        """
        # at the entry point, the model should be trained
        assert self.trained is True

        # number of validation points
        m = X_valid.shape[0]

        # evaluate the validation losses with the full model
        losses_full = self.sess.run(
            self.losses, feed_dict=self.get_new_feed_dict(
                {'X': X_valid, 'y': y_valid}))

        if leave_indices is None:
            leave_indices = range(self.n_samples)
        losses_loo = np.zeros((len(leave_indices), m))

        # leave-one-out refitting
        # restore the trained/partially trained model
        if not sklearn_refit:
            self.load(iter_to_load=iter_to_load)
        for counter, idx_to_remove in enumerate(leave_indices):
            start_time = time.time()
            rest_indices = np.array([True]*self.n_samples, dtype=bool)
            rest_indices[idx_to_remove] = False

            leave_one_feed_dict = {
                self.X_input: self.data['X'][rest_indices, :],
                self.y_input: self.data['y'][rest_indices, :]
            }

            if sklearn_refit:
                # refit with sklearn estimator
                self.refit_with_sklearn(
                    feed_dict=leave_one_feed_dict, **kwargs)
                emp_risk_val = self.sess.run(
                    self.emp_risk,
                    feed_dict=leave_one_feed_dict)
            else:
                # refit with some more iterations
                emp_risk_val = self.refit_with_feed_dict(
                    leave_one_feed_dict, n_iter,
                    verbose=(verbose >= 1), force_restart=force_restart)

            losses_val = self.sess.run(
                self.losses, feed_dict=self.get_new_feed_dict(
                    {'X': X_valid, 'y': y_valid}))

            losses_loo[counter:counter+1, :] = (
                losses_val - losses_full).T

            dur_time = time.time() - start_time
            if 0 < verbose < 1:
                if counter % int(len(leave_indices) * verbose) == 0:
                    print('LOO Fold %d: loss = %.8f (%.3f sec)' % (
                        counter, emp_risk_val, dur_time))

            # restore checkpoint
            if not sklearn_refit:
                self.load(iter_to_load=iter_to_load)
        return losses_loo

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
