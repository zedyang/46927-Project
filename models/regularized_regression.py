import numpy as np
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError
import tensorflow as tf
from influence.emp_risk_optimizer import EmpiricalRiskOptimizer


class RegularizedRegression(EmpiricalRiskOptimizer):
    """
    Regularized regression
    """

    def __init__(self, **kwargs):
        super(RegularizedRegression, self).__init__(**kwargs)
        self.C = kwargs.pop('C')
        self.hyperparams += ['C']

    def get_all_params(self):
        """

        :return:
        """
        # number of parameters.
        self.n_params = self.n_features

        # build the flat params container
        all_params = tf.get_variable(
            'flat_params', (self.n_params, 1),
            dtype=tf.float32,
            initializer=tf.zeros_initializer)

        # build the params dict
        self.all_params_dict['beta'] = all_params

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
            beta**2/self.n_samples, name='l2_penalty')

        losses = tf.add(
            tf.square(self.y_input - y_hat),
            l2_penalty, name='losses')

        emp_risk = tf.reduce_mean(losses, name='emp_risk')
        return losses, emp_risk

    def fit_with_sklearn(self, feed_dict=None, **kwargs):
        if feed_dict is None:
            feed_dict = {'X': self.data['X'], 'y': self.data['y']}

        if not self.trained:
            self.fit(
                feed_dict['X'], feed_dict['y'],
                n_iter=None, lazy_fit=True)

        self.sklearn_clf = Ridge(
            alpha=self.C,
            fit_intercept=False,
            tol=1e-8,
            max_iter=None
        )
        self.sklearn_clf.fit(
            feed_dict['X'],
            feed_dict['y'],
            **kwargs)

        # update params
        beta = self.sklearn_clf.coef_
        self.sess.run(
            self.all_params_assign,
            feed_dict={self.all_params_input: beta.reshape(
                (self.n_params, 1))}
        )
        self.trained = True
        self.show_eval()
        return self

    def refit_with_sklearn(self, feed_dict, **kwargs):
        assert self.sklearn_clf is not None
        n, _ = feed_dict[self.X_input].shape
        self.sklearn_clf.fit(
            feed_dict[self.X_input],
            feed_dict[self.y_input],
            **kwargs)

        # update params
        # sklearn_clf.coef_ returns (n_feature * n_classes)
        beta = self.sklearn_clf.coef_
        self.sess.run(
            self.all_params_assign,
            feed_dict={self.all_params_input: beta.reshape(
                (self.n_params, 1))}
        )
        self.trained = True
        # self.show_eval()

    def predict(self, X_test):
        params = self.get_eval(items=['params'])
        beta = params['beta']
        return X_test.dot(beta)