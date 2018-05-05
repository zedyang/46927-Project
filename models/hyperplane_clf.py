import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError
import tensorflow as tf
from influence.emp_risk_optimizer import EmpiricalRiskOptimizer


class MulticlassLogisticRegression(EmpiricalRiskOptimizer):
    """
    Multiclass Logistic Regression
    """
    def __init__(self, **kwargs):
        super(MulticlassLogisticRegression, self).__init__(**kwargs)
        self.C = kwargs.pop('C')
        self.hyperparams += ['C']
        self.y_prob = None

    def get_all_params(self):
        """
        :return:
        """
        # number of parameters: Weights W is (p * g) matrix
        self.n_params = self.n_features*self.n_classes

        # build the flat params container
        all_params = tf.get_variable(
            name='flat_params',
            shape=(self.n_params, 1),
            initializer=tf.zeros_initializer)

        # build the params dict
        self.all_params_dict['W'] = tf.reshape(
            all_params, [self.n_features, self.n_classes])

        return all_params

    def get_emp_risk(self):
        W = self.all_params_dict['W']

        # logits
        logits = tf.matmul(self.X_input, W)
        self.y_prob = tf.nn.softmax(logits)

        # L2 regularization
        l2_penalty = self.C * tf.reduce_sum(
            W ** 2 / self.n_samples, name='l2_penalty')

        # losses
        losses = tf.add(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.y_input, logits=logits,
                name='cross_entropy_loss'),
            l2_penalty)
        emp_risk = tf.reduce_mean(losses, name='emp_risk')
        return losses, emp_risk

    def fit_with_sklearn(self, feed_dict=None, **kwargs):
        # fit with sklearn model
        if feed_dict is None:
            feed_dict = {'X': self.data['X'], 'y': self.data['y']}

        if not self.trained:
            self.fit(
                feed_dict['X'], feed_dict['y'],
                n_iter=None, lazy_fit=True)

        self.sklearn_clf = LogisticRegression(
            C=1/self.C,
            fit_intercept=False,
            tol=1e-8,
            solver='lbfgs',
            warm_start=True,
            max_iter=1000
        )
        self.sklearn_clf.fit(
            feed_dict['X'],
            # feed_dict['y'] is one-hot encoded
            np.argmax(feed_dict['y'], axis=1),
            **kwargs)

        # update params
        W = self.sklearn_clf.coef_.T
        self.sess.run(
            self.all_params_assign,
            feed_dict={self.all_params_input: W.reshape(
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
            # feed_dict['y'] is one-hot encoded
            np.argmax(feed_dict[self.y_input], axis=1),
            **kwargs)

        # update params
        # sklearn_clf.coef_ returns (n_feature * n_classes)
        W = self.sklearn_clf.coef_.T
        self.sess.run(
            self.all_params_assign,
            feed_dict={self.all_params_input: W.reshape(
                (self.n_params, 1))}
        )
        self.trained = True
        # self.show_eval()

    def predict(self, X_test):
        y_prob_val = self.sess.run(
            self.y_prob,
            feed_dict={self.X_input: X_test}
        )
        y_pred_cls = np.argmax(y_prob_val, axis=1)
        return y_pred_cls


class BinaryLogisticRegression(EmpiricalRiskOptimizer):
    """
    Binary Logistic Regression
    """

    def __init__(self, **kwargs):
        super(BinaryLogisticRegression, self).__init__(**kwargs)
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
            name='flat_params',
            shape=(self.n_params, 1),
            initializer=tf.random_normal_initializer)

        # build the params dict
        self.all_params_dict[
            'beta'] = all_params

        return all_params

    def get_emp_risk(self):
        """
        The original formulation of cross-entropy loss is

          x - x * z + log(1 + exp(-x))

        It is highly susceptible to overflow in exp(-x) when x is negative,
        which causes the losses and empirical risk to blow up. The loss can
        be reformulate as:

          x - x * z + log(1 + exp(-x))
        = log(exp(x)) - x * z + log(1 + exp(-x))
        = - x * z + log(1 + exp(x))

        It's essential to implement cross entropy loss as
        tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=z),
        which actually uses the following formulation:

        max(x, 0) - x * z + log(1 + exp(-abs(x)))

        to avoid overflow in exp(-x) when x < 0. Same applies to softmax.
        :return:
        """
        beta = self.all_params_dict['beta']

        # logits
        logits = tf.matmul(self.X_input, beta)

        # L2 regularization
        l2_penalty = self.C*tf.reduce_sum(
            beta**2/self.n_samples, name='l2_penalty')

        # losses
        losses = tf.add(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.y_input, logits=logits,
                name='cross_entropy_loss'),
            l2_penalty)
        emp_risk = tf.reduce_mean(losses, name='emp_risk')
        return losses, emp_risk

    def fit_with_sklearn(self, feed_dict=None, **kwargs):
        # fit with sklearn model
        if feed_dict is None:
            feed_dict = {'X': self.data['X'], 'y': self.data['y']}

        if not self.trained:
            self.fit(
                feed_dict['X'], feed_dict['y'],
                n_iter=None, lazy_fit=True)

        self.sklearn_clf = LogisticRegression(
            C=1/self.C,
            fit_intercept=False,
            tol=1e-10,
            solver='lbfgs',
            warm_start=True,
            max_iter=2000
        )
        self.sklearn_clf.fit(
            feed_dict['X'],
            feed_dict['y'].astype('int').reshape(self.n_samples,),
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
            feed_dict[self.y_input].astype(
                'int').reshape(n,),
            **kwargs)

        # update params
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
        logits = 1 / (1 + np.exp(-X_test.dot(beta)))
        return 1*(logits > 0.5)


class SmoothedSupportVector(EmpiricalRiskOptimizer):

    def __init__(self, **kwargs):
        super(SmoothedSupportVector, self).__init__(**kwargs)
        self.C = kwargs.pop('C')
        self.t = kwargs.pop('t')
        self.hyperparams += ['C', 't']

    @staticmethod
    def smooth_hinge_loss(x, t):
        """

        :param x:
        :param t:
        :return:
        """

        if t != 0:
            exponents = (1-x)/t
            exponents_truncate = tf.maximum(
                exponents, tf.zeros_like(exponents))
            return tf.multiply(t, exponents_truncate + tf.log(
                tf.exp(exponents-exponents_truncate) +
                tf.exp(tf.zeros_like(exponents)-exponents_truncate)
            ), name='smooth_hinge_loss')
        else:
            return tf.maximum(tf.constant(
                0, dtype=tf.float32), 1-x, name='smooth_hinge_loss')
        
    def get_all_params(self):
        # number of parameters.
        self.n_params = self.n_features + 1
        # fit intercept to match sklearn.SVC

        # build the flat params container
        all_params = tf.get_variable(
            'flat_params', (self.n_params, 1), dtype=tf.float32,
            initializer=tf.zeros_initializer)

        # build the params dict
        self.all_params_dict['beta'] = \
            all_params[0:self.n_features, :]
        self.all_params_dict['intercept'] = \
            all_params[self.n_features:self.n_params, :]

        return all_params

    def get_emp_risk(self):
        # params dict split the params into groups, for convenience
        beta = self.all_params_dict['beta']
        beta_0 = self.all_params_dict['intercept']

        # unsigned inner product
        logits = tf.matmul(self.X_input, beta, name='logits')
        margin = tf.multiply(
            tf.cast(self.y_input, tf.float32),
            logits+beta_0, name='margin')

        # L2 regularization
        l2_penalty = self.C*tf.reduce_sum(
            beta**2/self.n_samples, name='l2_penalty')

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

    def fit_with_sklearn(self, feed_dict=None, **kwargs):
        # fit with sklearn model
        if feed_dict is None:
            feed_dict = {'X': self.data['X'], 'y': self.data['y']}

        if not self.trained:
            self.fit(
                feed_dict['X'], feed_dict['y'],
                n_iter=None, lazy_fit=True)

        tol, max_iter = 1e-7, 2000
        if 'tol' in kwargs:
            tol = kwargs.pop('tol')
        if 'max_iter' in kwargs:
            max_iter = kwargs.pop('max_iter')

        self.sklearn_clf = SVC(
            C=self.C,
            kernel='linear',
            tol=tol,
            max_iter=max_iter
        )
        self.sklearn_clf.fit(
            feed_dict['X'],
            feed_dict['y'].astype('int').reshape(self.n_samples,),
            **kwargs)

        # update params
        all_params = np.concatenate([
            self.sklearn_clf.coef_.reshape(self.n_features,),
            self.sklearn_clf.intercept_])
        self.sess.run(
            self.all_params_assign,
            feed_dict={self.all_params_input: all_params.reshape(
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
            feed_dict[self.y_input].astype(
                'int').reshape(n,),
            **kwargs)

        # update params
        all_params = np.concatenate([
            self.sklearn_clf.coef_.reshape(self.n_features,),
            self.sklearn_clf.intercept_])
        self.sess.run(
            self.all_params_assign,
            feed_dict={self.all_params_input: all_params.reshape(
                (self.n_params, 1))}
        )
        self.trained = True
        # self.show_eval()

    def predict(self, X_test):
        params = self.get_eval(items=['params'])
        beta = params['beta']
        beta_0 = params['intercept']
        return np.sign(X_test.dot(beta)+beta_0)
