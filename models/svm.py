import numpy as np
import tensorflow as tf
import pandas as pd
from influence.emp_risk_optimizer import EmpiricalRiskOptimizer
from sklearn.model_selection import train_test_split


class SupportVectorMachine(EmpiricalRiskOptimizer):

    def __init__(self, **kwargs):
        super(SupportVectorMachine, self).__init__(**kwargs)
        self.p1 = None
        self.C = kwargs.pop('C')
        self.t = kwargs.pop('t')
        self.hyperparams = ['C', 't']

    def get_all_params(self):
        # number of parameters.
        self.n_params = self.p

        # build the flat params container
        all_params = tf.get_variable(
            'flat_params', (self.n_params, 1),
            initializer=tf.zeros_initializer)

        # build the params dict
        self.all_params_dict[
            'beta'] = all_params[0:self.n_params, :]

        return all_params

    def get_emp_risk(self):
        # params dict split the params into groups, for convenience
        b = self.all_params_dict['beta']

        # x_dot_beta
        x_dot_beta = tf.matmul(
            self.X_input[:, 0:self.p1], b, name='x_dot_beta')

        # L2 loss
        L2_loss = tf.reduce_sum(tf.multiply(b, b), name='L2_loss')

        # hinge loss
        hinge_loss = tf.subtract(
            tf.constant(1.0),
            tf.multiply(
                self.y_input,
                x_dot_beta),
            name='hinge_loss')

        # smooth hinge
        t = tf.constant(self.t, name='t')
        smooth_hinge = tf.multiply(
            t, tf.log(tf.constant(1.0) + tf.exp(
                (tf.constant(1.0) - tf.reduce_sum(
                    tf.multiply(self.y_input, x_dot_beta))) / t)),
            name='smooth_hinge')

        # must return individual losses and total empirical risk
        losses = tf.add(
            smooth_hinge,
            tf.multiply(self.C, L2_loss),
            name='total_loss')
        emp_risk = tf.reduce_mean(losses, name='emp_risk')
        return losses, emp_risk

    def predict(self, X_test):
        params = self.get_eval(items=['params'])
        beta = params['beta']
        return np.sign(X_test.dot(beta))

if __name__ == '__main__':
    df = pd.read_csv('../data/svm_10.csv')
    n = len(df)
    X = df.values[:, 0:10]
    y = df.values[:, -1].reshape(n, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for data in [X_train, X_test, y_train, y_test]:
        print(data.shape)
    n, p = X_train.shape
    model = SupportVectorMachine(
        model_name='SVM',
        eta=0.01,
        C=0.1,
        t=0.1
    )
    model.fit(X_train, y_train, n_iter=10000)
    y_hat = model.predict(X_train)
    print(np.mean(y_train == y_hat))
