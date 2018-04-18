import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from influence.emp_risk_optimizer import EmpiricalRiskOptimizer


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
        l2_penalty = tf.reduce_sum(
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

if __name__ == "__main__":
    df = pd.read_csv('../data/lm_10.csv')
    n = len(df)
    X = df.values[:, 0:10]
    y = df.values[:, -1].reshape(n, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for data in [X_train, X_test, y_train, y_test]:
        print(data.shape)
    n, p = X_train.shape
    tf.reset_default_graph()
    model = RegularizedRegression(
        model_name='RegularizedRegression',
        eta=0.001, C=1)
    model.fit(X_train, y_train, n_iter=10000)