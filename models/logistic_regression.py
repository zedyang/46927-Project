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
       
        #loss
        losses = - self.y_input * tf.log(logits) - (1.0 - self.y_input) * tf.log(1.0 - logits)
        emp_risk = tf.reduce_mean(losses,name='emp_risk')
        return losses, emp_risk

    def predict(self, X_test):
        params = self.get_eval(items=['params'])
        beta = params['beta']
        logits = tf.sigmoid(tf.matmul(X_test, beta))
        return tf.to_double(tf.greater_equal(logits, 0.5))

