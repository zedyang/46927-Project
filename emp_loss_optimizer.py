import abc
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin

import tensorflow as tf
from tensorflow.python.ops import array_ops


class EmpiricalLossOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        pass

    def __repr__(self):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError
