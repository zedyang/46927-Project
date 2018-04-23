# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:56:45 2018

@author: Allen Liu
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib

data = pd.read_csv('revised marketing.csv')
X=pd.get_dummies(data.iloc[:,0:8])
y=np.array(data['y'])
lmfit = LogisticRegression(C=1000000).fit(X.loc[1:], y[1:])
theta = lmfit.coef_.T
X=np.array(X.T)
X_test,y_test = X[:,0],y[0]
X,y = X[:,1:],y[1:]
sigmoid = lambda x : 1/(1 + np.exp(-x))
# Hessian
H=np.zeros([24,24])
for i in range(len(X.T)):
    H+=sigmoid(theta.T.dot(X[:,i]))*sigmoid(-theta.T.dot(X[:,i]))*np.outer(X[:,i],X[:,i])
H=H*(1/X.shape[1])

I_uploss=-y_test*y*sigmoid(-y_test*theta.T.dot(X_test))*sigmoid(-y*theta.T.dot(X))*X_test.T.dot(np.linalg.inv(H)).dot(X)
I_uploss_noH=-y_test*y*sigmoid(-y_test*theta.T.dot(X_test))*sigmoid(-y*theta.T.dot(X))*X_test.T.dot(X)
I_uploss_noT=-y_test*y*sigmoid(-y_test*theta.T.dot(X_test))*X_test.T.dot(np.linalg.inv(H)).dot(X)
I_uploss_noTH=-y_test*y*sigmoid(-y_test*theta.T.dot(X_test))*X_test.T.dot(X)
d=pd.DataFrame(np.column_stack((I_uploss.T,I_uploss_noH.T,I_uploss_noT,I_uploss_noTH,y)))

matplotlib.rcParams['figure.figsize'] = (15,8)
plt.subplot(1,3,1)
plt.scatter(-d[2],-d[0],c=d[4])
plt.xlabel('-I_up,loss (without train loss)')
plt.ylabel('-I_up,loss')
plt.subplot(1,3,2)
plt.scatter(-d[1],-d[0],c=d[4])
plt.xlabel('-I_up,loss (without H)')
plt.ylabel('-I_up,loss')
plt.subplot(1,3,3)
plt.scatter(-d[3],-d[0],c=d[4])
plt.xlabel('-I_up,loss (without train loss & H)')
plt.ylabel('-I_up,loss')
