# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:49:27 2018

@author: Sandeep
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


#getting data
np.random.seed(12)
num_observations=5000

x1=np.random.multivariate_normal([0,0],[[1,.75],[.75,1]],num_observations)
x2=np.random.multivariate_normal([1,4],[[1,.75],[.75,1]],num_observations)

simulated_separableish_features=np.vstack((x1,x2)).astype(np.float32)
simulated_labels=np.hstack((np.zeros(num_observations),
                            np.ones(num_observations)))

plt.figure(figsize=(12,8))
plt.scatter(simulated_separableish_features[:,0],simulated_separableish_features[:,1],
            c=simulated_labels,alpha=.4)

#Logistic regression
clf=LogisticRegression(fit_intercept=True, C=1e15)
clf.fit(simulated_separableish_features,simulated_labels)

#accuracy
pred=clf.score(simulated_separableish_features, simulated_labels)
print (clf.intercept_,clf.coef_)
print ('Accuracy from sk-learn: {0}'.format(pred))

#plotting results
plt.figure(figsize = (12, 8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = pred==simulated_labels - 1, alpha = .8, s = 50)