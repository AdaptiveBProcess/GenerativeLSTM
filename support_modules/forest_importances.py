"""
=========================================
Feature importances with forests of trees
=========================================

This examples shows the use of forests of trees to evaluate the importance of
features on an artificial classification task. The red bars are the feature
importances of the forest, along with their inter-trees variability.

As expected, the plot suggests that 3 features are informative, while the
remaining are not.
"""
#print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


from sklearn.model_selection import train_test_split
from support_modules import nn_support as nsup

import pandas as pd



def calculate_importances(df, keep_cols):
    X = df[df.columns.difference(keep_cols)]
    y = df['ac_index']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)
    

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=500, random_state=0)
    
    print(X_train.dtypes)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    
    imp_table = pd.DataFrame([{'feature':x,'importance':y}
                            for x, y in list(zip(X_train.columns, importances))])
    print(imp_table.sort_values(by=['importance'], ascending=False))
        
    
    # linear regression
    X = df[df.columns.difference(keep_cols)]
    y = nsup.scale_feature(df.copy(), 'dur', 'max', True)['dur_norm']

#    X = df[df.columns.difference(keep_cols)]
#    y = df['ac_index']
    
    # create a fitted model with all three features
    lm1 = smf.ols(formula='dur_norm ~ ev_rd_p + ev_acc_t_norm', data=pd.concat([X, y], axis=1, sort=False)).fit()
#    lm1 = smf.ols(formula='ac_index ~ city5_norm + ev_acc_t_norm + ev_et_t_norm + snap2_norm', data=pd.concat([X, y], axis=1, sort=False)).fit()
    
    # print the coefficients
    print(lm1.summary())


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)
    
    lm2 = LinearRegression()
    lm2.fit(X_train, y_train)
    
    for x, y in list(zip(X_train.columns, lm2.coef_)):
        print(x, y, sep=' ')
    
    
#     Print the feature ranking
    print("Feature ranking:", importances, ' ')