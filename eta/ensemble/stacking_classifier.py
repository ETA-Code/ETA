'''
Author: your name
Date: 2021-04-06 19:39:31
LastEditTime: 2021-05-14 09:23:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/defences/ensemble/stacking_classifier.py
'''
import numpy as np
from eta.classifiers.classic.logistic_regression import LogisticRegression
class StackingClassifier():
    def __init__(self,trained_models):
        self.trained_models=trained_models
        self.model=LogisticRegression()

    def train(self,X_val,y_val):
        X_new_train=np.zeros((len(y_val),len(self.trained_models)))
        for i in range(len(self.trained_models)):
            y_pred=self.trained_models[i].predict(X_val)
            X_new_train[:,i]=y_pred
        self.model.train(X_new_train,y_val)

    def predict(self,X_test,y_test):
        
        X_new_test=np.zeros((len(y_test),len(self.trained_models)))
        for i in range(len(self.trained_models)):
            y_pred=self.trained_models[i].predict(X_test)
            X_new_test[:,i]=y_pred
        return self.model.predict(X_new_test)


