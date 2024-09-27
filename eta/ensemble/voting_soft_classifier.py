'''
Author: your name
Date: 2021-04-06 15:40:01
LastEditTime: 2021-04-22 12:40:35
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/defences/ensemble/voting_classifier.py
'''


import numpy as np
import sys

class VotingSoftClassifier():
    def __init__(self,trained_models):
        self.trained_models=trained_models

    def predict(self,X_test,y_test):
        return self.soft_ensemble(X_test,y_test)

    def soft_ensemble(self,X_test,y_test):
        y_pred_all=np.zeros(X_test.shape[0])
        for i in range(0,len(self.trained_models)):
            y_pred = self.trained_models[i].predict_proba(X_test)[:,1]
            y_pred_all=y_pred_all+y_pred/float(len(self.trained_models))

        y_pred_all[y_pred_all<0.5]=0
        y_pred_all[y_pred_all>=0.5]=1
        return y_pred_all