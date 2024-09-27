'''
Author: your name
Date: 2021-05-15 19:41:45
LastEditTime: 2021-05-15 19:45:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/ensemble/voting_weight_classifier.py
'''
import numpy as np
import sys

class VotingWeightClassifier():
    def __init__(self,trained_models,weight):
        self.trained_models=trained_models
        self.weight=weight

    def predict(self,X_test,y_test):
        return self.weight_ensemble(X_test,y_test)

    def weight_ensemble(self,X_test,y_test):
        y_pred_all=0
        x_all=0
        for i in range(len(self.trained_models)):
            y_pred_single=self.trained_models[i].predict(X_test)
            y_pred_all=y_pred_all+y_pred_single*self.weight[i]
            x_all=x_all+self.weight[i]
        y_pred_avg=y_pred_all/x_all
        y_pred_avg[y_pred_avg<0.5]=0
        y_pred_avg[y_pred_avg>=0.5]=1
        return y_pred_avg