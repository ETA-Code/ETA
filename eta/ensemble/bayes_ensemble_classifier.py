'''
Author: your name
Date: 2021-04-06 19:39:31
LastEditTime: 2021-07-10 19:20:18
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/defences/ensemble/stacking_classifier.py
'''
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import shape
from eta.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
from eta.classifiers.scores import get_class_scores

import matplotlib
matplotlib.use('TkAgg')
from eta.zoopt.DE import DE
import pandas as pd
import matplotlib.pyplot as plt

class BayesEnasembleClassifier():
    def __init__(self,trained_models):
        self.trained_models=trained_models

    def bayes_pre(self,X_val,y_val,X_test,y_test):
        def get_result_0(x):
            y_pred_all=0
            x_all=0
            for i in range(len(self.trained_models)):
                y_pred_single=self.trained_models[i].predict(X_val)
                y_pred_all=y_pred_all+y_pred_single*x[i]
                x_all=x_all+x[i]
            y_pred_avg=y_pred_all/x_all

            y_pred_avg[y_pred_avg<0.5]=0
            y_pred_avg[y_pred_avg>=0.5]=1
            result=get_class_scores(y_val, y_pred_avg)
            return result[1]

        # 实例化一个bayes优化对象
        bound=[]
        keys=[]
        for i in range(len(self.trained_models)):
            bound.append([0.01,0.99])
            keys.append('x'+str(i))

        bo = BayesianOptimization(f=get_result_0,pbounds={'x':bound},random_state=7)
        
        bo.maximize(init_points=10,n_iter=20,distribute=None)
        print(bo.max['params'])
        max_x=np.array([bo.max['params'][key] for key in keys])
        weight_distribute=self.get_distribute(max_x,len(self.trained_models))
        print(weight_distribute)

        # plot_line(bo.res,keys)
        
        max_x=np.array([bo.max['params'][key] for key in keys ])
        y_pred_all=0
        x_all=0
        for i in range(len(self.trained_models)):
            y_pred_single=self.trained_models[i].predict(X_test)
            y_pred_all=y_pred_all+y_pred_single*max_x[i]
            x_all=x_all+max_x[i]
        y_pred_avg=y_pred_all/x_all
        y_pred_avg[y_pred_avg<0.5]=0
        y_pred_avg[y_pred_avg>=0.5]=1
                    
        return y_pred_avg

    def predict(self,X_val,y_val,X_test,y_test):
        return self.bayes_pre(X_val,y_val,X_test,y_test)

    def get_distribute(self,max_x,len_distribute):
        x_all=0
        for i in range(len_distribute):
            x_all=x_all+max_x[i]
        distribute=[]
        for i in range(len_distribute):
            distribute.append(max_x[i]/x_all)
        return distribute
