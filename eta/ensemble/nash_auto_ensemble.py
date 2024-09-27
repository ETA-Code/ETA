'''
Author: your name
Date: 2021-05-09 20:27:32
LastEditTime: 2021-05-19 12:02:59
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/ensemble/nash_auto_ensemble.py
'''

import numpy as np
from eta.zoopt.bayes_opt.bayesian_optimization import BayesianOptimization
from eta.classifiers.scores import get_binary_class_scores
from eta.attacks.evasion.evasion_attack import EvasionAttack,EnsembleEvasionAttack
from eta.table_save_csv import save_nash_csv

def get_distribute(max_x,len_distribute):
    x_all=0
    for i in range(len_distribute):
        x_all=x_all+max_x[i]
    distribute=[]
    for i in range(len_distribute):
        distribute.append(max_x[i]/x_all)
    return distribute


def NashAutoEnsemble(options,trained_models,X_test,y_test,mask):

    len_ensemble_models=len(trained_models)
    weight_distribute=[]
    for i in range(len_ensemble_models):
        weight_distribute.append(1.0/len_ensemble_models)
    #这组参数收敛
    # weight_distribute=[0.004122504749508993, 0.4081279702013903, 0.07186355154716799, 0.4081279702013903, 0.004122504749508993, 0.004122504749508993, 0.09951299380152458]
        
    evasion_distribute=[0.25, 0.25, 0.25, 0.25]
    evasion_algorithms=options.evasion_algorithms

    X_test_1=X_test[y_test==1]
    y_test_1=y_test[y_test==1]

    X_test_0=X_test[y_test==0]
    y_test_0=y_test[y_test==0]

    def get_ensemble_result(weight_distribute,X_test_adv,y_test_adv):
        y_pred_all=0
        x_all=0
        for i in range(len(trained_models)):
            y_pred_single=trained_models[i].predict(X_test_adv)
            y_pred_all=y_pred_all+y_pred_single*weight_distribute[i]
            x_all=x_all+weight_distribute[i]
        y_pred_avg=y_pred_all/x_all

        y_pred_avg[y_pred_avg<0.5]=0
        y_pred_avg[y_pred_avg>=0.5]=1
        result=get_binary_class_scores(y_test_adv, y_pred_avg)
        return result
    
    def get_evasion_success_rate(x):
        # print(weight_distribute)
        evasion_distribute=get_distribute(x,len(evasion_algorithms))
        X_adv,y_adv=EnsembleEvasionAttack(options,evasion_distribute,X_test_1,y_test_1,mask)
        X_test_adv = np.append(X_test_0, X_adv, axis=0)
        y_test_adv = np.append(y_test_0, y_adv, axis=0)
        result=get_ensemble_result(weight_distribute,X_test_adv,y_test_adv)
        return 1-result[3]

    def get_ensemble_detector_success_rate(x):
        # print(evasion_distribute)
        weight_distribute=get_distribute(x,len_ensemble_models)
        X_adv,y_adv=EnsembleEvasionAttack(options,evasion_distribute,X_test_1,y_test_1,mask)
        X_test_adv = np.append(X_test_0, X_adv, axis=0)
        y_test_adv = np.append(y_test_0, y_adv, axis=0)
        result=get_ensemble_result(weight_distribute,X_test_adv,y_test_adv)
        return result[3]

    # 实例化两个bayes优化对象
    bound_evasion=[]
    keys_evasion=[]

    bound_ensemble=[]
    keys_ensemble=[]

    for i in range(len(evasion_distribute)):
        bound_evasion.append([0.01,0.99])
        keys_evasion.append('x'+str(i))

    for i in range(len(weight_distribute)):
        bound_ensemble.append([0.01,0.99])
        keys_ensemble.append('x'+str(i))


        
    for i in range(50):

        bo_ensemble = BayesianOptimization(
        get_ensemble_detector_success_rate,
        {'x':bound_ensemble}
        )
        bo_evasion = BayesianOptimization(
        get_evasion_success_rate,
        {'x':bound_evasion}
        )

        bo_evasion.maximize(init_points=5,n_iter=5,distribute=evasion_distribute)
        max_evasion_x=np.array([bo_evasion.max['params'][key] for key in keys_evasion ])
        target=bo_evasion.max['target']
        evasion_distribute=get_distribute(max_evasion_x,len(evasion_algorithms))
        print('evasion')
        print(evasion_distribute)
        print('target:'+str(target))
        new_keys_evasion=keys_evasion.copy()
        save_nash_csv(i,bo_evasion.res,new_keys_evasion,options.datasets,'evasion_ensemble_nash')

        bo_ensemble.maximize(init_points=5,n_iter=5,distribute=weight_distribute)
        max_adv_x=np.array([bo_ensemble.max['params'][key] for key in keys_ensemble ])
        target=bo_ensemble.max['target']
        weight_distribute=get_distribute(max_adv_x,len_ensemble_models)
        print('weight')
        print(weight_distribute)
        print('target:'+str(target))
        new_keys_ensemble=keys_ensemble.copy()
        save_nash_csv(i,bo_ensemble.res,new_keys_ensemble,options.datasets,'ensemble_nash')