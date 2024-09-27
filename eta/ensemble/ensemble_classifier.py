'''
Author: your name
Date: 2021-04-07 09:41:04
LastEditTime: 2021-07-10 19:20:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/ensemble/ensemble_classifier.py
'''
from eta.ensemble.stacking_classifier import StackingClassifier
from eta.ensemble.voting_hard_classifier import VotingHardClassifier
from eta.ensemble.voting_soft_classifier import VotingSoftClassifier
from eta.ensemble.bayes_ensemble_classifier import BayesEnasembleClassifier
from eta.ensemble.voting_weight_classifier import VotingWeightClassifier
import numpy as np
from tabulate import tabulate

def EnsembleClassifier(X_val,y_val,X_test,y_test,trained_models):
        
    ensemble_array=['hard','soft','stacking','bayes_ensemble']
    y_pred_array=[]
    # hard voting
    vc_hard=VotingHardClassifier(trained_models)
    y_pred=vc_hard.predict(X_test,y_test)
    y_pred_array.append(y_pred)

    # soft voting
    vc_soft=VotingSoftClassifier(trained_models)
    y_pred=vc_soft.predict(X_test,y_test)
    y_pred_array.append(y_pred)

    # stacking
    stacking=StackingClassifier(trained_models)
    stacking.train(X_val,y_val)
    y_pred=stacking.predict(X_test,y_test)
    y_pred_array.append(y_pred)

    #bayes_ensemble
    bayes_ens=BayesEnasembleClassifier(trained_models)
    y_pred=bayes_ens.predict(X_val,y_val,X_test,y_test)
    y_pred_array.append(y_pred)

    #weight_ensemble
    # weight_distribute=[]
    # len_ensemble_models=len(trained_models)
    # for i in range(len_ensemble_models):
    #     weight_distribute.append(1.0/len_ensemble_models)
    # vc_weight=VotingWeightClassifier(trained_models,weight_distribute)
    # y_pred=vc_weight.predict(X_test,y_test)
    # y_pred_array.append(y_pred)

    
    return ensemble_array,y_test,y_pred_array


    