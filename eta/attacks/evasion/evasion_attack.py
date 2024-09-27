'''
Author: your name
Date: 2021-04-01 17:38:22
LastEditTime: 2021-08-02 08:54:11
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/attacks/evasion/evasion_attack.py
'''
import sys
import math
import numpy as np

from eta.classifiers.model_op import ModelOperation
from eta.get_model_data import model_dict
from eta.classifiers.scores import get_class_scores
from eta.ensemble.stacking_classifier import StackingClassifier
from eta.ensemble.voting_hard_classifier import VotingHardClassifier
from eta.ensemble.voting_soft_classifier import VotingSoftClassifier
from eta.ensemble.bayes_ensemble_classifier import BayesEnasembleClassifier
from eta.ensemble.voting_weight_classifier import VotingWeightClassifier
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import copy
from eta.attacks.evasion.zosgd_shap_sum import ZOSGDShapSumMethod

def evasion_dict(attack_model,feature_importance,evasion_algorithm,upper,lower):

    def get_score(p):
        score=attack_model.predict(p.reshape(1,-1))
        return -score[0][0]

    if evasion_algorithm == 'gradient':
        model=attack_model
        attack=GradientEvasionAttack(model.classifier)
        return attack 
    elif evasion_algorithm== 'zo_shap_scd':
        model=attack_model
        attack=ZOShapSCDMethod(estimator=model,feature_importance=feature_importance,eps=1,eps_step=0.005,max_iter=20)
        return attack
    elif evasion_algorithm== 'zo_shap_sgd':
        model=attack_model
        attack=ZOShapSGDMethod(estimator=model,feature_importance=feature_importance,upper=upper,lower=lower,eps=1,eps_step=0.02,max_iter=20)
        return attack


def EvasionAttack(attack_model,feature_importance,evasion_algorithm,X_test,y_test,upper,lower,mask):
    attack=evasion_dict(attack_model,feature_importance,evasion_algorithm,upper,lower)
    X_adv,X_adv_path,y_adv=attack.generate(X_test,y_test)

    def objective_function(X):
        pred=attack_model.predict(X)
        target_label=1
        max_pred=pred[:,target_label]
        inter_pred=copy.deepcopy(max_pred)

    return X_adv,X_adv_path,y_adv,objective_function


    


    
    

