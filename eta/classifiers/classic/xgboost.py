'''
Author: your name
Date: 2021-04-06 13:57:54
LastEditTime: 2021-07-10 20:14:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/classic/xgboost.py
'''
from eta.classifiers.abstract_model import AbstractModel
import xgboost
import pickle
from eta.estimators.classification.ensemble_tree import EnsembleTree

class XGBoost(AbstractModel):
    def __init__(self,input_size,output_size):
        model=xgboost.XGBClassifier()
        self.classifier=EnsembleTree(model=model,nb_features=input_size, nb_classes=output_size)
