'''
Author: your name
Date: 2021-04-06 14:32:38
LastEditTime: 2021-07-10 20:15:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/classic/catboost.py
'''
from eta.classifiers.abstract_model import AbstractModel
import catboost
from eta.estimators.classification.ensemble_tree import EnsembleTree

class CatBoost(AbstractModel):
    def __init__(self,input_size,output_size):
        model=catboost.CatBoostClassifier(logging_level='Silent')
        self.classifier=EnsembleTree(model=model,nb_features=input_size, nb_classes=output_size)