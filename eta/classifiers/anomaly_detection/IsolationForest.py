'''
Author: your name
Date: 2021-07-15 14:54:43
LastEditTime: 2021-07-20 19:18:14
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/anomaly_detection/IsolationForest.py
'''
from eta.classifiers.abstract_model import AbstractModel
from sklearn import ensemble
import numpy as np
from eta.estimators.classification.anomaly_classifier import AnomalyClassifeir
class IsolationForest(AbstractModel):

    def __init__(self,input_size,output_size):
        model=ensemble.IsolationForest(n_estimators=100, max_samples=256, contamination=0.1, random_state=42)
        # kitsune-botnet
        threshold=0
        # kitsune-ddos
        # threshold=0.08
        # cicids2017-botnet
        # threshold=0.134
        # cicids2017-ddos
        # threshold=0.13
        self.classifier=AnomalyClassifeir(model=model,nb_features=input_size, nb_classes=output_size,threshold=threshold)