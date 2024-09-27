'''
Author: your name
Date: 2021-07-15 15:03:35
LastEditTime: 2021-07-20 19:44:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/anomaly_detection/diff_rf.py
'''
from eta.classifiers.abstract_model import AbstractModel
from sklearn import ensemble
import numpy as np
from eta.classifiers.anomaly_detection.diff_rf_packet.diff_RF import DiFF_TreeEnsemble
from eta.estimators.classification.anomaly_classifier import AnomalyClassifeir
class DIFFRF(AbstractModel):

    def __init__(self,input_size,output_size):
        model=DiFF_TreeEnsemble(n_trees=256)
        # kitsune-botnet
        threshold=0.1
        # kitsune-ddos
        # threshold=0.08
        # cicids2017-botnet
        # threshold=0.9
        # cicids2017-ddos
        # threshold=0.1
        self.classifier=AnomalyClassifeir(model=model,nb_features=input_size, nb_classes=output_size,threshold=threshold)