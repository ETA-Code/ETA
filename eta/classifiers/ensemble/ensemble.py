'''
Author: your name
Date: 2021-07-12 09:59:00
LastEditTime: 2021-08-02 08:52:34
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/ensemble/ensemble.py
'''
'''
Author: your name
Date: 2021-07-12 09:59:00
LastEditTime: 2021-07-28 09:32:47
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/eta/classifiers/ensemble/ensemble.py
'''
from eta.classifiers.classic.decision_tree import DecisionTree
from eta.classifiers.classic.k_nearest_neighbours import KNearestNeighbours
from eta.classifiers.classic.logistic_regression import LogisticRegression
from eta.classifiers.classic.random_forest import RandomForest
from eta.classifiers.classic.support_vector_machine import SupportVectorMachine
from eta.classifiers.classic.naive_bayes import NaiveBayes
from eta.classifiers.classic.xgboost import XGBoost
from eta.classifiers.classic.lightgbm import LightGBM
from eta.classifiers.classic.catboost import CatBoost
from eta.classifiers.classic.deepforest import DeepForest
from eta.classifiers.classic.hidden_markov_model import HMM
from eta.classifiers.torch.mlp import MLPTorch
from eta.classifiers.torch.cnn import CNNTorch
from eta.classifiers.keras.mlp import MLPKeras
from eta.classifiers.abstract_model import AbstractModel
import lightgbm
from eta.estimators.classification.ensemble_tree import EnsembleTree
from eta.estimators.classification.soft_ensemble import SoftEnsemble
from eta.estimators.classification.origin_ensemble import OriginEnsemble
from eta.classifiers.anomaly_detection.IsolationForest import IsolationForest

def model_dict(algorithm,input_size,output_size):
    if algorithm=='lr' :
        model=LogisticRegression()
    elif algorithm=='knn':
        model=KNearestNeighbours()
    elif algorithm=='dt':
        model=DecisionTree()
    elif algorithm=='nb':
        model=NaiveBayes()
    elif algorithm=='svm':
        model=SupportVectorMachine()
    elif algorithm=='hmm':
        model=HMM()
    elif algorithm=='rf':
        model=RandomForest()
    elif algorithm=='xgboost':
        model=XGBoost(input_size=input_size,output_size=output_size)
    elif algorithm=='if':
        model=IsolationForest(input_size=input_size,output_size=2)
    elif algorithm=='lightgbm':
        model=LightGBM(input_size=input_size,output_size=output_size)
    elif algorithm=='catboost':
        model=CatBoost(input_size=input_size,output_size=output_size)
    elif algorithm=='deepforest':
        model=DeepForest(input_size=input_size,output_size=output_size)
    elif algorithm=='mlp_torch':
        model=MLPTorch(input_size=input_size,output_size=output_size)
    elif algorithm=='cnn_torch':
        model=CNNTorch(input_size=input_size,output_size=output_size)
    elif algorithm=='mlp_keras':
        model=MLPKeras(input_size=input_size,output_size=output_size)
    else:
        raise Exception(f'"{algorithm}" is not a valid choice of algorithm.')
    return model

class SoftEnsembleModel(AbstractModel):
    def __init__(self,input_size,output_size):
        models_name=['lr','dt','svm','mlp_torch']
        models=[]
        for i in range(len(models_name)):
            model=model_dict(models_name[i],input_size,output_size)
            models.append(model)
        self.classifier=SoftEnsemble(model=models,nb_features=input_size, nb_classes=output_size)

class OriginEnsembleModel(AbstractModel):
    def __init__(self,input_size,output_size):
        self.models_name=['mlp_torch','lr']
        models=[]
        for i in range(len(self.models_name)):
            model=model_dict(self.models_name[i],input_size,output_size)
            models.append(model)
        self.classifier=OriginEnsemble(model=models,nb_features=input_size, nb_classes=output_size)
