'''
Author: your name
Date: 2021-03-24 19:23:21
LastEditTime: 2021-07-27 17:16:05
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/get_model_data.py
'''
# from eta.utils import load_mnist
from collections import defaultdict
import sys
import numpy as np

from eta.datasets import load_cicids2017
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from eta.classifiers.model_op import ModelOperation
from sklearn.model_selection import train_test_split
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
from eta.classifiers.torch.lr import LRTorch
from eta.classifiers.torch.cnn import CNNTorch,CNN2Torch
from eta.classifiers.anomaly_detection.KitNET import KitNET
from eta.classifiers.anomaly_detection.IsolationForest import IsolationForest
from eta.classifiers.anomaly_detection.diff_rf import DIFFRF
from eta.classifiers.ensemble.ensemble import SoftEnsembleModel
from eta.classifiers.ensemble.ensemble import OriginEnsembleModel

import pandas as pd
from pandas.core.frame import DataFrame

from eta.classifiers.scores import get_class_scores
from sklearn.metrics import confusion_matrix
from eta.ensemble.ensemble_classifier import EnsembleClassifier
from tabulate import tabulate
import configargparse
import yaml


def parse_arguments(arguments):
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('--attack_models',required=False,default='lr')
    parser.add('--algorithms', required=False, default=['lr'],choices=['lr', 'svm','dt','rf','xgboost','lightgbm','catboost','deepforest','knn','hmm','mlp_keras','mlp_torch','cnn_torch','kitnet','if','diff-rf','soft_ensemble','origin_ensemble'])
    parser.add('--datasets',required=False, default='cicids2017',choices=['cicids2017'])
    parser.add('--evasion_algorithm',required=False,default='zosgd_shap_sum',choices=['gradient','zosgd_sum','zosgd_shap_sum'])
    parser.add('--evasion_distribute',required=False,default=[0.5,0.3,0.1,0.1])
    options = parser.parse_args(arguments)
    return options
    
def model_dict(algorithm,n_features,out_size):
    if algorithm=='lr' :
        return LogisticRegression()
    elif algorithm=='knn':
        return KNearestNeighbours()
    elif algorithm=='dt':
        return DecisionTree()
    elif algorithm=='nb':
        return NaiveBayes()
    elif algorithm=='svm':
        return SupportVectorMachine()
    elif algorithm=='hmm':
        return HMM()
    elif algorithm=='rf':
        return RandomForest()
    elif algorithm=='xgboost':
        return XGBoost(input_size=n_features,output_size=out_size)
    elif algorithm=='lightgbm':
        return LightGBM(input_size=n_features,output_size=out_size)
    elif algorithm=='catboost':
        return CatBoost(input_size=n_features,output_size=out_size)
    elif algorithm=='deepforest':
        return DeepForest(input_size=n_features,output_size=out_size)
    elif algorithm=='mlp_torch':
        return MLPTorch(input_size=n_features,output_size=out_size)
    elif algorithm=='lr_torch':
        return LRTorch(input_size=n_features,output_size=out_size)
    elif algorithm=='cnn_torch':
        return CNNTorch(input_size=n_features,output_size=out_size)
    elif algorithm=='cnn2_torch':
        return CNN2Torch(input_size=n_features,output_size=out_size)
    elif algorithm=='mlp_keras':
        return MLPKeras(input_size=n_features,output_size=out_size)
    elif algorithm=='kitnet':
        return KitNET(input_size=n_features,output_size=out_size)
    elif algorithm=='if':
        return IsolationForest(input_size=n_features,output_size=out_size)
    elif algorithm=='diff-rf':
        return DIFFRF(input_size=n_features,output_size=out_size)
    elif algorithm=='soft_ensemble':
        return SoftEnsembleModel(input_size=n_features,output_size=out_size)
    elif algorithm=='origin_ensemble':
        return OriginEnsembleModel(input_size=n_features,output_size=out_size)
    else:
        raise Exception(f'"{algorithm}" is not a valid choice of algorithm.')

def get_model(options,n_features,out_size):
    algorithms_name = options.algorithms
    models_array=[]
    for i in range(len(algorithms_name)):
        models_array.append(model_dict(algorithms_name[i],n_features,out_size))
    return models_array,algorithms_name

    

def get_datasets(options):
    datasets=options.datasets
    if datasets=='mnist':
        X_train,y_train,X_val,y_val,X_test,y_test,mask=load_mnist_zhs()
        return X_train,y_train,X_val,y_val,X_test,y_test,mask
    if datasets=='donut':
        X,y,mask=load_donut()
    elif datasets=='blob':
        X,y,mask=load_blob()
    elif datasets=='digits':
        X,y,mask=load_digits_zhs()
    elif datasets=='nslkdd':
        X,y,mask=load_nslkdd()
    elif datasets=='cicids2017':
        X,y,mask=load_cicids2017()
    elif datasets=='kitsune':
        X,y,mask=load_kitsune()
    else:
        raise Exception(f'"{datasets}" is not a valid choice of dataset.')
    mm=MinMaxScaler()
    X=mm.fit_transform(X)


    X_train,y_train,X_val,y_val,X_test,y_test=train_val_test_split(X,y,0.6,0.2,0.2)
    

    if type(y_train) is not np.ndarray:
        y_train, y_test,y_val=y_train.values, y_test.values,y_val.values
    X_train=X_train.astype(np.float64)
    X_test=X_test.astype(np.float64)
    X_val=X_val.astype(np.float64)
    y_train=y_train.astype(np.int64)
    y_test=y_test.astype(np.int64)
    y_val=y_val.astype(np.int64)
        
    return X_train,y_train,X_val,y_val,X_test,y_test,mask


def attack_models_train(options,if_adv,X_train,y_train):
    datasets_name=options.datasets
    attack_models_name=options.attack_models
    out_size=len(np.unique(y_train))
    model=model_dict(attack_models_name,X_train.shape[1],out_size)
    model_=ModelOperation()
    trained_model=model_.train(attack_models_name,datasets_name,model,if_adv,X_train,y_train)
    return trained_model

def models_train(options,if_adv,X_train,y_train):
    out_size=len(np.unique(y_train))
    models_array,algorithms_name=get_model(options,X_train.shape[1],out_size)
    datasets_name=options.datasets
    trained_models_array=[]
    for i in range(0,len(models_array)):
        model_=ModelOperation()
        trained_model=model_.train(algorithms_name[i],datasets_name,models_array[i],if_adv,X_train,y_train)
        trained_models_array.append(trained_model)
    return trained_models_array

def models_predict(trained_models,X_test,y_test):
    if len(np.unique(y_test))<=2:
        len_y=2
    else:
        len_y=len(np.unique(y_test))
    y_pred_arr=np.zeros((len(trained_models),X_test.shape[0],len_y))
    for i in range(len(trained_models)):
        y_pred=trained_models[i].predict(X_test)
        y_pred_arr[i]=y_pred
    return y_test,y_pred_arr

def models_predict_ensemble(trained_models,X_test,y_test):
    y_pred=trained_models[0].predict(X_test)
    models_name=trained_models[0].models_name
    return y_test,y_pred,models_name

def models_load(options,n_features):
    models_array,algorithms_name=get_model(options,n_features)
    datasets_name=options.datasets
    trained_models_array=[]
    for i in range(0,len(models_array)):
        model_=ModelOperation()
        trained_model=model_.load(algorithms_name[i],datasets_name,models_array[i])
        trained_models_array.append(trained_model)
    return trained_models_array

def print_results(datasets_name,models_name,y_test,y_pred_arr,label):
    headers = ['datasets','algorithm','accuracy', 'f1', 'precision', 'recall','roc_auc']
    rows=[]
    for i in range(0,len(models_name)):
        y_pred=np.argmax(y_pred_arr[i], axis=1)
        result=get_class_scores(y_test, y_pred)
        row=list(result)
        row.insert(0,models_name[i])
        row.insert(0,datasets_name)
        rows.append(row)
    rows_pandas=DataFrame(rows)
    rows_pandas.columns=headers
    rows_pandas.to_csv('experiments/plot/'+datasets_name+'/'+label+'.csv',index=False)
    # plot_roc(models_name,y_test,y_pred_arr)
    print(label)
    print(tabulate(rows, headers=headers))
    return rows_pandas


def print_results_ensemble(datasets_name,models_name,y_test,y_pred_arr,label):
    headers = ['datasets','algorithm','accuracy', 'f1', 'precision', 'recall','roc_auc']
    rows=[]
    for i in range(0,len(models_name)):
        y_pred=np.argmax(y_pred_arr[i], axis=1)
        result=get_class_scores(y_test, y_pred)
        row=list(result)
        row.insert(0,models_name[i])
        row.insert(0,datasets_name)
        rows.append(row)
    rows_pandas=DataFrame(rows)
    rows_pandas.columns=headers
    rows_pandas.to_csv('experiments/plot/'+datasets_name+'/'+label+'.csv',index=False)
    # plot_roc(models_name,y_test,y_pred_arr)
    print(label)
    print(tabulate(rows, headers=headers))

def print_ensemble_results(datasets_name,models_name,y_test,y_pred_arr,label):
    headers = ['datasets','algorithm','accuracy', 'f1', 'precision', 'recall','roc_auc']
    rows=[]
    for i in range(0,len(models_name)):
        result=get_class_scores(y_test, y_pred_arr[i])
        row=list(result)
        row.insert(0,models_name[i])
        row.insert(0,datasets_name)
        rows.append(row)
    rows_pandas=DataFrame(rows)
    rows_pandas.columns=headers
    plot_roc(models_name,y_test,y_pred_arr)
    print(label)
    print(tabulate(rows, headers=headers))

def models_and_ensemble_predict(trained_models,models_name,X_val,y_val,X_test,y_test):
    y_test,y_pred_arr=models_predict(trained_models,X_test,y_test)
    ensembles_name,y_test,y_pred_ens_arr=EnsembleClassifier(X_val,y_val,X_test,y_test,trained_models)
    models_name=models_name+ensembles_name
    y_pred_arr.extend(y_pred_ens_arr)
    return models_name,y_test,y_pred_arr

def train_val_test_split(X,y, ratio_train, ratio_test, ratio_val):
    X_train,X_middle,y_train,y_middle = train_test_split(X,y, stratify=y,train_size=ratio_train, test_size=ratio_test + ratio_val,random_state=42)
    ratio = ratio_val/(1-ratio_train)
    X_test,X_val,y_test,y_val = train_test_split(X_middle,y_middle,stratify=y_middle,test_size=ratio,random_state=42)
    return X_train,y_train,X_val,y_val,X_test,y_test